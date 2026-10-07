"""agent 主循环。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from mini_code_agent.providers.base import Provider, ToolCall
from mini_code_agent.tools import ToolRegistry, ToolResult


@dataclass
class AgentResult:
    final: str | None
    trace: list[str]
    messages: list[dict[str, Any]]


def _as_text(value: Any) -> str:
    """压成一行文本。非字符串走 JSON，这样一来换行会被转义，不会把流水账冲散。"""
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)


def _render_call(call: ToolCall) -> str:
    """渲染一次工具调用，例如 `→ list_files({"path": "."})`。"""
    return f"→ {call.name}({_as_text(call.arguments) if call.arguments else ''})"


def _render_result(result: ToolResult) -> str:
    """渲染一次工具结果，失败时用 ✗ 开头。"""
    mark = "✗" if result.is_error else "←"
    return f"{mark} {result.name}: {_as_text(result.result)}"


def runagent(
    prompt: str,
    provider: Provider,
    registry: ToolRegistry | None = None,
    system: str | None = None,
    max_steps: int = 8,
) -> AgentResult:
    """跑一轮对话，模型要求用工具时自动执行并把结果交回去。

    max_steps 限制模型调用次数，用满还没等到纯文本回复时 final 为 None。
    trace 是一串渲染好的文本，按发生顺序记下每次工具调用、它的结果、以及最后那条回复，
    专门用来打印给人看。
    """
    registry = registry if registry is not None else ToolRegistry()
    tools = registry.schemas()
    messages: list[dict[str, Any]] = [{"role": "user", "content": prompt}]
    trace: list[str] = []
    final: str | None = None

    for _ in range(max_steps):
        response = provider.complete(messages, tools=tools, system=system)

        if not response.tool_calls:
            final = response.text
            if final is not None:
                trace.append(final)
            break

        # assistant 这轮的内容要原样回填：tool_use 块藏在里面，模型靠它认领结果。
        messages.append({"role": "assistant", "content": response.assistant_content})

        results: list[ToolResult] = []
        for call in response.tool_calls:
            # 请求和它的结果挨着记，打印出来才是一条顺得下来的流水。
            trace.append(_render_call(call))
            result = registry.dispatch(call)
            trace.append(_render_result(result))
            results.append(result)

        # 同一条回复里可能有多个 tool_use，对应的 tool_result 必须合成一条 user 消息发回去。
        messages.append({"role": "user", "content": [r.to_anthropic_block() for r in results]})

    return AgentResult(final=final, trace=trace, messages=messages)
