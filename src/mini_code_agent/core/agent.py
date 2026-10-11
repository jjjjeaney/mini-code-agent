"""agent 主循环。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mini_code_agent.console import console
from mini_code_agent.permissions import EDIT_TOOLS, PermissionRequest, decide_permission
from mini_code_agent.prompts import NullPrompter, Prompter
from mini_code_agent.providers.base import Provider, ToolCall
from mini_code_agent.tools import ToolContext, ToolRegistry, ToolResult, default_tools


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


def _ask(prompter: Prompter, request: PermissionRequest) -> bool:
    """真的去问一次用户。

    问**哪一种**由这里定：改文件的给他看改动、别的看参数（bash 也走这条，prompter 认得
    工具名，自己会去取那行命令）。渲染成什么样是 prompter 的事，但"该问哪一种"只有 harness
    知道——它手上才有工具名。
    """
    if request.tool_name in EDIT_TOOLS:
        return prompter.confirm_edit(request)
    return prompter.confirm_tool(request)


def _gate(
    call: ToolCall, prompter: Prompter, allow_shell: bool, permission_mode: str
) -> ToolResult | None:
    """权限闸门：放行返回 None，不放行则返回要回给模型的那条结果。

    拒绝走**结果**不走异常：模型收到"用户不同意"才能换个做法，抛异常是把一次正常的人机
    协商变成崩溃。也不让它原样重试——那只会再被拒一次。
    """
    request = PermissionRequest(tool_name=call.name, tool_args=call.arguments)
    decision = decide_permission(request, allow_shell, permission_mode)

    blocked: str | None = None
    if decision.decision == "deny":
        blocked = f"这次调用没有放行：{decision.message}"
    elif decision.decision == "ask" and not _ask(prompter, request):
        blocked = f"用户拒绝了这次调用（{decision.message}）"

    if blocked is None:
        return None
    return ToolResult(
        id=call.id,
        name=call.name,
        result=f"{blocked}。换个不需要它的做法，别原样重试。",
        is_error=True,
    )


def runagent(
    prompt: str,
    provider: Provider,
    registry: ToolRegistry | None = None,
    system: str | None = None,
    max_steps: int = 8,
    cwd: Path | None = None,
    allow_shell: bool = False,
    permission_mode: str = "default",
    prompter: Prompter | None = None,
) -> AgentResult:
    """跑一轮对话，模型要求用工具时自动执行并把结果交回去。

    max_steps 限制模型调用次数，用满还没等到纯文本回复时 final 为 None。
    trace 是一串渲染好的文本，按发生顺序记下每次工具调用、它的结果、以及最后那条回复，
    专门用来打印给人看。
    不传 registry 就用 default_tools() 装好的那套内置工具。

    permission_mode 是权限档位（三档见 permissions.py），决定改文件的工具要不要问；
    prompter 是"问用户"的方式，不传就用一律拒绝的那个——非交互的跑法不该挂在提示符上。
    """
    registry = registry if registry is not None else default_tools()
    tools = registry.schemas()
    messages: list[dict[str, Any]] = [{"role": "user", "content": prompt}]
    trace: list[str] = []
    final: str | None = None
    # prompter 不放进 ctx：问用户不是工具的能力，确认是这一层在调工具之前发的。
    prompter = prompter if prompter is not None else NullPrompter()
    ctx = ToolContext(cwd or Path.cwd(), allow_shell=allow_shell)
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
            console.print(_render_call(call), style="cyan", markup=False)

            # 闸门搁在派发之前：判了不放行，就不该碰到那个工具。
            result = _gate(call, prompter, allow_shell, permission_mode)
            if result is None:
                result = registry.dispatch(ctx, call)

            trace.append(_render_result(result))
            console.print(
                _render_result(result),
                style="red" if result.is_error else "green",
                markup=False,
            )
            results.append(result)
        
        
        # 同一条回复里可能有多个 tool_use，对应的 tool_result 必须合成一条 user 消息发回去。
        messages.append({"role": "user", "content": [r.to_anthropic_block() for r in results]})
    if final is None:
        messages.append({"role": "user", "content": "调用工具次数达到上限，不要再调用工具了，根据已有结果直接输出结果"})
        response = provider.complete(messages, system=system)
        final = response.text
        trace.append(final)
    return AgentResult(final=final, trace=trace, messages=messages)
