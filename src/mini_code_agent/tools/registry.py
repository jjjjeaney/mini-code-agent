"""工具注册表：登记工具，并把模型发起的调用派发出去。"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from mini_code_agent.providers.base import ToolCall
from mini_code_agent.tools.base import Tool, ToolResult


class ToolRegistry:
    def __init__(self, tools: Iterable[Tool] | None = None) -> None:
        self.tools: dict[str, Tool] = {}
        for tool in tools or ():
            self.register(tool)

    def register(self, tool: Tool) -> Tool:
        """注册一个工具。重名直接报错，不让后注册的悄悄顶掉先注册的。"""
        if tool.name in self.tools:
            raise ValueError(f"工具 {tool.name!r} 已经注册过了")
        self.tools[tool.name] = tool
        return tool

    def list(self) -> list[Tool]:
        """按注册顺序列出所有工具。"""
        return list(self.tools.values())

    def run(self, name: str, arguments: dict[str, Any] | None = None) -> Any:
        """按名字调用一个工具，工具自身的异常照常往上抛。"""
        tool = self.tools.get(name)
        if tool is None:
            raise KeyError(f"没有注册名为 {name!r} 的工具")
        return tool.fn(**(arguments or {}))

    def schemas(self) -> list[dict[str, Any]]:
        """传给 provider.complete(tools=...) 的列表。"""
        return [tool.to_anthropic_schema() for tool in self.tools.values()]

    def dispatch(self, call: ToolCall) -> ToolResult:
        """执行一次模型发起的调用，返回可以直接回填进对话的 ToolResult。

        和 run 的区别是这里不往外抛异常：工具失败要作为结果交回模型让它自己改，
        否则一次工具报错就会终结整轮对话。
        """
        try:
            result = self.run(call.name, call.arguments)
        except Exception as exc:
            return ToolResult(
                id=call.id,
                name=call.name,
                result=f"{type(exc).__name__}: {exc}",
                is_error=True,
            )
        return ToolResult(id=call.id, name=call.name, result=result)
