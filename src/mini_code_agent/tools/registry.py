"""工具注册表：登记工具，并把模型发起的调用派发出去。"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from mini_code_agent.providers.base import ToolCall
from mini_code_agent.tools.base import Tool, ToolContext, ToolResult
from mini_code_agent.tools.builtin import ECHO_SCHEMA, echo
from mini_code_agent.tools.files import (
    GLOB_SCHEMA,
    GREP_SCHEMA,
    LIST_FILES_SCHEMA,
    PROJECT_TREE_SCHEMA,
    READ_FILE_SCHEMA,
    WRITE_FILE_SCHEMA,
    glob,
    grep,
    list_files,
    project_tree,
    read_file,
    write_file,
)
from mini_code_agent.tools.web import WEB_SEARCH_SCHEMA, web_search


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

    def run(
        self, ctx: ToolContext, name: str, arguments: dict[str, Any] | None = None
    ) -> Any:
        """按名字调用一个工具。工具自身的异常、以及缺了必填参数，都照常往上抛。"""
        tool = self.tools.get(name)
        if tool is None:
            raise KeyError(f"没有注册名为 {name!r} 的工具")

        # 按工具自己的 schema 校验必填参数——校验的就是发给模型的那份契约。
        # 少了参数在这里就拦住，别让工具拿着 None 悄悄跑完：那样模型会以为成功了，
        # 而它本来能收到一句"你少传了 text"然后自己改对。
        args = arguments or {}
        missing = [key for key in tool.params.get("required", []) if key not in args]
        if missing:
            raise ValueError(f"调用 {tool.name} 缺少必填参数：{'、'.join(missing)}")

        return tool.fn(args, ctx)

    def schemas(self) -> list[dict[str, Any]]:
        """传给 provider.complete(tools=...) 的列表。"""
        return [tool.to_anthropic_schema() for tool in self.tools.values()]

    def dispatch(self, ctx: ToolContext, call: ToolCall) -> ToolResult:
        """执行一次模型发起的调用，返回可以直接回填进对话的 ToolResult。

        和 run 的区别是这里不往外抛异常：工具失败要作为结果交回模型让它自己改，
        否则一次工具报错就会终结整轮对话。
        """
        try:
            result = self.run(ctx, call.name, call.arguments)
        except Exception as exc:
            return ToolResult(
                id=call.id,
                name=call.name,
                result=f"{type(exc).__name__}: {exc}",
                is_error=True,
            )
        return ToolResult(id=call.id, name=call.name, result=result)


def default_tools() -> ToolRegistry:
    """装配一份可以直接用的工具集。

    每次调用都返回新实例：调用方想往自己的那份里再注册工具时，
    不会污染到别人手里那份。
    """
    return ToolRegistry(
        [
            Tool(
                name="echo",
                description="把一段文本打印到终端。需要让用户看到某段内容时用它。",
                fn=echo,
                params=ECHO_SCHEMA,
            ),
            Tool(
                name="list_files",
                description=(
                    "列出某个目录下面一层的文件和子目录，不往下钻。"
                    "想知道某个目录里有什么时用它。"
                ),
                fn=list_files,
                params=LIST_FILES_SCHEMA,
            ),
            Tool(
                name="project_tree",
                description=(
                    "把整个项目画成一棵树，用于一眼看清整体结构。"
                    "想知道项目怎么组织、有哪些模块时用它。"
                ),
                fn=project_tree,
                params=PROJECT_TREE_SCHEMA,
            ),
            Tool(
                name="read_file",
                description=(
                    "读一个文本文件的内容。文件很长时可以只读一段："
                    "start_line / end_line 指定行号范围，从 1 数起。"
                ),
                fn=read_file,
                params=READ_FILE_SCHEMA,
            ),
            Tool(
                name="write_file",
                description=(
                    "写一个文件：不存在就新建，存在就用给的内容整个替换。"
                    "传的是**整篇**内容，不是片段；改局部用 edit_file。"
                    "覆盖已存在的文件前必须先读过它。"
                ),
                fn=write_file,
                params=WRITE_FILE_SCHEMA,
            ),
            Tool(
                name="glob",
                description=(
                    "按文件名或路径模式递归查找文件和目录。"
                    "想知道某类文件都在哪儿时用它，比逐个目录 list_files 快；"
                    "返回的路径可以直接拿去读。"
                ),
                fn=glob,
                params=GLOB_SCHEMA,
            ),
            Tool(
                name="grep",
                description=(
                    "按内容搜索，返回每条匹配的文件、行号和那一行。"
                    "找某个函数、变量、字符串在哪儿定义或使用时用它，比逐个文件读快。"
                ),
                fn=grep,
                params=GREP_SCHEMA,
            ),
            Tool(
                name="web_search",
                description=(
                    "联网搜索。遇到你不确定、可能已过时、或需要查证的事实时用它，"
                    "比如近期新闻、最新版本号、人物近况。返回若干条结果的标题、链接和摘要。"
                ),
                fn=web_search,
                params=WEB_SEARCH_SCHEMA,
            ),
        ]
    )

