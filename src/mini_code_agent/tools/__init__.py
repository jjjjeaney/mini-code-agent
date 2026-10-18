"""工具层：工具协议、注册表与具体工具实现。

只依赖标准库和 providers.base 里的中立类型，不 import cli / commands，也不碰 agent 的内部状态。
**唯一的例外是 `ForkContext`**：fork 子 agent 必须复用父 agent 的 system、工具表和历史，那几样
只能由 harness 递进来。它被圈在一个小对象里，只有 `core/subagent.py` 的 fork 分支读。
"""

from mini_code_agent.tools.base import (
    ForkContext,
    ReadFileState,
    Tool,
    ToolContext,
    ToolResult,
)
from mini_code_agent.tools.registry import ToolRegistry, default_tools

__all__ = [
    "ForkContext",
    "ReadFileState",
    "Tool",
    "ToolContext",
    "ToolRegistry",
    "ToolResult",
    "default_tools",
]
