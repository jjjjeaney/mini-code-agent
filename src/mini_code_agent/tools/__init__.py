"""工具层：工具协议、注册表与具体工具实现。

只依赖标准库和 providers.base 里的中立类型，不 import cli / commands，也不碰 agent 的内部状态。
"""

from mini_code_agent.tools.base import Tool, ToolResult
from mini_code_agent.tools.registry import ToolRegistry

__all__ = ["Tool", "ToolRegistry", "ToolResult"]
