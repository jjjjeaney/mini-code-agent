"""内置工具的具体实现。

这里只放普通函数：能直接单测，不用过模型、也不用先建一个 registry。
包装成 Tool 是 registry.default_tools() 的活。
"""

from __future__ import annotations

from typing import Any

from mini_code_agent.tools.base import ToolContext
from mini_code_agent.tools.safety import MAX_TOOL_CHARS, truncate_text

ECHO_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"text": {"type": "string", "description": "要打印到终端的文本"}},
    "required": ["text"],
}


def echo(kwargs: dict[str, Any], ctx: ToolContext) -> str:
    """把文本打到终端，并回一句确认给模型。

    回给模型的那句也过限额：真正的输出在终端上、用户看得见，回填进对话的这份只是确认，
    没必要把同一段长文本在上下文里再放一遍（那本来就在模型自己上一条回复里）。
    """
    text = kwargs["text"]
    print(text)
    return truncate_text(f"已在终端打印：{text}", MAX_TOOL_CHARS)
