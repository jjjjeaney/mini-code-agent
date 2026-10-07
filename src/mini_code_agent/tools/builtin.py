"""内置工具的具体实现。

这里只放普通函数：能直接单测，不用过模型、也不用先建一个 registry。
包装成 Tool 是 registry.default_tools() 的活。
"""

from __future__ import annotations

from typing import Any

ECHO_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"text": {"type": "string", "description": "要打印到终端的文本"}},
    "required": ["text"],
}


def echo(text: str) -> str:
    """把文本打到终端，并回一句确认给模型。"""
    print(text)
    return f"已在终端打印：{text}"
