"""工具的格式定义。"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Tool:
    """一个可以被模型调用的工具。

    params 是 JSON Schema，会原样成为请求里的 input_schema，留空表示这个工具不收参数。
    fn 收到的是模型给出的关键字参数，返回值由 ToolRegistry 包成 ToolResult 交回模型。
    """

    name: str
    fn: Callable[..., Any]
    description: str | None = None
    params: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("工具名不能为空")
        if not callable(self.fn):
            raise TypeError(f"工具 {self.name!r} 的 fn 必须可调用")

    def to_anthropic_schema(self) -> dict[str, Any]:
        """转成 Messages API 的 tools 条目。description 为空时不带这个键，免得发 null 过去。"""
        schema: dict[str, Any] = {"name": self.name}
        if self.description:
            schema["description"] = self.description
        schema["input_schema"] = self.params or {
            "type": "object",
            "properties": {},
            "required": [],
        }
        return schema


@dataclass
class ToolResult:
    """一次工具调用的结果。

    id 取自发起这次调用的 ToolCall.id，回填时模型靠它把结果和调用配对，不能错。
    name 只是本地元信息，给日志和 trace 看，协议里用不到。
    result 保留工具返回的原值，序列化推迟到出网的那一刻，这样 trace 里还是结构化的数据。
    """

    id: str
    name: str
    result: Any = None
    is_error: bool = False

    def to_anthropic_block(self) -> dict[str, Any]:
        """转成可以直接回填进 messages 的 tool_result block。"""
        content = self.result
        if not isinstance(content, str):
            content = json.dumps(content, ensure_ascii=False, default=str)
        block: dict[str, Any] = {
            "type": "tool_result",
            "tool_use_id": self.id,
            "content": content,
        }
        if self.is_error:
            block["is_error"] = True
        return block
