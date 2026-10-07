"""LLM 后端的抽象接口，以及后端无关的响应类型。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass
class ToolCall:
    """模型发起的一次工具调用。"""

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class ModelResponse:
    """一次请求归一化之后的结果。

    assistant_content 是后端原始的 content blocks，续接对话时要原样回填，
    因为里面除了文字还有 tool_use（以及 DeepSeek 可能返回的 thinking）。
    """

    text: str | None = None
    tool_calls: list[ToolCall] | None = None
    assistant_content: list[dict[str, Any]] | None = None
    stop_reason: str | None = None


class Provider(Protocol):
    """agent 需要的最小后端能力：发一次请求，拿回归一化后的响应。"""

    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        system: str | None = None,
    ) -> ModelResponse: ...
