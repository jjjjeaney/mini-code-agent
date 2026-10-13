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

    # 这一发的用量。默认 0 是为了让假 provider（测试里那一堆）不用管它，真后端一定会填。
    #
    # 两个字段**不重叠**：input_tokens 是没走缓存的那部分，cache_read_input_tokens 是命中的
    # 那部分，所以这次请求的总输入 = 两者相加。这是 Anthropic 的 usage 语义，而打的后端
    # （DeepSeek）是自动前缀缓存——命中多少只看后一个数，请求里不用传任何东西。
    input_tokens: int = 0
    cache_read_input_tokens: int = 0


class Provider(Protocol):
    """agent 需要的最小后端能力：发一次请求，拿回归一化后的响应。"""

    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        system: str | None = None,
    ) -> ModelResponse: ...
