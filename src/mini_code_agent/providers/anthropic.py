"""基于 Anthropic Messages API 的 provider。

默认指向 DeepSeek 的 Anthropic 兼容端点，端点和密钥都从环境变量读，换后端不用改代码。
"""

from __future__ import annotations

import os
from typing import Any

from anthropic import Anthropic
from dotenv import load_dotenv

from mini_code_agent.providers.base import ModelResponse, ToolCall

load_dotenv()

DEFAULT_MODEL = "deepseek-v4-flash"

def _content_block_to_dict(block: Any) -> dict[str, Any]:
    # DeepSeek 的 Anthropic-compatible endpoint 可能返回 thinking block。
    # 后续请求必须把这些 assistant content blocks 原样传回去。
    if hasattr(block, "model_dump"):
        return block.model_dump(exclude_none=True)
    if hasattr(block, "dict"):
        return block.dict(exclude_none=True)
    data = {"type": block.type}
    for name in ("text", "id", "name", "input", "thinking", "signature"):
        if hasattr(block, name):
            data[name] = getattr(block, name)
    return data

class AnthropicProvider:
    def __init__(
        self,
        *,
        model: str = DEFAULT_MODEL,
        max_tokens: int = 1024,
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        self.model = model
        self.max_tokens = max_tokens
        self.client = Anthropic(
            api_key=api_key or os.getenv("DEEPSEEK_API_KEY"),
            base_url=base_url or os.getenv("DEEPSEEK_ANTHROPIC_BASE_URL"),
        )

    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        system: str | None = None,
    ) -> Any:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_tokens,
        }
        if tools:
            kwargs["tools"] = tools
        if system:
            kwargs["system"] = system
        response= self.client.messages.create(**kwargs)
        text_parts=[]
        tool_calls=[]
        assistant_content=[]
        for block in response.content:
            assistant_content.append(_content_block_to_dict(block))
            if block.type=='text':
                text_parts.append(block.text)
            if block.type=='tool_use':
                tool_calls.append(
                    ToolCall(
                        id=block.id,
                        name=block.name,
                        arguments=block.input
                    )
                )
        return ModelResponse(
            text="\n".join(text_parts) or None,
            tool_calls=tool_calls or None,
            assistant_content=assistant_content,
            stop_reason=response.stop_reason or "end_turn",
        )
