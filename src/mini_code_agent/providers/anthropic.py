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

# 输出被截断时，最多把 max_tokens 抬到这里。
# 16384 是**实测**过这个端点会收的值：4096 / 8192 / 16384 都正常返回，而 65536 起 SDK
# 自己就拒绝发非流式请求了（"Streaming is required for operations that may take longer
# than 10 minutes"）。所以再往上就不是换个数字的事，得改成流式——那是另一摊。
MAX_OUTPUT_TOKENS_CEILING = 16384

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
        ceiling: int = MAX_OUTPUT_TOKENS_CEILING,
    ) -> None:
        self.model = model
        self.max_tokens = max_tokens
        self.ceiling = ceiling
        self.client = Anthropic(
            api_key=api_key or os.getenv("DEEPSEEK_API_KEY"),
            base_url=base_url or os.getenv("DEEPSEEK_ANTHROPIC_BASE_URL"),
        )

    def create(self, kwargs: dict[str, Any]) -> Any:
        """发一次请求；输出撞上上限就抬 max_tokens 再发一次。

        **在这层做、不往 agent 递，是有意的**：这是"这次请求向模型要多少生成空间"的事，不是
        对话的事——agent 手上只有归一化之后的 ModelResponse，本来也看不见这些。顺带保住了
        Provider 协议：一行不用动，别的后端想各自实现也行。

        **只重试一次，而且直接跳到天花板**：

        - `max_tokens` 是**上限不是目标**——写大了不额外花钱（只按真正生成的那些计费）。所以
          第一次就跳到能要到的最大值，比 1024/2048/4096 一级一级爬便宜得多：爬的每一级都要
          为一次马上就被丢掉的生成付钱。实测这个端点的天花板是 16384（见上面的常量）。
        - 只重试一次，所以"循环"是有界的，不会转个没完。每次 `+1024` 那种写法在模型一直撞
          上限（或者天花板本身比它需要的还低）时会一直转，而且越转越贵。

        抬上去这件事**不记在实例上**：这是对一次意外截断的修补，不是新配置。要是每轮都撞，
        那说明这个默认值该抬了——那是人该做的决定，不该由 provider 悄悄替了。
        """
        response = self.client.messages.create(**kwargs)
        if response.stop_reason != "max_tokens" or kwargs["max_tokens"] >= self.ceiling:
            # 没撞上限；或者已经要到天花板了还是被切——那就到此为止，原样交回去。
            # 上层至少还能从 stop_reason 看出"这份是被切过的"。
            return response
        kwargs["max_tokens"] = self.ceiling
        return self.client.messages.create(**kwargs)

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
        response = self.create(kwargs)
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
