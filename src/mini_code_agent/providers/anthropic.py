"""基于 Anthropic Messages API 的 provider。

默认指向 DeepSeek 的 Anthropic 兼容端点，端点和密钥都从环境变量读，换后端不用改代码。
"""

from __future__ import annotations

import os
from typing import Any

from anthropic import (
    Anthropic,
    APIConnectionError,
    APIError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    NotFoundError,
    PermissionDeniedError,
    RateLimitError,
)
from dotenv import load_dotenv

from mini_code_agent.providers.base import (
    ModelResponse,
    ProviderError,
    ToolCall,
    TransientProviderError,
)

load_dotenv()

DEFAULT_MODEL = "deepseek-v4-flash"

# 输出被截断时，最多把 max_tokens 抬到这里。
# 16384 是**实测**过这个端点会收的值：4096 / 8192 / 16384 都正常返回，而 65536 起 SDK
# 自己就拒绝发非流式请求了（"Streaming is required for operations that may take longer
# than 10 minutes"）。所以再往上就不是换个数字的事，得改成流式——那是另一摊。
MAX_OUTPUT_TOKENS_CEILING = 16384

# SDK 会重试的状态码——照抄 `_should_retry` 里那几条（还有 ≥500，见下）。列在这儿是为了让
# "我们的判断"和"它的判断"对得上：它那 2 次重试失败之后才轮到我们决定要不要补一次，两边的
# 标准得是同一套，不然会漏掉本来能救的。
SDK_RETRY_STATUSES = (408, 409, 429)

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

def _translate(exc: APIError) -> ProviderError:
    """把 SDK 的异常翻成项目自己的异常：**类**表示该不该重试，**消息**是给用户看的那句话。

    分档只按"用户下一步该做什么"分，不按异常类型本身。"该不该重试"照抄 SDK 的 `_should_retry`
    （408 / 409 / 429 / ≥500，加上连接类），**不自立一套**——两套标准迟早对不上，而抄漏一个的
    后果是"本来能救的没救"。

    先认具体的类、再退回状态码：类名更准（超时是连接错的**子类**，只看父类会把两者混成一句）。

    **瞬时那几条的消息不写"该怎么办"**——因为我们马上就要替用户重试，这时候写"再发一次"自相
    矛盾；真到补的那一次也用不上、要告诉他了，那句描述本身也够明白（网络断了、对面 5xx）。
    反过来，**不重试的那几条一定要写清怎么办**（去查哪个变量、哪个地址），因为那才是他唯一
    能做的事。
    """
    status = getattr(exc, "status_code", None)

    # 先挑出"重试也没用"的——这几条得带着能照着做的建议。
    if isinstance(exc, AuthenticationError):
        return ProviderError("API key 不对或已经过期（401）。检查 .env 里的 DEEPSEEK_API_KEY。")
    if isinstance(exc, PermissionDeniedError):
        return ProviderError("这个 key 没有访问权限（403）。")
    if isinstance(exc, NotFoundError):
        return ProviderError(
            "模型名或端点不对（404）。检查 .env 里的模型名和 DEEPSEEK_ANTHROPIC_BASE_URL。"
        )
    if isinstance(exc, BadRequestError):
        # 400 里服务端那句话往往就是唯一有用的线索（到底哪儿不合法），照抄给用户。
        return ProviderError(
            f"模型服务拒绝了这次请求（400）：{getattr(exc, 'message', '') or exc}"
        )

    # 剩下的是"可以再试一次"的。
    if isinstance(exc, APITimeoutError):
        return TransientProviderError("模型响应太慢，超时了")
    if isinstance(exc, APIConnectionError):
        return TransientProviderError("连不上模型服务（网络断了，或者服务本身不可达）")
    if isinstance(exc, RateLimitError):
        return TransientProviderError("被限流了（429）——SDK 已经退避重试过两轮还是不行")
    if status in SDK_RETRY_STATUSES or (status is not None and status >= 500):
        return TransientProviderError(f"模型服务端出错（{status}），这是对面的事")

    return ProviderError(f"模型调用失败（{type(exc).__name__}）：{exc}")


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
        try:
            response = self.create(kwargs)
        except APIError as exc:
            # 翻译成项目自己的异常再往上抛：上游不认识 anthropic 的类名，换后端时也不该改。
            # （`APIError` 是 SDK 所有异常的共同父类，所以"没预料到的"也会被接住、翻成人话——
            # 不会漏一个出去把会话崩掉。反过来，我们自己代码里的 bug 不是 `APIError`，
            # 该带着 traceback 崩掉，不该被这段吞掉。）
            raise _translate(exc) from exc
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
        usage = response.usage
        return ModelResponse(
            text="\n".join(text_parts) or None,
            tool_calls=tool_calls or None,
            assistant_content=assistant_content,
            stop_reason=response.stop_reason or "end_turn",
            # 用量原样带出去，别在这儿算、也别在这儿判断——这一层不认识"省钱"这件事。
            # 谁想看谁看（主循环会打一行出来）。
            input_tokens=usage.input_tokens or 0,
            cache_read_input_tokens=usage.cache_read_input_tokens or 0,
        )
