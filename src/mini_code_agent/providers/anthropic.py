"""基于 Anthropic Messages API 的 provider。

默认指向 DeepSeek 的 Anthropic 兼容端点，端点和密钥都从环境变量读，换后端不用改代码。
"""

from __future__ import annotations

import os
from collections.abc import Callable
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

try:  # SDK 的 HTTP 实现；正常安装时一定在，不在就少认一类异常
    import httpx2

    _TRANSPORT_ERRORS: tuple[type[BaseException], ...] = (httpx2.TransportError,)
except ImportError:  # pragma: no cover
    _TRANSPORT_ERRORS = ()

from mini_code_agent.providers.base import (
    ModelResponse,
    ProviderError,
    ToolCall,
    TransientProviderError,
)

load_dotenv()

DEFAULT_MODEL = "deepseek-v4-flash"

# 输出被截断时，最多把 max_tokens 抬到这里；流式路径也直接从这里起步。
# 16384 是**实测**过这个端点会收的值：4096 / 8192 / 16384 都正常返回，而 65536 起 SDK
# 自己就拒绝发非流式请求了（"Streaming is required for operations that may take longer
# than 10 minutes"）。流式能绕过那条客户端限制，但 16384 以上这个端点收不收还没验证过，
# 所以先停在这儿。
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


def _stream_failure(error: ProviderError, shown: bool) -> ProviderError:
    """把流式失败收成一个要往上层抛的异常。

    **显示过正文之后，瞬时错误也不能再自动重试**：上层 `ask()` 靠异常类型决定要不要补一次，
    补一次就是把已经打出来的字再打一遍。所以这里把 `TransientProviderError` 降级成普通
    `ProviderError`，并让消息告诉用户该做什么。
    """
    if shown and isinstance(error, TransientProviderError):
        return ProviderError("输出到一半连接断了；已经显示的内容可能不完整，再发一次。")
    return error


def _normalize(response: Any) -> ModelResponse:
    """把一次模型响应归一化成项目自己的类型。

    流式的 `get_final_message()` 和非流式的 `create()` 返回的结构一致，所以两条路径共用
    这一份——终端怎么显示是一回事，进历史、给工具用的必须是同一个结果。
    """
    text_parts = []
    tool_calls = []
    assistant_content = []
    for block in response.content:
        assistant_content.append(_content_block_to_dict(block))
        if block.type == "text":
            text_parts.append(block.text)
        if block.type == "tool_use":
            tool_calls.append(
                ToolCall(
                    id=block.id,
                    name=block.name,
                    arguments=block.input,
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
        output_tokens=usage.output_tokens or 0,
        cache_read_input_tokens=usage.cache_read_input_tokens or 0,
        cache_creation_input_tokens=usage.cache_creation_input_tokens or 0,
    )


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
        return _normalize(response)

    def stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        system: str | None = None,
        on_text: Callable[[str], None] | None = None,
    ) -> ModelResponse:
        """发一次流式请求：正文边生成边交给 `on_text`，结果仍然以完整消息为准。

        **终端显示和程序使用是两回事**：`on_text` 只负责把 delta 写出去；历史回填、工具
        调用、用量、stop_reason 全部来自 `get_final_message()`，和非流式路径共用 `_normalize`。

        推理过程不显示：`thinking_delta` 直接跳过；thinking block 仍然留在最终消息里，由
        `_normalize` 原样带进 `assistant_content`，续接对话时要用。

        几条定下来的：

        - `max_tokens` 一开始就要到天花板。非流式撞顶还能抬上去重发，流式不行——文字已经
          显示在屏幕上了，重发只会重复一遍。上限不是目标，写大了不额外花钱。
        - **必须把流读完**。只消费一部分再取 final，没显示的部分会被 SDK 静默读掉，用户
          看到的和程序拿到的就对不上了。
        - **截断不抛异常**。连接提前关掉时 final 的 `stop_reason` 是 `None`，得自己当失败。
        - **显示过文字之后不再自动重试**。上层 `ask()` 靠异常类型决定要不要补一次；补一次
          就是把已经打出来的字再打一遍。所以这种情况把瞬时错误降级成普通的 `ProviderError`。
        """
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max(self.max_tokens, self.ceiling),
        }
        if tools:
            kwargs["tools"] = tools
        if system:
            kwargs["system"] = system

        shown = False
        try:
            with self.client.messages.stream(**kwargs) as live:
                # 遍历原始事件而不是 `text_stream`：后者只吐正文，thinking 拿不到。派生事件
                # （text / thinking / input_json）也在这条迭代里，但它们的 type 不是
                # content_block_delta，下面会跳过。
                for event in live:
                    if event.type != "content_block_delta":
                        continue
                    delta = event.delta
                    # thinking delta 不显示；thinking block 仍然在最终消息里，由 _normalize
                    # 原样带回 assistant_content，续接对话时要用。
                    if delta.type == "text_delta" and on_text is not None:
                        shown = True
                        on_text(delta.text)
                try:
                    response = live.get_final_message()
                except AssertionError as exc:
                    # SDK 在一个事件都没读到时断言 final snapshot 存在——这不是我们的 bug，
                    # 是连接在返回任何内容之前就断了。
                    raise _stream_failure(
                        TransientProviderError("模型连接在返回任何内容前断开了"), shown
                    ) from exc
        except APIError as exc:
            raise _stream_failure(_translate(exc), shown) from exc
        except _TRANSPORT_ERRORS as exc:
            # 流开始之后的传输错误不经过 SDK 的翻译（重试循环只包住建立连接那一步）。
            raise _stream_failure(
                TransientProviderError("模型连接在流式输出中途断了"), shown
            ) from exc

        if response.stop_reason is None:
            raise _stream_failure(
                TransientProviderError("模型的输出在结束前断开了"), shown
            )

        return _normalize(response)
