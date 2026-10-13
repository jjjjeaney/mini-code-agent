"""AnthropicProvider 的离线单元测试：换成假客户端，一个请求都不发。

**为什么不并进 test_anthropic_provider.py**：那边全是真实接口用例，缺 key 就整组跳过；
把这些也放进去等于让它们跟着一起被跳掉。两边分开，这边永远会跑。
"""

from __future__ import annotations

from types import SimpleNamespace

import httpx2
import pytest
from anthropic import (
    APIConnectionError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    ConflictError,
    InternalServerError,
    NotFoundError,
    PermissionDeniedError,
    RateLimitError,
    UnprocessableEntityError,
)

from mini_code_agent.providers.anthropic import MAX_OUTPUT_TOKENS_CEILING, AnthropicProvider
from mini_code_agent.providers.base import ProviderError, TransientProviderError


def a_request() -> httpx2.Request:
    return httpx2.Request("POST", "https://example.invalid/v1/messages")


def an_error(cls, status: int, message: str = "服务端说不行"):
    """造一个**真的** SDK 异常对象（不是打桩）——SDK 用的是 httpx2，不是更常见的那个 httpx。"""
    return cls(message, response=httpx2.Response(status, request=a_request()), body=None)


def a_response(
    stop_reason: str = "end_turn",
    text: str = "好",
    *,
    input_tokens: int | None = 0,
    output_tokens: int | None = 0,
    cached: int | None = 0,
    written: int | None = 0,
) -> SimpleNamespace:
    """一次响应，形状只保留 provider 会读的那几样。

    用量那四个字段默认给 None 也行——SDK 里的类型就是 `int | None`，provider 得把它兜成 0。
    """
    return SimpleNamespace(
        content=[SimpleNamespace(type="text", text=text)],
        stop_reason=stop_reason,
        usage=SimpleNamespace(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_read_input_tokens=cached,
            cache_creation_input_tokens=written,
        ),
    )


class FakeMessages:
    def __init__(self, responses) -> None:
        self.responses = list(responses)
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(dict(kwargs))
        item = self.responses.pop(0)
        if isinstance(item, BaseException):
            raise item  # 剧本里排的是异常就抛出去
        return item


class FakeClient:
    def __init__(self, responses) -> None:
        self.messages = FakeMessages(responses)


def a_provider(*responses, max_tokens: int = 1024) -> AnthropicProvider:
    """构造一个不会联网的 provider。key 和 base_url 随便给——构造函数本身不发请求。"""
    provider = AnthropicProvider(
        api_key="x", base_url="https://example.invalid", max_tokens=max_tokens
    )
    provider.client = FakeClient(responses)
    return provider


class TestOutputLimitEscalation:
    def test_a_normal_answer_is_asked_for_once(self):
        provider = a_provider(a_response())

        provider.complete([{"role": "user", "content": "问"}])

        assert [call["max_tokens"] for call in provider.client.messages.calls] == [1024]

    def test_a_truncated_answer_is_asked_for_again_with_more_room(self):
        provider = a_provider(a_response("max_tokens", "半截"), a_response("end_turn", "完整"))

        result = provider.complete([{"role": "user", "content": "写篇长的"}])

        assert [call["max_tokens"] for call in provider.client.messages.calls] == [
            1024,
            MAX_OUTPUT_TOKENS_CEILING,  # 直接跳到天花板，不是一级一级爬
        ]
        assert result.text == "完整"  # 交回去的是重试那一次
        assert result.stop_reason == "end_turn"

    def test_the_retry_sends_exactly_the_same_request(self):
        """只动 max_tokens——消息、工具、system 一个字节都不改。"""
        provider = a_provider(a_response("max_tokens"), a_response())

        provider.complete(
            [{"role": "user", "content": "问"}], tools=[{"name": "t"}], system="S"
        )

        first, second = provider.client.messages.calls
        assert {k: v for k, v in second.items() if k != "max_tokens"} == {
            k: v for k, v in first.items() if k != "max_tokens"
        }

    def test_it_does_not_retry_when_it_is_already_at_the_ceiling(self):
        """到顶了还是被切——再问一次也是一样的结果，不如原样交回去，别白花一次生成。"""
        provider = a_provider(a_response("max_tokens"), max_tokens=MAX_OUTPUT_TOKENS_CEILING)

        result = provider.complete([{"role": "user", "content": "问"}])

        assert len(provider.client.messages.calls) == 1
        assert result.stop_reason == "max_tokens"  # 上层至少看得出这份是被切过的

    def test_a_configured_max_tokens_above_the_ceiling_is_left_alone(self):
        provider = a_provider(a_response("max_tokens"), max_tokens=32_000)

        provider.complete([{"role": "user", "content": "问"}])

        assert [call["max_tokens"] for call in provider.client.messages.calls] == [32_000]

    def test_escalating_does_not_change_the_configured_default(self):
        """抬上去是**这一次**的事，不记在实例上——那是修补，不是新配置。"""
        provider = a_provider(a_response("max_tokens"), a_response(), a_response())

        provider.complete([{"role": "user", "content": "一"}])
        provider.complete([{"role": "user", "content": "二"}])

        assert [call["max_tokens"] for call in provider.client.messages.calls] == [
            1024,
            MAX_OUTPUT_TOKENS_CEILING,
            1024,  # 第二次又从配置的那个默认值起步
        ]


class TestUsage:
    """用量原样带出去——这一层不判断"省没省钱"，只把数交上去（主循环会打一行出来）。"""

    def test_it_carries_all_four_counts_out(self):
        provider = a_provider(
            a_response(input_tokens=120, output_tokens=34, cached=880, written=56)
        )

        result = provider.complete([{"role": "user", "content": "问"}])

        assert result.input_tokens == 120
        assert result.output_tokens == 34
        assert result.cache_read_input_tokens == 880
        assert result.cache_creation_input_tokens == 56

    def test_a_null_count_becomes_zero(self):
        """SDK 里那几个字段是 `int | None`，None 得落到 0——不然上层一相加就炸。"""
        provider = a_provider(
            a_response(input_tokens=None, output_tokens=None, cached=None, written=None)
        )

        result = provider.complete([{"role": "user", "content": "问"}])

        assert result.input_tokens == 0
        assert result.output_tokens == 0
        assert result.cache_read_input_tokens == 0
        assert result.cache_creation_input_tokens == 0


class TestErrors:
    """SDK 的异常翻成**人话**，再以项目自己的 ProviderError 抛上去。

    上游（agent、CLI）不认识 anthropic 的类名，换后端时也不该改——所以翻译只在这一层做。
    """

    def test_a_bad_key_says_which_variable_to_check(self):
        provider = a_provider(an_error(AuthenticationError, 401))

        with pytest.raises(ProviderError, match="DEEPSEEK_API_KEY"):
            provider.complete([{"role": "user", "content": "问"}])

    def test_a_rate_limit_says_to_wait(self):
        provider = a_provider(an_error(RateLimitError, 429))

        with pytest.raises(ProviderError, match="429"):
            provider.complete([{"role": "user", "content": "问"}])

    def test_a_server_error_names_the_status(self):
        provider = a_provider(an_error(InternalServerError, 503))

        with pytest.raises(ProviderError, match="503"):
            provider.complete([{"role": "user", "content": "问"}])

    def test_a_bad_request_repeats_what_the_server_said(self):
        """400 里服务端那句话往往就是唯一有用的线索（到底哪儿不合法），得照抄给用户。"""
        provider = a_provider(an_error(BadRequestError, 400, "max_tokens 超出上限"))

        with pytest.raises(ProviderError, match="max_tokens 超出上限"):
            provider.complete([{"role": "user", "content": "问"}])

    def test_a_timeout_is_not_mixed_up_with_a_connection_problem(self):
        """超时是连接错的**子类**——顺序搞反的话两者会合成同一句话，而"对面太慢"和"网络不通"
        是两件事。"""
        provider = a_provider(APITimeoutError(request=a_request()))

        with pytest.raises(TransientProviderError, match="超时"):
            provider.complete([{"role": "user", "content": "问"}])

    def test_a_connection_problem_names_the_network(self):
        provider = a_provider(APIConnectionError(request=a_request()))

        with pytest.raises(TransientProviderError, match="连不上"):
            provider.complete([{"role": "user", "content": "问"}])

    def test_the_ones_worth_retrying_are_marked_transient(self):
        """该不该重试是**类型**，不是消息文本——认字会漂，类型不会。

        这份名单就是 SDK 的 `_should_retry`（408 / 409 / 429 / ≥500、加上连接类）。**409 和
        503 特意放进来**：它们没有专门的类，只靠状态码认——少认一个的后果是"本来能救的没救"。
        """
        retryable = (
            an_error(RateLimitError, 429),
            an_error(InternalServerError, 503),
            an_error(ConflictError, 409),
            APITimeoutError(request=a_request()),
            APIConnectionError(request=a_request()),
        )
        for exc in retryable:
            provider = a_provider(exc)
            with pytest.raises(TransientProviderError):
                provider.complete([{"role": "user", "content": "问"}])

    def test_the_ones_that_will_never_work_are_not_transient(self):
        """401 再试一百次还是 401——只该尽快把话给用户，一个字都别多等。"""
        hopeless = (
            an_error(AuthenticationError, 401),
            an_error(PermissionDeniedError, 403),
            an_error(NotFoundError, 404),
            an_error(BadRequestError, 400),
            an_error(UnprocessableEntityError, 422),
        )
        for exc in hopeless:
            provider = a_provider(exc)
            with pytest.raises(ProviderError) as caught:
                provider.complete([{"role": "user", "content": "问"}])
            assert not isinstance(caught.value, TransientProviderError)

    def test_it_keeps_the_original_error_attached(self):
        """翻成人话不等于把现场丢了——`from exc` 让 traceback 还指着根因。"""
        provider = a_provider(an_error(NotFoundError, 404))

        with pytest.raises(ProviderError) as caught:
            provider.complete([{"role": "user", "content": "问"}])

        assert isinstance(caught.value.__cause__, NotFoundError)

    def test_our_own_bugs_are_not_swallowed(self):
        """只翻译 SDK 的异常。自己代码里的 bug 该带着 traceback 崩掉，不该被这段吞成人话。"""
        provider = a_provider(RuntimeError("我自己写错了"))

        with pytest.raises(RuntimeError, match="我自己写错了"):
            provider.complete([{"role": "user", "content": "问"}])
