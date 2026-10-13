"""AnthropicProvider 的离线单元测试：换成假客户端，一个请求都不发。

**为什么不并进 test_anthropic_provider.py**：那边全是真实接口用例，缺 key 就整组跳过；
把这些也放进去等于让它们跟着一起被跳掉。两边分开，这边永远会跑。
"""

from __future__ import annotations

from types import SimpleNamespace

from mini_code_agent.providers.anthropic import MAX_OUTPUT_TOKENS_CEILING, AnthropicProvider


def a_response(stop_reason: str = "end_turn", text: str = "好") -> SimpleNamespace:
    """一次响应，形状只保留 provider 会读的那几样。"""
    return SimpleNamespace(
        content=[SimpleNamespace(type="text", text=text)], stop_reason=stop_reason
    )


class FakeMessages:
    def __init__(self, responses) -> None:
        self.responses = list(responses)
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(dict(kwargs))
        return self.responses.pop(0)


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
