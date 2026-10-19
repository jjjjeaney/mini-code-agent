"""流式路径的测试：provider 的 delta → final 归一化，以及 agent / CLI 的接线。

全部本地执行，不联网。真实端点的流式冒烟在 test_anthropic_provider.py 里。
"""

from __future__ import annotations

import threading
from types import SimpleNamespace

import httpx2
import pytest
from anthropic import APIConnectionError

from mini_code_agent import cli
from mini_code_agent.core.agent import runagent
from mini_code_agent.permissions import Grants
from mini_code_agent.prompts import NullPrompter
from mini_code_agent.providers.anthropic import AnthropicProvider
from mini_code_agent.providers.base import (
    ModelResponse,
    ProviderError,
    TransientProviderError,
)
from mini_code_agent.sessions import start
from mini_code_agent.tools import ToolRegistry


def _usage() -> SimpleNamespace:
    return SimpleNamespace(
        input_tokens=10,
        output_tokens=5,
        cache_read_input_tokens=2,
        cache_creation_input_tokens=0,
    )


def _final(content, *, stop_reason="end_turn") -> SimpleNamespace:
    return SimpleNamespace(content=content, stop_reason=stop_reason, usage=_usage())


def _text(value: str) -> SimpleNamespace:
    return SimpleNamespace(type="text", text=value)


def _thinking(value: str) -> SimpleNamespace:
    return SimpleNamespace(type="thinking", thinking=value, signature="sig")


def _tool(call_id="t1", name="read_file", arguments=None) -> SimpleNamespace:
    return SimpleNamespace(type="tool_use", id=call_id, name=name, input=arguments or {})


class FakeStream:
    """够 provider 用的最小流：逐条给出 content_block_delta，最后给完整消息。

    `chunks` 里的元素可以是字符串（正文）或 `("thinking", "…")` 这样的二元组。
    """

    def __init__(self, chunks, final, *, fail_after=None, error=None):
        self._chunks = list(chunks)
        self._final = final
        self._fail_after = fail_after
        self._error = error
        self._iterator = self._generate()

    def __iter__(self):
        return self._iterator

    def _generate(self):
        for index, chunk in enumerate(self._chunks):
            if self._error is not None and index == self._fail_after:
                raise self._error
            kind, text = chunk if isinstance(chunk, tuple) else ("text", chunk)
            yield SimpleNamespace(
                type="content_block_delta",
                delta=SimpleNamespace(type=f"{kind}_delta", **{kind: text}),
            )
        if (
            self._error is not None
            and self._fail_after is not None
            and self._fail_after >= len(self._chunks)
        ):
            raise self._error

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def get_final_message(self):
        return self._final


class FakeMessages:
    def __init__(self, stream=None, *, error=None):
        self._stream = stream
        self._error = error
        self.kwargs = None

    def stream(self, **kwargs):
        self.kwargs = kwargs
        if self._error is not None:
            raise self._error
        return self._stream


def provider_with(messages) -> AnthropicProvider:
    provider = AnthropicProvider(api_key="test", base_url="http://localhost")
    provider.client = SimpleNamespace(messages=messages)
    return provider


def connection_error() -> APIConnectionError:
    return APIConnectionError(request=httpx2.Request("POST", "http://localhost"))


# --- provider -------------------------------------------------------------


def test_text_deltas_are_shown_and_the_final_message_is_normalized():
    messages = FakeMessages(FakeStream(["你", "好"], _final([_text("你好")])))
    provider = provider_with(messages)
    seen: list[str] = []

    response = provider.stream([{"role": "user", "content": "hi"}], on_text=seen.append)

    assert seen == ["你", "好"]
    assert response.text == "你好"
    assert response.stop_reason == "end_turn"
    assert response.input_tokens == 10
    assert response.cache_read_input_tokens == 2
    # 流式不能抬了重发，所以一开始就向天花板要。
    assert messages.kwargs["max_tokens"] == provider.ceiling


def test_thinking_and_tool_blocks_are_kept_but_not_printed():
    final = _final([_thinking("想一下"), _text("好的"), _tool()], stop_reason="tool_use")
    provider = provider_with(FakeMessages(FakeStream(["好的"], final)))
    seen: list[str] = []

    response = provider.stream([], on_text=seen.append)

    assert seen == ["好的"]
    assert [block["type"] for block in response.assistant_content] == [
        "thinking",
        "text",
        "tool_use",
    ]
    assert response.tool_calls[0].name == "read_file"


def test_thinking_deltas_go_to_their_own_callback():
    final = _final([_thinking("想一下"), _text("好的")])
    provider = provider_with(
        FakeMessages(FakeStream([("thinking", "想一下"), "好的"], final))
    )
    thinking: list[str] = []
    text: list[str] = []

    response = provider.stream([], on_text=text.append, on_thinking=thinking.append)

    assert thinking == ["想一下"]
    assert text == ["好的"]
    assert response.text == "好的"
    assert response.assistant_content[0]["thinking"] == "想一下"


def test_a_tool_only_response_prints_nothing():
    final = _final([_tool(arguments={"path": "a.txt"})], stop_reason="tool_use")
    provider = provider_with(FakeMessages(FakeStream([], final)))
    seen: list[str] = []

    response = provider.stream([], on_text=seen.append)

    assert seen == []
    assert response.text is None
    assert response.tool_calls[0].arguments == {"path": "a.txt"}


def test_a_truncated_stream_without_text_is_transient():
    provider = provider_with(FakeMessages(FakeStream([], _final([], stop_reason=None))))

    with pytest.raises(TransientProviderError):
        provider.stream([], on_text=lambda chunk: None)


def test_a_truncated_stream_after_text_is_not_retried():
    final = _final([_text("半句话")], stop_reason=None)
    provider = provider_with(FakeMessages(FakeStream(["半句话"], final)))

    with pytest.raises(ProviderError) as info:
        provider.stream([], on_text=lambda chunk: None)

    assert not isinstance(info.value, TransientProviderError)
    assert "再发一次" in str(info.value)


def test_a_failure_before_any_text_can_be_retried():
    stream = FakeStream([], _final([_text("x")]), fail_after=0, error=connection_error())
    provider = provider_with(FakeMessages(stream))

    with pytest.raises(TransientProviderError):
        provider.stream([], on_text=lambda chunk: None)


def test_a_failure_after_text_is_not_retried():
    stream = FakeStream(["半"], _final([_text("半")]), fail_after=1, error=connection_error())
    provider = provider_with(FakeMessages(stream))

    with pytest.raises(ProviderError) as info:
        provider.stream([], on_text=lambda chunk: None)

    assert not isinstance(info.value, TransientProviderError)


def test_without_a_callback_nothing_was_shown_so_retry_is_safe():
    stream = FakeStream(["半"], _final([_text("半")]), fail_after=1, error=connection_error())
    provider = provider_with(FakeMessages(stream))

    with pytest.raises(TransientProviderError):
        provider.stream([], on_text=None)


def test_the_initial_connection_failure_is_translated():
    provider = provider_with(FakeMessages(error=connection_error()))

    with pytest.raises(TransientProviderError):
        provider.stream([], on_text=lambda chunk: None)


def test_a_stream_that_dies_before_any_event_translates_the_sdk_assertion():
    class NoEvents(FakeStream):
        def get_final_message(self):
            raise AssertionError

    provider = provider_with(FakeMessages(NoEvents([], _final([]))))

    with pytest.raises(TransientProviderError):
        provider.stream([], on_text=lambda chunk: None)


def test_a_transport_error_during_iteration_is_translated():
    stream = FakeStream(
        [], _final([_text("x")]), fail_after=0, error=httpx2.ReadError("boom")
    )
    provider = provider_with(FakeMessages(stream))

    with pytest.raises(TransientProviderError):
        provider.stream([], on_text=lambda chunk: None)


def test_a_transport_error_after_text_is_not_retried():
    stream = FakeStream(
        ["半"], _final([_text("半")]), fail_after=1, error=httpx2.ReadError("boom")
    )
    provider = provider_with(FakeMessages(stream))

    with pytest.raises(ProviderError) as info:
        provider.stream([], on_text=lambda chunk: None)

    assert not isinstance(info.value, TransientProviderError)


# --- agent ----------------------------------------------------------------


class StreamingProvider:
    model = "fake"

    def __init__(self, chunks=("你", "好"), thinking=(), input_tokens=0):
        self.chunks = chunks
        self.thinking = thinking
        self.input_tokens = input_tokens
        self.stream_calls: list[dict] = []
        self.complete_calls: list[dict] = []

    def stream(self, messages, tools=None, system=None, on_text=None, on_thinking=None):
        self.stream_calls.append({"tools": tools, "system": system})
        if on_thinking is not None:
            for chunk in self.thinking:
                on_thinking(chunk)
        if on_text is not None:
            for chunk in self.chunks:
                on_text(chunk)
        text = "".join(self.chunks)
        return ModelResponse(
            text=text,
            assistant_content=[{"type": "text", "text": text}],
            stop_reason="end_turn",
            input_tokens=self.input_tokens,
        )

    def complete(self, messages, tools=None, system=None):
        self.complete_calls.append({"tools": tools, "system": system})
        return ModelResponse(
            text="完整回复",
            assistant_content=[{"type": "text", "text": "完整回复"}],
            stop_reason="end_turn",
        )


class PlainProvider:
    def __init__(self):
        self.calls = 0

    def complete(self, messages, tools=None, system=None):
        self.calls += 1
        return ModelResponse(
            text="完整回复",
            assistant_content=[{"type": "text", "text": "完整回复"}],
            stop_reason="end_turn",
        )


def test_the_agent_streams_the_foreground_turn():
    provider = StreamingProvider()
    seen: list[str] = []

    result = runagent("hi", provider, ToolRegistry([]), stream=True, on_text=seen.append)

    assert provider.stream_calls
    assert provider.complete_calls == []
    # 首尾各一个换行：前面隔开提示符，后面隔开用量行。
    assert seen == ["\n", "你", "好", "\n"]
    assert result.final == "你好"
    assert result.final_streamed is True


def test_thinking_is_shown_before_the_answer():
    provider = StreamingProvider(chunks=("好",), thinking=("想",))
    text: list[str] = []
    thinking: list[str] = []

    result = runagent(
        "hi",
        provider,
        ToolRegistry([]),
        stream=True,
        on_text=text.append,
        on_thinking=thinking.append,
    )

    assert thinking == ["想"]
    # 开头一个换行、thinking→正文切换一个换行、正文、结尾一个换行。
    assert text == ["\n", "\n", "好", "\n"]
    assert result.final == "好"
    assert result.final_streamed is True


class RecordingHint:
    def __init__(self):
        self.calls: list[str] = []

    def begin(self):
        self.calls.append("begin")

    def end(self):
        self.calls.append("end")


def test_the_wait_hint_brackets_a_streamed_request():
    hint = RecordingHint()

    runagent(
        "hi",
        StreamingProvider(),
        ToolRegistry([]),
        stream=True,
        on_text=lambda chunk: None,
        wait_hint=hint,
    )

    assert hint.calls == ["begin", "end"]


def test_the_wait_hint_is_cleared_when_the_request_fails():
    class FailingProvider:
        model = "fake"

        def stream(self, messages, tools=None, system=None, on_text=None, on_thinking=None):
            raise ProviderError("断了")

        def complete(self, messages, tools=None, system=None):
            raise AssertionError("不该走到这里")

    hint = RecordingHint()
    result = runagent(
        "hi",
        FailingProvider(),
        ToolRegistry([]),
        stream=True,
        on_text=lambda chunk: None,
        wait_hint=hint,
    )

    assert result.error == "断了"
    assert hint.calls == ["begin", "end"]


def test_a_provider_without_stream_does_not_show_the_wait_hint():
    hint = RecordingHint()

    runagent(
        "hi",
        PlainProvider(),
        ToolRegistry([]),
        stream=True,
        on_text=lambda chunk: None,
        wait_hint=hint,
    )

    assert hint.calls == []


def test_a_provider_without_stream_falls_back_to_complete():
    provider = PlainProvider()
    seen: list[str] = []

    result = runagent("hi", provider, ToolRegistry([]), stream=True, on_text=seen.append)

    assert provider.calls == 1
    assert seen == []
    assert result.final == "完整回复"
    assert result.final_streamed is False


def test_stream_without_a_callback_uses_complete():
    provider = StreamingProvider()

    result = runagent("hi", provider, ToolRegistry([]), stream=True)

    assert provider.complete_calls
    assert provider.stream_calls == []
    assert result.final_streamed is False


def test_a_background_handoff_does_not_stream():
    provider = StreamingProvider()
    seen: list[str] = []
    held: dict = {}

    def handoff(work):
        held["work"] = work
        return "bg-1"

    result = runagent(
        "hi",
        provider,
        ToolRegistry([]),
        stream=True,
        on_text=seen.append,
        background_after=0,
        handoff=handoff,
        allow_background=True,
    )

    assert result.handoff == "bg-1"
    assert seen == []
    assert provider.stream_calls == []

    held["work"](lambda text, style: None, threading.Event())

    assert provider.stream_calls == []
    assert provider.complete_calls


# --- CLI ------------------------------------------------------------------


def test_the_cli_does_not_print_a_streamed_answer_twice(monkeypatch, tmp_path, capsys):
    provider = StreamingProvider()
    answers = iter(["你好", "/exit"])
    monkeypatch.setattr(cli.typer, "prompt", lambda *args, **kwargs: next(answers))
    session = start(tmp_path, permission_mode="default", allow_shell=False)

    cli._chat(
        provider,
        ToolRegistry([]),
        session,
        [],
        cwd=tmp_path,
        allow_shell=False,
        permission_mode="default",
        prompter=NullPrompter(),
        grants=Grants(),
        stream=True,
    )

    out = capsys.readouterr().out
    assert out.count("你好") == 1


def test_a_streamed_answer_also_prints_usage_after_it(monkeypatch, tmp_path, capsys):
    provider = StreamingProvider(chunks=("答", "案"), input_tokens=10)
    answers = iter(["你好", "/exit"])
    monkeypatch.setattr(cli.typer, "prompt", lambda *args, **kwargs: next(answers))
    session = start(tmp_path, permission_mode="default", allow_shell=False)

    cli._chat(
        provider,
        ToolRegistry([]),
        session,
        [],
        cwd=tmp_path,
        allow_shell=False,
        permission_mode="default",
        prompter=NullPrompter(),
        grants=Grants(),
        stream=True,
    )

    out = capsys.readouterr().out
    assert out.index("答案") < out.index("这一发 token")


def test_the_cli_can_show_thinking(monkeypatch, tmp_path, capsys):
    provider = StreamingProvider(chunks=("好",), thinking=("想",))
    answers = iter(["你好", "/exit"])
    monkeypatch.setattr(cli.typer, "prompt", lambda *args, **kwargs: next(answers))
    session = start(tmp_path, permission_mode="default", allow_shell=False)

    cli._chat(
        provider,
        ToolRegistry([]),
        session,
        [],
        cwd=tmp_path,
        allow_shell=False,
        permission_mode="default",
        prompter=NullPrompter(),
        grants=Grants(),
        stream=True,
        show_thinking=True,
    )

    out = capsys.readouterr().out
    assert out.count("想") == 1
    assert out.count("好") == 1
    assert out.index("想") < out.index("好")


def test_the_wait_hint_is_printed_and_erased_on_a_terminal(monkeypatch, capsys):
    monkeypatch.setattr(type(cli.console), "is_terminal", property(lambda self: True))
    writer = cli._StreamWriter()

    writer.begin()
    writer.end()

    out = capsys.readouterr().out
    assert "等待模型输出" in out
    assert "\r" in out


def test_the_wait_hint_is_silent_when_output_is_not_a_terminal(capsys):
    writer = cli._StreamWriter()

    writer.begin()
    writer.end()

    assert capsys.readouterr().out == ""
