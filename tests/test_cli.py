"""cli 交互循环的测试：多轮之间把历史接下去、并落盘。全部本地执行——假 provider、假输入。"""

from __future__ import annotations

from mini_code_agent import cli
from mini_code_agent.permissions import Grants
from mini_code_agent.prompts import NullPrompter
from mini_code_agent.providers.base import ModelResponse
from mini_code_agent.sessions import Session, load, start
from mini_code_agent.tools import ToolRegistry

SYSTEM = "你是一个 coding agent。"


class ScriptedProvider:
    """按剧本回答，记下每次收到了什么。"""

    def __init__(self, responses: list[ModelResponse]) -> None:
        self.responses = list(responses)
        self.calls: list[list[dict]] = []
        self.systems: list[str | None] = []

    def complete(self, messages, tools=None, system=None) -> ModelResponse:
        self.calls.append([dict(m) for m in messages])
        self.systems.append(system)
        return self.responses.pop(0)


def text_turn(text: str) -> ModelResponse:
    return ModelResponse(
        text=text, assistant_content=[{"type": "text", "text": text}], stop_reason="end_turn"
    )


def run_loop(
    monkeypatch,
    provider: ScriptedProvider,
    lines: list[str],
    tmp_path,
    *,
    session: Session | None = None,
    messages: list[dict] | None = None,
) -> Session:
    """喂一串「用户输入」，把交互循环跑起来。最后一句给 /exit 收尾。"""
    answers = iter(lines)
    monkeypatch.setattr(cli.typer, "prompt", lambda *args, **kwargs: next(answers))
    session = session or start(tmp_path, permission_mode="default", allow_shell=False)
    cli._chat(
        provider,
        ToolRegistry([]),
        SYSTEM,
        session,
        [] if messages is None else messages,
        cwd=tmp_path,
        allow_shell=False,
        permission_mode="default",
        prompter=NullPrompter(),
        grants=Grants(),
    )
    return session


def test_the_loop_carries_the_conversation(monkeypatch, tmp_path):
    provider = ScriptedProvider([text_turn("嗯"), text_turn("接着说")])

    run_loop(monkeypatch, provider, ["第一句", "第二句", "/exit"], tmp_path)

    assert len(provider.calls) == 2
    # 第二轮带着第一轮：user / assistant / user
    assert [m["role"] for m in provider.calls[1]] == ["user", "assistant", "user"]
    assert provider.calls[1][0]["content"] == "第一句"


def test_the_system_prompt_goes_out_on_every_turn(monkeypatch, tmp_path):
    provider = ScriptedProvider([text_turn("嗯"), text_turn("嗯")])

    run_loop(monkeypatch, provider, ["一", "二", "/exit"], tmp_path)

    assert provider.systems == [SYSTEM, SYSTEM]


def test_a_blank_line_is_not_a_turn(monkeypatch, tmp_path):
    provider = ScriptedProvider([text_turn("嗯")])

    run_loop(monkeypatch, provider, ["", "   ", "正经问一句", "/exit"], tmp_path)

    assert len(provider.calls) == 1
    assert provider.calls[0][0]["content"] == "正经问一句"


def test_exit_stops_before_calling_the_model(monkeypatch, tmp_path):
    provider = ScriptedProvider([])

    run_loop(monkeypatch, provider, ["/exit"], tmp_path)

    assert provider.calls == []


def test_the_transcript_lands_on_disk(monkeypatch, tmp_path):
    """循环跑完，这几轮说的话得能在盘上读回来。"""
    provider = ScriptedProvider([text_turn("嗯")])

    run_loop(monkeypatch, provider, ["第一句", "/exit"], tmp_path)

    _, meta, messages = load(tmp_path)

    assert meta["permission_mode"] == "default"
    assert [m["role"] for m in messages] == ["user", "assistant"]
    assert messages[0]["content"] == "第一句"


def test_a_long_history_is_compacted_before_the_turn(monkeypatch, tmp_path):
    """历史太长就先压一次再发：压出来那段顶在最前面，而且留下的那些恢复时还在。"""
    long_history = [
        message
        for i in range(10)
        for message in (
            {"role": "user", "content": f"第 {i} 句" + "x" * 15_000},
            {"role": "assistant", "content": [{"type": "text", "text": "好"}]},
        )
    ]
    # 和真实流程一样：先写进会话，再当成"接着上一次"的那份历史用
    session = start(tmp_path, permission_mode="default", allow_shell=False)
    session.append(long_history)
    provider = ScriptedProvider([text_turn("前面聊过十件事"), text_turn("接着说")])

    run_loop(
        monkeypatch,
        provider,
        ["再来一句", "/exit"],
        tmp_path,
        session=session,
        messages=long_history,
    )

    # 第一次请求是摘要那次：历史后面跟着那条摘要指令
    assert "摘要" in provider.calls[0][-1]["content"]
    # 第二次才是真正这一轮，历史已经压过：第一条就是那段摘要
    assert "前面聊过十件事" in provider.calls[1][0]["content"]
    # 恢复出来的那份和压过的一致：摘要在最前，压掉的没有，留着的都在
    _, _, stored = load(tmp_path, session.id)
    texts = [str(message.get("content")) for message in stored]
    assert "前面聊过十件事" in texts[0]
    assert not any("第 0 句" in text for text in texts)  # 被压掉的那几轮
    assert any("第 4 句" in text for text in texts)  # 留着的


def test_a_short_history_is_not_compacted(monkeypatch, tmp_path):
    provider = ScriptedProvider([text_turn("嗯")])

    run_loop(monkeypatch, provider, ["一句话", "/exit"], tmp_path)

    # 只有一次请求（就这一轮），没有摘要那次
    assert len(provider.calls) == 1
