"""cli 交互循环的测试：多轮之间把历史接下去、并落盘。全部本地执行——假 provider、假输入。"""

from __future__ import annotations

import os

from mini_code_agent import cli
from mini_code_agent.memory import Memory, write
from mini_code_agent.permissions import Grants, PermissionRequest
from mini_code_agent.prompts import NullPrompter
from mini_code_agent.providers.base import ModelResponse, ToolCall
from mini_code_agent.sessions import SESSIONS_DIR, Session, load, start
from mini_code_agent.system_prompt import build_system_prompt
from mini_code_agent.tools import Tool, ToolRegistry


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


def tool_turn(call_id: str, name: str, arguments: dict) -> ModelResponse:
    return ModelResponse(
        text=None,
        tool_calls=[ToolCall(id=call_id, name=name, arguments=arguments)],
        assistant_content=[{"type": "tool_use", "id": call_id, "name": name, "input": arguments}],
        stop_reason="tool_use",
    )


def run_loop(
    monkeypatch,
    provider: ScriptedProvider,
    lines: list[str],
    tmp_path,
    *,
    session: Session | None = None,
    messages: list[dict] | None = None,
    registry: ToolRegistry | None = None,
    grants: Grants | None = None,
) -> Session:
    """喂一串「用户输入」，把交互循环跑起来。最后一句给 /exit 收尾。"""
    answers = iter(lines)
    monkeypatch.setattr(cli.typer, "prompt", lambda *args, **kwargs: next(answers))
    session = session or start(tmp_path, permission_mode="default", allow_shell=False)
    cli._chat(
        provider,
        registry or ToolRegistry([]),
        session,
        [] if messages is None else messages,
        cwd=tmp_path,
        allow_shell=False,
        permission_mode="default",
        prompter=NullPrompter(),
        grants=grants if grants is not None else Grants(),
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

    expected = build_system_prompt(tmp_path, "default")
    assert provider.systems == [expected, expected]


def test_a_memory_written_mid_run_shows_up_next_turn(monkeypatch, tmp_path):
    """模型这一轮记了一条，下一轮的提示词里就该有它——所以每轮重拼一次。

    （一轮内部不重拼：`runagent` 那一串步骤用的是同一份，所以这一轮记的东西要**下一轮**
    才看得见——工具返回的话也是这么说的。）
    """

    def scribble(kwargs: dict, ctx) -> str:
        write(ctx.cwd, Memory(name="fresh-note", description="刚记的", type="user", body="正文"))
        return "记下了"

    registry = ToolRegistry([Tool(name="scribble", description="", fn=scribble, params={})])
    provider = ScriptedProvider(
        [tool_turn("t1", "scribble", {}), text_turn("接着说"), text_turn("嗯")]
    )

    run_loop(monkeypatch, provider, ["记一条", "再说一句", "/exit"], tmp_path, registry=registry)

    assert "fresh-note" not in provider.systems[0]  # 第一轮（包括它内部那几步）还没有
    assert "fresh-note" in provider.systems[-1]  # 第二轮就有了


def test_a_blank_line_is_not_a_turn(monkeypatch, tmp_path):
    provider = ScriptedProvider([text_turn("嗯")])

    run_loop(monkeypatch, provider, ["", "   ", "正经问一句", "/exit"], tmp_path)

    assert len(provider.calls) == 1
    assert provider.calls[0][0]["content"] == "正经问一句"


def test_exit_stops_before_calling_the_model(monkeypatch, tmp_path):
    provider = ScriptedProvider([])

    run_loop(monkeypatch, provider, ["/exit"], tmp_path)

    assert provider.calls == []


def test_a_command_line_never_reaches_the_model(monkeypatch, tmp_path):
    """`/` 开头的那几行由程序处理：不发给模型，也不进 transcript。"""
    provider = ScriptedProvider([text_turn("嗯")])

    session = run_loop(monkeypatch, provider, ["/hlep", "/help", "正经问一句", "/exit"], tmp_path)

    assert len(provider.calls) == 1
    assert provider.calls[0][0]["content"] == "正经问一句"
    # 命令是用户对程序说的话，不是对话的一部分——接上历史时不该看见它
    _, _, stored = load(tmp_path, session.id)
    assert [m["role"] for m in stored] == ["user", "assistant"]
    assert stored[0]["content"] == "正经问一句"


def test_switching_the_mode_lands_on_the_next_turn(monkeypatch, tmp_path):
    """`/permission_mode` 改的是循环每轮现读的那份，所以换完这一句就该按新档走。"""
    provider = ScriptedProvider([text_turn("嗯")])

    run_loop(monkeypatch, provider, ["/permission_mode plan", "看看再说", "/exit"], tmp_path)

    assert provider.systems == [build_system_prompt(tmp_path, "plan")]


def test_status_reports_the_live_loop(monkeypatch, tmp_path, capsys):
    """`/status` 拿到的是循环里那份会话和历史，不是一份副本。"""
    provider = ScriptedProvider([text_turn("嗯")])

    session = run_loop(monkeypatch, provider, ["你好", "/status", "/exit"], tmp_path)

    out = capsys.readouterr().out
    assert f"工作目录：{tmp_path}" in out
    assert session.id in out  # 报告的正是这次落盘的那个会话文件
    assert "落盘 2 条消息" in out  # 上一轮那句问和那句答


def test_a_session_that_never_says_anything_leaves_nothing_behind(monkeypatch, tmp_path):
    """从头到尾只用命令、一句话没说——盘上不该留下会话文件（以前会留下一个空壳）。"""
    provider = ScriptedProvider([])

    run_loop(monkeypatch, provider, ["/status", "/help", "/exit"], tmp_path)

    assert provider.calls == []  # 命令是程序答的，没往模型那儿跑一趟
    assert list((tmp_path / SESSIONS_DIR).glob("*.jsonl")) == []


def test_grants_reports_what_the_loop_has_approved(monkeypatch, tmp_path, capsys):
    """`/grants` 看的是传进循环的那个 Grants——这一会话真的批过什么，它就知道什么。"""
    grants = Grants()
    grants.remember(PermissionRequest(tool_name="write_file", tool_args={"path": "src/a.py"}))

    run_loop(monkeypatch, ScriptedProvider([]), ["/grants", "/exit"], tmp_path, grants=grants)

    assert os.path.normpath("src/a.py") in capsys.readouterr().out


def test_compact_squeezes_history_that_is_nowhere_near_the_limit(monkeypatch, tmp_path):
    """手动那一条**不看阈值**：短得自动那条路永远碰不到的历史，照样压。"""
    short_history = [
        message
        for i in range(8)
        for message in (
            {"role": "user", "content": f"第 {i} 句"},
            {"role": "assistant", "content": [{"type": "text", "text": "好"}]},
        )
    ]
    session = start(tmp_path, permission_mode="default", allow_shell=False)
    session.append(short_history)
    provider = ScriptedProvider([text_turn("前面聊过八件事"), text_turn("接着说")])

    run_loop(
        monkeypatch,
        provider,
        ["/compact", "再来一句", "/exit"],
        tmp_path,
        session=session,
        messages=short_history,
    )

    # 第一次请求是摘要那次（历史后面跟着那条摘要指令），第二次才是真正这一轮
    assert "摘要" in provider.calls[0][-1]["content"]
    # 这一轮发出去的历史已经压过：第一条就是那段摘要
    assert "前面聊过八件事" in provider.calls[1][0]["content"]
    # 而且落了盘，不是只活在这一次进程里——接上会话还能看见
    _, _, stored = load(tmp_path, session.id)
    assert "前面聊过八件事" in str(stored[0].get("content"))


def test_compact_says_so_when_there_is_no_safe_place_to_cut(monkeypatch, tmp_path, capsys):
    """轮数不够就没有合法的切点——这时候不该白调一次模型。"""
    provider = ScriptedProvider([])

    run_loop(monkeypatch, provider, ["/compact", "/exit"], tmp_path)

    assert provider.calls == []
    assert "没什么可压的" in capsys.readouterr().out


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
