"""cli 交互循环的测试：多轮之间把历史接下去、并落盘。全部本地执行——假 provider、假输入。"""

from __future__ import annotations

import os

import pytest
import typer

from mini_code_agent import cli
from mini_code_agent.memory import Memory, write
from mini_code_agent.permissions import Grants, PermissionRequest
from mini_code_agent.prompts import NullPrompter
from mini_code_agent.providers.base import ModelResponse, ProviderError, ToolCall
from mini_code_agent.sessions import SESSIONS_DIR, Session, load, start
from mini_code_agent.summarizing import MAX_HISTORY_CHARS
from mini_code_agent.system_prompt import build_system_prompt
from mini_code_agent.tools import Tool, ToolRegistry, spill
from mini_code_agent.tools.safety import MAX_TOOL_CHARS


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


class FlakyProvider:
    """第一次报错，之后正常——模拟网络抖了一下。"""

    def __init__(self, responses: list[ModelResponse]) -> None:
        self.responses = list(responses)
        self.calls = 0

    def complete(self, messages, tools=None, system=None) -> ModelResponse:
        self.calls += 1
        if self.calls == 1:
            raise ProviderError("连不上模型服务。")
        return self.responses.pop(0)


def test_a_failed_turn_does_not_kill_the_session(monkeypatch, tmp_path, capsys):
    """请求失败只该收掉**这一轮**：说一句，然后下一句照常发得出去。

    以前这儿没有接住的地方——SDK 重试完还是失败的话，异常会一路冒到顶，**整个交互会话跟着
    死掉**，用户看见的是 Python 的 traceback。
    """
    provider = FlakyProvider([text_turn("第二句答上了")])

    run_loop(monkeypatch, provider, ["第一句", "第二句", "/exit"], tmp_path)

    out = capsys.readouterr()
    assert "这一轮没跑成" in out.err
    assert "连不上模型服务" in out.err
    assert provider.calls == 2  # 第二句照常发出去了
    assert "第二句答上了" in out.out


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
    # 每条按阈值算，不写死数字：阈值从 12 万调到 20 万之后，原来写死的 15_000 就刚好不够了
    # （10 × 15010 ≈ 151k < 200k），这条测试会静默地退化成"什么都没压"，而断言还是那几条。
    filler = MAX_HISTORY_CHARS // 8
    long_history = [
        message
        for i in range(10)
        for message in (
            {"role": "user", "content": f"第 {i} 句" + "x" * filler},
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


def test_the_list_flag_prints_the_sessions_and_stops(tmp_path, capsys):
    """`mca --list` 只看不做：它排在所有构造之前，所以连 provider 都不用建——没有 API key
    也跑得起来，也不会因为列个目录就在盘上留下一个会话。"""
    session = start(tmp_path, permission_mode="default", allow_shell=False)
    session.append([{"role": "user", "content": "给我看看那个文件"}])

    with pytest.raises(typer.Exit):
        cli.main_command(
            prompt="",
            cwd=tmp_path,
            shell=False,
            permission_mode="default",
            continue_=False,
            resume="",
            list_=True,
            delete_="",
        )

    out = capsys.readouterr().out
    assert session.id in out
    assert "给我看看那个文件" in out
    assert "--resume" in out


def test_sessions_marks_the_one_you_are_in(monkeypatch, tmp_path, capsys):
    """`/sessions` 跟 `--list` 是同一份渲染，多的那个星号指着这次会话——所以它得认得
    循环里那个活着的 session，而不是别处的一份。"""
    provider = ScriptedProvider([text_turn("嗯")])

    session = run_loop(monkeypatch, provider, ["你好", "/sessions", "/exit"], tmp_path)

    assert f"* {session.id}" in capsys.readouterr().out


def run_delete(tmp_path, session_id: str, *, yes: bool) -> None:
    """跑一次 `mca --delete <id>`。两条路都以 typer.Exit 收场——删了，或者用户没同意。"""
    with pytest.raises(typer.Exit):
        cli.main_command(
            prompt="",
            cwd=tmp_path,
            shell=False,
            permission_mode="default",
            continue_=False,
            resume="",
            list_=False,
            delete_=session_id,
            yes=yes,
        )


def test_delete_shows_what_it_would_remove_and_asks_first(tmp_path, capsys, monkeypatch):
    """id 是一串时间戳，光看它认不出是哪一回——所以要把"最后说了什么"摆出来，再问一次。

    删错了拿不回来：没有回收站，也不像改文件那样有 `.agent/.history` 兜着（会话是纯追加的，
    从来不备份）。所以默认是要问的。
    """
    session = start(tmp_path, permission_mode="default", allow_shell=False)
    session.append([{"role": "user", "content": "这一段对话我不要了"}])
    monkeypatch.setattr(cli.typer, "confirm", lambda *args, **kwargs: False)

    run_delete(tmp_path, session.id, yes=False)

    out = capsys.readouterr().out
    assert session.id in out
    assert "1 条消息" in out
    assert "这一段对话我不要了" in out
    assert "没删" in out
    assert session.path.exists()


def test_answering_yes_removes_the_file(tmp_path, capsys, monkeypatch):
    """顺带证明它**不需要 provider**：删除排在构造之前，没有 API key 也跑得起来。"""
    session = start(tmp_path, permission_mode="default", allow_shell=False)
    session.append([{"role": "user", "content": "删了我"}])
    monkeypatch.setattr(cli.typer, "confirm", lambda *args, **kwargs: True)

    run_delete(tmp_path, session.id, yes=False)

    assert not session.path.exists()
    assert "删掉了" in capsys.readouterr().out


def test_the_yes_flag_skips_the_question(tmp_path, monkeypatch):
    session = start(tmp_path, permission_mode="default", allow_shell=False)
    session.append([{"role": "user", "content": "删了我"}])

    def never_ask(*args, **kwargs):
        raise AssertionError("给了 --yes 就不该再问")

    monkeypatch.setattr(cli.typer, "confirm", never_ask)

    run_delete(tmp_path, session.id, yes=True)

    assert not session.path.exists()


def test_deleting_an_unknown_id_is_a_bad_parameter(tmp_path):
    with pytest.raises(typer.BadParameter, match="没有"):
        cli.main_command(
            prompt="",
            cwd=tmp_path,
            shell=False,
            permission_mode="default",
            continue_=False,
            resume="",
            list_=False,
            delete_="20200101-000000",
            yes=True,
        )


def test_spilled_output_lands_in_this_sessions_directory(monkeypatch, tmp_path):
    """走一遍完整的：CLI 把**这次会话的目录**交给工具层，太长的输出就落在那里。

    落在会话目录里，`--delete` 才谈得上"连同落盘文件一起删"。不传的话会落到共用的
    `.agent/tool-results/` 里，那些文件谁也认不出来。
    """
    chatty = Tool(
        name="chatty",
        fn=lambda kwargs, ctx: "x" * (MAX_TOOL_CHARS + 1),
        spill=True,
    )
    provider = ScriptedProvider([tool_turn("t1", "chatty", {}), text_turn("看完了")])

    session = run_loop(
        monkeypatch, provider, ["跑一下", "/exit"], tmp_path, registry=ToolRegistry([chatty])
    )

    saved = list(spill.session_dir(tmp_path, session.id).glob("chatty-*.txt"))
    assert len(saved) == 1
    assert saved[0].read_text(encoding="utf-8") == "x" * (MAX_TOOL_CHARS + 1)


def test_delete_takes_the_spilled_files_with_it(tmp_path, capsys, monkeypatch):
    """落盘的工具结果属于那次会话——删会话要一起端掉，而且**动手之前要说清有几个**。"""
    session = start(tmp_path, permission_mode="default", allow_shell=False)
    session.append([{"role": "user", "content": "跑了一条很长的命令"}])
    directory = spill.session_dir(tmp_path, session.id)
    directory.mkdir(parents=True)
    (directory / "bash-abc123.txt").write_text("很长的输出", encoding="utf-8")

    monkeypatch.setattr(cli.typer, "confirm", lambda *args, **kwargs: False)
    run_delete(tmp_path, session.id, yes=False)
    capsys.readouterr()
    assert directory.exists()  # 答了"不删"，那批文件一个都不动

    monkeypatch.setattr(cli.typer, "confirm", lambda *args, **kwargs: True)
    run_delete(tmp_path, session.id, yes=False)

    assert "1 个落盘文件" in capsys.readouterr().out
    assert not directory.exists()
