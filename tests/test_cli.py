"""cli 交互循环的测试：多轮之间把历史接下去、并落盘。全部本地执行——假 provider、假输入。"""

from __future__ import annotations

import os
import queue
import threading
import time

import pytest
import typer

from mini_code_agent import cli
from mini_code_agent.core.agent import AgentResult
from mini_code_agent.core.tasks import TaskManager
from mini_code_agent.folding import FOLD_AFTER_SECONDS, MIN_FOLD_CHARS
from mini_code_agent.interaction import PermissionBroker
from mini_code_agent.memory import Memory, write
from mini_code_agent.permissions import Grants, PermissionRequest
from mini_code_agent.prompting import TerminalPrompter
from mini_code_agent.prompts import Answer, NullPrompter, Prompter
from mini_code_agent.providers.base import ModelResponse, ProviderError, ToolCall
from mini_code_agent.sessions import SESSIONS_DIR, Session, load, start
from mini_code_agent.system_prompt import build_system_prompt
from mini_code_agent.tools import Tool, ToolRegistry, default_tools, spill
from mini_code_agent.tools.safety import MAX_TOOL_CHARS


class ScriptedProvider:
    """按剧本回答，记下每次收到了什么。"""

    def __init__(self, responses: list[ModelResponse]) -> None:
        self.responses = list(responses)
        self.calls: list[list[dict]] = []
        self.systems: list[str | None] = []
        # 每一发收到的那张工具表。和 calls / systems 一样按下标对齐。
        self.tools: list[list[dict] | None] = []

    def complete(self, messages, tools=None, system=None) -> ModelResponse:
        self.calls.append([dict(m) for m in messages])
        self.systems.append(system)
        self.tools.append(tools)
        return self.responses.pop(0)


def text_turn(text: str, *, input_tokens: int = 0) -> ModelResponse:
    return ModelResponse(
        text=text,
        assistant_content=[{"type": "text", "text": text}],
        stop_reason="end_turn",
        input_tokens=input_tokens,
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
    allow_fork: bool = False,
    tasks: TaskManager | None = None,
    broker: PermissionBroker | None = None,
    prompter: Prompter | None = None,
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
        prompter=prompter if prompter is not None else NullPrompter(),
        grants=grants if grants is not None else Grants(),
        allow_fork=allow_fork,
        tasks=tasks,
        broker=broker,
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


def an_eight_turn_history() -> list[dict]:
    """盘上那种"聊了八轮"的历史——够 8 轮，压缩才有安全的切点（最近 6 轮要留着）。"""
    return [
        message
        for i in range(8)
        for message in (
            {"role": "user", "content": f"第 {i} 句"},
            {"role": "assistant", "content": [{"type": "text", "text": "好"}]},
        )
    ]


def a_registry_with_one_tool() -> ToolRegistry:
    """**得有一个工具**：没有的话 `tools` 是空列表，那一发请求就不带工具 schema，
    `runagent` 也不会拿它当锚点（锚点的判据就是"这一发带了工具"，见 ContextAnchor）。"""
    return ToolRegistry([Tool(name="noop", fn=lambda kwargs, ctx: "ok")])


def test_a_big_context_gets_compacted_before_the_next_turn(monkeypatch, tmp_path):
    """上一发请求报回来的用量占了窗口八成以上，下一轮就先压一次再发。

    **触发靠的是接口报的真数，不是数出来的字符。** 这段历史本身很小（八轮短句子，几千字符），
    按老那套字符阈值永远不会压——压它是因为假响应说自己那一发有一百万 token。
    """
    provider = ScriptedProvider(
        [
            text_turn("第一轮", input_tokens=1_000_000),  # 把锚点顶到窗口以上
            text_turn("前面聊过八件事"),  # 摘要那一次
            text_turn("接着说"),  # 压完之后的这一轮
        ]
    )

    run_loop(
        monkeypatch,
        provider,
        ["第一句", "第二句", "/exit"],
        tmp_path,
        messages=an_eight_turn_history(),
        registry=a_registry_with_one_tool(),
    )

    # 第一次请求就是第一轮那句本身，前面没有摘要请求
    assert provider.calls[0][-1]["content"] == "第一句"
    # 第二次是摘要那次：历史后面跟着那条摘要指令
    assert "摘要" in provider.calls[1][-1]["content"]
    # 第三次才是第二轮，历史已经压过：第一条就是那段摘要
    assert "前面聊过八件事" in provider.calls[2][0]["content"]


def test_compaction_does_not_happen_while_the_context_is_small(monkeypatch, tmp_path):
    """报回来的用量没到阈值就不压——哪怕历史里有很多轮。"""
    provider = ScriptedProvider(
        [text_turn("第一轮", input_tokens=1_000), text_turn("第二轮")]
    )

    run_loop(
        monkeypatch,
        provider,
        ["第一句", "第二句", "/exit"],
        tmp_path,
        messages=an_eight_turn_history(),
        registry=a_registry_with_one_tool(),
    )

    # 两次请求就是两轮，中间没有摘要那次
    assert len(provider.calls) == 2
    assert all("摘要" not in call[-1]["content"] for call in provider.calls)


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
            fork=False,
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
            fork=False,
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
            fork=False,
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

    saved = list(spill.session_dir(tmp_path, session.id).glob("*.txt"))
    assert len(saved) == 1
    assert saved[0].stem == "t1"  # 文件名是那次工具调用的 id
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


class FakeOneShotProvider:
    """一次性那条路上顶掉真 provider：不发请求，只记下模型见到了什么。"""

    def __init__(self) -> None:
        self.model = "fake"
        self.tools: list[dict] | None = None

    def complete(self, messages, tools=None, system=None) -> ModelResponse:
        self.tools = tools
        return ModelResponse(
            text="好",
            assistant_content=[{"type": "text", "text": "好"}],
            stop_reason="end_turn",
        )


def test_the_agent_tool_is_offered_to_the_model(monkeypatch, tmp_path, capsys):
    """`agent` 是在 CLI 层 `registry.register()` 装进去的（和 `ask_user` 一样），
    所以它**不在** `default_tools()` 里——而它必须真的发到模型手上。

    这条走的是 `main_command` 那条真路：注册点排在 provider 后面（那个工具要 provider 才能
    跑），所以光看单测验不到"它到底装没装上"。
    """
    provider = FakeOneShotProvider()
    monkeypatch.setattr(cli, "AnthropicProvider", lambda: provider)

    cli.main_command(
        prompt="你好",
        cwd=tmp_path,
        shell=False,
        fork=False,
        permission_mode="default",
        continue_=False,
        resume="",
        list_=False,
        delete_="",
    )

    offered = {schema["name"] for schema in provider.tools or []}
    assert "agent" in offered
    assert "ask_user" in offered  # 另一个 harness 注入的，顺带确认这条路没被改坏
    assert "fork" not in offered  # fork 是 agent 的一个 type，不是单独的工具
    # 而且这个 type 一直在枚举里——档位/开关开关的是运行时能不能用，不是发不发。
    agent = next(schema for schema in provider.tools or [] if schema["name"] == "agent")
    assert "fork" in agent["input_schema"]["properties"]["type"]["enum"]
    capsys.readouterr()


def test_fork_is_a_type_of_the_agent_tool_in_both_cases(monkeypatch, tmp_path, capsys):
    """fork 不再是独立工具：枚举里一直有它，开没开只影响系统提示里那段说明。"""
    provider = FakeOneShotProvider()
    monkeypatch.setattr(cli, "AnthropicProvider", lambda: provider)

    def types_offered(*, fork: bool) -> set[str]:
        cli.main_command(
            prompt="你好",
            cwd=tmp_path,
            shell=False,
            fork=fork,
            permission_mode="default",
            continue_=False,
            resume="",
            list_=False,
            delete_="",
        )
        agent = next(schema for schema in provider.tools or [] if schema["name"] == "agent")
        return set(agent["input_schema"]["properties"]["type"]["enum"])

    assert "fork" in types_offered(fork=False)
    assert "fork" in types_offered(fork=True)
    capsys.readouterr()


def test_the_fork_switch_can_be_toggled_from_the_loop(monkeypatch, tmp_path, capsys):
    """`/fork on` 之后，下一轮的提示词里多出那段说明。

    工具表不变（fork 是 `agent` 的 type，枚举是静态的）——但提示词变了，前缀照样重建。
    """
    provider = ScriptedProvider([text_turn("好")])

    run_loop(monkeypatch, provider, ["/fork on", "你好", "/exit"], tmp_path)

    assert "什么时候用 fork" in provider.systems[0]
    capsys.readouterr()


def test_fork_is_off_by_default_in_the_loop(monkeypatch, tmp_path, capsys):
    provider = ScriptedProvider([text_turn("好")])

    run_loop(monkeypatch, provider, ["你好", "/exit"], tmp_path)

    assert "什么时候用 fork" not in provider.systems[0]
    capsys.readouterr()


def a_foldable_history(turns: int = 7) -> list[dict]:
    """几轮"读了个大文件"的历史。够长的那些会在冷的时候被折掉。"""
    messages: list[dict] = []
    for index in range(turns):
        call_id = f"t{index}"
        messages.append({"role": "user", "content": f"第 {index} 句"})
        messages.append(
            {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "id": call_id, "name": "read_file", "input": {}}
                ],
            }
        )
        messages.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": call_id,
                        "content": f"第 {index} 条结果" + "x" * (MIN_FOLD_CHARS + 100),
                    }
                ],
            }
        )
    return messages


def result_contents(messages: list[dict]) -> list[str]:
    """这几条消息里所有 tool_result 的 content，按出现顺序。"""
    found: list[str] = []
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if isinstance(block, dict) and block.get("type") == "tool_result":
                found.append(str(block.get("content")))
    return found


def a_bash_history(turns: int = 7) -> list[dict]:
    """几轮"跑了个长命令"的历史——`bash` 是那类"折掉之后重跑不等价"的工具。"""
    messages: list[dict] = []
    for index in range(turns):
        call_id = f"t{index}"
        messages.append({"role": "user", "content": f"第 {index} 句"})
        messages.append(
            {
                "role": "assistant",
                "content": [{"type": "tool_use", "id": call_id, "name": "bash", "input": {}}],
            }
        )
        messages.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": call_id,
                        "content": f"第 {index} 条结果" + "x" * (MIN_FOLD_CHARS + 100),
                    }
                ],
            }
        )
    return messages


def age_the_file(session: Session) -> None:
    """把会话文件做旧，装成"上一轮结束之后过了很久"——也就是缓存该冷了。

    `is_stale` 判的就是这个文件的 mtime（每轮结束都会 append 一次，所以它等于"上一轮什么
    时候结束的"）。做旧它比等 30 分钟强。
    """
    old = time.time() - FOLD_AFTER_SECONDS - 60
    os.utime(session.path, (old, old))


def test_a_stale_session_gets_folded_before_the_turn(monkeypatch, tmp_path):
    """会话文件很久没被写过 = 缓存大概冷了 = 那时候折是纯赚。

    折的判据只有"冷不冷"，不看多大——热的时候折是亏的（中间改一处会让整个前缀几乎全不中，
    见 known-issues D5）。
    """
    history = a_foldable_history()
    session = start(tmp_path, permission_mode="default", allow_shell=False)
    session.append(history)
    age_the_file(session)
    provider = ScriptedProvider([text_turn("好")])

    run_loop(
        monkeypatch,
        provider,
        ["接着说", "/exit"],
        tmp_path,
        session=session,
        messages=history,
        # 名单是从注册表现算的（`Tool.foldable`）：空注册表里一个可折的都没有。
        registry=default_tools(),
    )

    sent = result_contents(provider.calls[0])
    assert "已清除" in sent[0]  # 最早那条被折了
    assert sent[-1].startswith("第 6 条结果")  # 最近那条一字不动
    assert "已清除" not in sent[-1]


def test_a_fresh_session_is_not_folded(monkeypatch, tmp_path):
    """刚聊完接着聊，缓存还热着——这时候折会让整个前缀失效，是亏的。"""
    history = a_foldable_history()
    session = start(tmp_path, permission_mode="default", allow_shell=False)
    session.append(history)  # 刚写过，mtime 是新的
    provider = ScriptedProvider([text_turn("好")])

    run_loop(
        monkeypatch,
        provider,
        ["接着说", "/exit"],
        tmp_path,
        session=session,
        messages=history,
        # 注册表得带着可折的工具，不然这条会因为"压根没什么可折的"而假通过。
        registry=default_tools(),
    )

    assert "已清除" not in str(result_contents(provider.calls[0]))


def test_folding_does_not_duplicate_or_overwrite_what_lands_on_disk(monkeypatch, tmp_path):
    """**这一条守的是 `session.append(result.messages[len(state.messages):])` 那个偏移。**

    折叠只换内存里那份的内容、**条数一条不变**，所以那个切片仍然正好是"这一轮新加的几条"
    ——不会把折过的历史再写一遍。而且盘上拿回来的还是**原文**：折的是视图，不是文件。
    """
    history = a_foldable_history()
    session = start(tmp_path, permission_mode="default", allow_shell=False)
    session.append(history)
    age_the_file(session)
    provider = ScriptedProvider([text_turn("好")])

    run_loop(
        monkeypatch,
        provider,
        ["接着说", "/exit"],
        tmp_path,
        session=session,
        messages=history,
        registry=default_tools(),
    )

    # 先确认**真折过**：不然下面两条会因为"什么都没发生"而假通过
    assert "已清除" in result_contents(provider.calls[0])[0]

    _, _, stored = load(tmp_path, session.id)
    # 原来那些一条不多一条不少，加上这一轮的用户那句和回答那句
    assert len(stored) == len(history) + 2
    assert "已清除" not in str(stored)  # 盘上还是完整的原文


def test_folding_persists_unrecoverable_results_to_the_session_directory(
    monkeypatch, tmp_path
):
    """`bash` 的结果折掉之后重跑不等价，所以折之前先按 `tool_use_id` 落一份到会话目录。"""
    history = a_bash_history()
    session = start(tmp_path, permission_mode="default", allow_shell=False)
    session.append(history)
    age_the_file(session)
    provider = ScriptedProvider([text_turn("好")])

    run_loop(
        monkeypatch,
        provider,
        ["接着说", "/exit"],
        tmp_path,
        session=session,
        messages=history,
        registry=default_tools(),
    )

    saved = spill.session_dir(tmp_path, session.id) / "t0.txt"
    assert saved.is_file()
    assert saved.read_text(encoding="utf-8") == history[2]["content"][0]["content"]

    sent = result_contents(provider.calls[0])
    assert ".agent/tool-results/" in sent[0]
    assert 'id 传 "t0"' in sent[0]
    assert "重跑" not in sent[0]  # 别让模型去重跑一条有副作用的命令


def test_folding_never_overwrites_a_file_that_is_already_there(monkeypatch, tmp_path):
    """已经存在就不动它——折叠会反复重算，不能把之前那份覆盖掉。"""
    history = a_bash_history()
    session = start(tmp_path, permission_mode="default", allow_shell=False)
    session.append(history)
    age_the_file(session)
    existing = spill.session_dir(tmp_path, session.id) / "t0.txt"
    existing.parent.mkdir(parents=True)
    existing.write_text("原来是这份", encoding="utf-8")
    provider = ScriptedProvider([text_turn("好")])

    run_loop(
        monkeypatch,
        provider,
        ["接着说", "/exit"],
        tmp_path,
        session=session,
        messages=history,
        registry=default_tools(),
    )

    assert existing.read_text(encoding="utf-8") == "原来是这份"


def test_folding_a_bash_history_still_appends_only_new_messages(monkeypatch, tmp_path):
    """落盘是消息列表之外的动作——`session.append` 那个偏移一点没动。"""
    history = a_bash_history()
    session = start(tmp_path, permission_mode="default", allow_shell=False)
    session.append(history)
    age_the_file(session)
    provider = ScriptedProvider([text_turn("好")])

    run_loop(
        monkeypatch,
        provider,
        ["接着说", "/exit"],
        tmp_path,
        session=session,
        messages=history,
        registry=default_tools(),
    )

    assert "已清除" in result_contents(provider.calls[0])[0]
    _, _, stored = load(tmp_path, session.id)
    assert len(stored) == len(history) + 2
    assert "已清除" not in str(stored)


def test_the_one_shot_path_persists_before_folding(monkeypatch, tmp_path):
    history = a_bash_history()
    session = start(tmp_path, permission_mode="default", allow_shell=False)
    session.append(history)
    age_the_file(session)
    monkeypatch.setattr(cli, "AnthropicProvider", lambda: FakeOneShotProvider())

    cli.main_command(
        prompt="接着说",
        cwd=tmp_path,
        shell=False,
        fork=False,
        permission_mode="default",
        continue_=False,
        resume=session.id,
        list_=False,
        delete_="",
    )

    assert (spill.session_dir(tmp_path, session.id) / "t0.txt").is_file()


def test_the_loop_turns_background_mode_on(monkeypatch, tmp_path):
    """交互模式下 `ctx.allow_background` 必须是 True——否则 `run_in_background` 永远被忽略。"""
    seen: list[bool] = []

    def probe(kwargs, ctx):
        seen.append(ctx.allow_background)
        return "ok"

    tool = Tool(name="probe", description="", fn=probe, params={})
    provider = ScriptedProvider([tool_turn("t1", "probe", {}), text_turn("好了")])

    run_loop(
        monkeypatch,
        provider,
        ["试试", "/exit"],
        tmp_path,
        registry=ToolRegistry([tool]),
    )

    assert seen == [True]


class TestBackgroundReporting:
    """后台任务结束时要让模型自己开口，而不是等用户再问一句。"""

    def test_a_finished_task_wakes_the_idle_loop(self, monkeypatch, tmp_path, capsys):
        class ReportingProvider(ScriptedProvider):
            def __init__(self, responses):
                super().__init__(responses)
                self.called = threading.Event()

            def complete(self, messages, tools=None, system=None):
                self.called.set()
                return super().complete(messages, tools, system)

        provider = ReportingProvider([text_turn("后台任务跑完了：找到三处")])
        manager = TaskManager()
        release = threading.Event()

        def work(emit, cancel):
            release.wait(2)
            return AgentResult(final="找到三处", trace=[], messages=[])

        manager.start(
            work,
            label="[explore] ",
            agent_type="explore",
            description="找压缩逻辑",
            cwd=tmp_path,
            spill_dir=None,
        )

        # 读键盘用一个会阻塞的队列：主循环停在 events.get() 上，任务完成的事件才叫得醒它。
        # `ready` 由读键盘线程在真正开始等输入时置位——那时 `_chat` 已经把 on_finish 接上了。
        inputs: queue.Queue[str] = queue.Queue()
        ready = threading.Event()

        def prompt(*args, **kwargs) -> str:
            ready.set()
            return inputs.get()

        monkeypatch.setattr(cli.typer, "prompt", prompt)
        session = start(tmp_path, permission_mode="default", allow_shell=False)

        def feed() -> None:
            try:
                assert ready.wait(3)
                release.set()
                # 等模型真的被叫醒去汇报，再放 /exit——否则"任务事件"和"用户这一行"谁先到没准。
                assert provider.called.wait(3)
            finally:
                # 断言失败时也要放行，否则 `_chat` 会一直卡在等输入上，测试变成挂死而不是变红。
                inputs.put("/exit")

        threading.Thread(target=feed, daemon=True).start()

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
            tasks=manager,
        )

        assert "后台任务跑完了" in capsys.readouterr().out
        assert len(provider.calls) == 1
        # 完成通知和模型的话都要落进会话文件——下次 --resume 接上来时还看得见。
        _, _, history = load(tmp_path, session.id)
        assert any("后台任务" in str(message.get("content")) for message in history)
        assert any("后台任务跑完了" in str(message.get("content")) for message in history)

    def test_a_task_already_merged_into_a_turn_is_not_reported_again(
        self, monkeypatch, tmp_path
    ):
        """`take_notices` 取走即标记：主 agent 那一轮已经并进去的，空闲唤醒不再说一遍。"""
        manager = TaskManager()
        release = threading.Event()

        def work(emit, cancel):
            release.wait(2)
            return AgentResult(final="找到三处", trace=[], messages=[])

        manager.start(
            work,
            label="[explore] ",
            agent_type="explore",
            description="找压缩逻辑",
            cwd=tmp_path,
            spill_dir=None,
        )

        def probe(kwargs, ctx):
            return "ok"

        class ReleasingProvider(ScriptedProvider):
            def complete(self, messages, tools=None, system=None):
                if not release.is_set():
                    release.set()
                    manager.wait(manager.list()[0].id, timeout=2)
                return super().complete(messages, tools, system)

        provider = ReleasingProvider([tool_turn("t1", "probe", {}), text_turn("好了")])

        run_loop(
            monkeypatch,
            provider,
            ["跑一下", "/exit"],
            tmp_path,
            registry=ToolRegistry([Tool(name="probe", description="", fn=probe, params={})]),
            tasks=manager,
        )

        # 那一轮的第二发之前，`notices` 把结果并进了上下文；所以只有两发，没有额外的汇报回合。
        assert len(provider.calls) == 2


def a_write_request(path: str = "a.txt") -> PermissionRequest:
    return PermissionRequest(tool_name="write_file", tool_args={"path": path, "content": "x"})


class TestBackgroundApprovals:
    """后台任务排队等授权；读键盘线程在持有终端时处理它。"""

    def test_a_pending_request_is_denied_when_the_prompter_cannot_ask(
        self, monkeypatch, tmp_path, capsys
    ):
        """`NullPrompter` 是测试/一次性模式的缺省：它不问，直接拒绝，也不消耗脚本行。"""
        broker = PermissionBroker()
        item = broker.defer(a_write_request(), "[general-purpose] ")
        provider = ScriptedProvider([text_turn("好")])

        run_loop(monkeypatch, provider, ["你好", "/exit"], tmp_path, broker=broker)

        assert item.future.result(timeout=0.1).allowed is False
        assert "后台任务请求" in capsys.readouterr().out
        assert len(provider.calls) == 1  # 用户那一行照常处理

    def test_the_real_prompter_asks_and_the_answer_reaches_the_task(
        self, monkeypatch, tmp_path, capsys
    ):
        broker = PermissionBroker()
        item = broker.defer(a_write_request(), "[general-purpose] ")
        provider = ScriptedProvider([text_turn("好")])

        run_loop(
            monkeypatch,
            provider,
            ["y", "你好", "/exit"],
            tmp_path,
            prompter=TerminalPrompter(),
            broker=broker,
        )

        assert item.future.result(timeout=0.1) == Answer(True)
        # 预览走 stderr（和前台权限提示同一份渲染），用户那一行照常处理
        assert "要写 a.txt" in capsys.readouterr().err
        assert len(provider.calls) == 1

    def test_a_no_answer_denies_it(self, monkeypatch, tmp_path, capsys):
        broker = PermissionBroker()
        item = broker.defer(a_write_request(), "[general-purpose] ")
        provider = ScriptedProvider([text_turn("好")])

        run_loop(
            monkeypatch,
            provider,
            ["n", "你好", "/exit"],
            tmp_path,
            prompter=TerminalPrompter(),
            broker=broker,
        )

        assert item.future.result(timeout=0.1).allowed is False
        capsys.readouterr()

    def test_leaving_the_loop_closes_the_broker(self, monkeypatch, tmp_path):
        broker = PermissionBroker()

        run_loop(monkeypatch, ScriptedProvider([]), ["/exit"], tmp_path, broker=broker)

        # close 之后新的 submit 立刻拒绝，不会挂住一个已经没人处理的队列
        assert broker.submit(a_write_request(), "[x] ").allowed is False

    def test_leaving_the_loop_cancels_running_tasks(self, monkeypatch, tmp_path):
        manager = TaskManager()
        began = threading.Event()

        def work(emit, cancel):
            began.set()
            cancel.wait(2)
            return AgentResult(
                final=None, trace=[], messages=[], error="已取消"
            )

        task_id = manager.start(
            work,
            label="[explore] ",
            agent_type="explore",
            description="找压缩逻辑",
            cwd=tmp_path,
            spill_dir=None,
        )
        assert began.wait(2)

        run_loop(monkeypatch, ScriptedProvider([]), ["/exit"], tmp_path, tasks=manager)

        assert manager.wait(task_id, timeout=2)
        assert manager.get(task_id).status == "cancelled"
