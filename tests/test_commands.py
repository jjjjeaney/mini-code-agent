"""斜杠命令：认出来就地办掉、改得动档位、认不出来也不发给模型。全部本地跑，不碰网络。"""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path

import pytest

from mini_code_agent.commands import (
    ChatState,
    Command,
    CommandContext,
    Commands,
    default_commands,
    render_session_list,
)
from mini_code_agent.commands.base import split_command
from mini_code_agent.commands.builtin import make_help
from mini_code_agent.console import console
from mini_code_agent.context_window import window_for
from mini_code_agent.permissions import Grants, PermissionRequest
from mini_code_agent.sessions import SESSIONS_DIR, Session, start
from mini_code_agent.summarizing import COMPACT_AT


def make_ctx(
    mode: str = "default",
    *,
    cwd: Path | None = None,
    allow_shell: bool = False,
    session: Session | None = None,
    grants: Grants | None = None,
    messages: list[dict] | None = None,
    compact: Callable[[], str | None] | None = None,
    window: int | None = None,
    context_tokens: int | None = None,
    context_covered: int = 0,
) -> CommandContext:
    """一份手边的东西。全给上默认值，测哪一块就只改哪一块。

    `compact` 默认是个"什么都不做、什么都不说"的桩——真那个要连 provider，在 cli 里。
    `window` 默认按"未知模型"那一档（见 context_window），所以是 40 万而不是 0——印出
    "窗口 0" 那种数谁看都不对。
    """
    state = ChatState(permission_mode=mode, messages=list(messages or []))
    if context_tokens is not None:
        state.remember_context(context_tokens, context_covered)
    return CommandContext(
        console=console,
        state=state,
        window=window_for(None) if window is None else window,
        cwd=cwd or Path("proj"),
        allow_shell=allow_shell,
        # opened=True：这就是"盘上已经有一次会话"的样子。没落盘的那种由 /status 那条单独的
        # 用例去测（它说的是另一句话）。
        session=session
        or Session(id="s1", path=Path("proj/.agent/sessions/s1.jsonl"), opened=True),
        grants=grants if grants is not None else Grants(),
        compact=compact if compact is not None else (lambda: None),
    )


def approved(*paths: str, tool: str = "write_file") -> Grants:
    """一份"本会话批过这几个文件"的 Grants，走的是真的 remember（所以键是归一化过的）。"""
    grants = Grants()
    for path in paths:
        grants.remember(PermissionRequest(tool_name=tool, tool_args={"path": path}))
    return grants


def a_command(name: str, **kwargs) -> Command:
    """一条不做任何事的命令，只用来占名字。"""
    return Command(name=name, run=lambda ctx, arg: "handled", **kwargs)


class TestDispatch:
    def test_a_plain_line_is_left_to_the_caller(self, capsys):
        """不是命令就返回 not_a_command，让调用方照常发给模型。"""
        assert default_commands().dispatch("帮我看看这个文件", make_ctx()) == "not_a_command"
        assert capsys.readouterr().out == ""

    def test_a_line_starting_with_a_slash_is_never_forwarded(self, capsys):
        """以 `/` 开头就是这个命名空间：打错了只该得到一句"没有这个命令"。"""
        assert default_commands().dispatch("/hlep", make_ctx()) == "handled"
        out = capsys.readouterr().out
        assert "/hlep" in out
        assert "没有发给模型" in out  # 用户得知道那句话没递出去

    def test_a_bare_slash_is_an_unknown_command(self, capsys):
        assert default_commands().dispatch("/", make_ctx()) == "handled"
        assert "这个命令" in capsys.readouterr().out

    def test_the_name_is_case_insensitive(self, capsys):
        ctx = make_ctx()
        assert default_commands().dispatch("/PERMISSION_MODE plan", ctx) == "handled"
        assert ctx.state.permission_mode == "plan"


class TestHelp:
    def test_it_lists_every_command(self, capsys):
        default_commands().dispatch("/help", make_ctx())

        out = capsys.readouterr().out
        wanted = (
            "/help",
            "/permission_mode",
            "/status",
            "/sessions",
            "/compact",
            "/grants",
            "/exit",
        )
        for name in wanted:
            assert name in out
        assert "/quit" in out  # 别名也列出来，不然没人知道它存在

    def test_it_lists_commands_registered_after_itself(self, capsys):
        """`/help` 抓的是那张表本身、不是快照——它是第一条注册的，别的一条都还没进来。"""
        commands = Commands()
        commands.register(make_help(commands))
        commands.register(a_command("later", summary="后加的"))

        commands.dispatch("/help", make_ctx())

        assert "later" in capsys.readouterr().out


class TestPermissionMode:
    def test_no_argument_reports_the_current_one(self, capsys):
        default_commands().dispatch("/permission_mode", make_ctx("plan"))

        out = capsys.readouterr().out
        assert "plan" in out
        assert "只读" in out  # 光有个名字不算说清楚，档位是"会问/不会问"的意思

    def test_it_switches_the_mode(self, capsys):
        ctx = make_ctx()
        assert default_commands().dispatch("/permission_mode plan", ctx) == "handled"

        assert ctx.state.permission_mode == "plan"
        assert "plan" in capsys.readouterr().out

    def test_the_short_form_works(self, capsys):
        """`accept` 是 `accept edits` 的短写法，跟命令行上一样。"""
        ctx = make_ctx()
        default_commands().dispatch("/permission_mode accept", ctx)

        assert ctx.state.permission_mode == "accept edits"

    def test_a_mode_that_makes_no_sense_changes_nothing(self, capsys):
        """看不懂的词**不兜底**：兜底的后果是用户以为切过去了，实际还在原来那档上跑。"""
        ctx = make_ctx("plan")
        assert default_commands().dispatch("/permission_mode 随便写点什么", ctx) == "handled"

        assert ctx.state.permission_mode == "plan"
        assert "不是权限模式" in capsys.readouterr().out

    def test_bypass_cannot_be_switched_on_from_within(self, capsys):
        """F4：唯一一档把用户整个从回路里拿掉的，只能启动时显式开。"""
        ctx = make_ctx()
        assert default_commands().dispatch("/permission_mode bypass", ctx) == "handled"

        assert ctx.state.permission_mode == "default"
        assert "F4" in capsys.readouterr().out

    def test_switching_to_the_mode_it_is_already_on_says_so(self, capsys):
        ctx = make_ctx("plan")
        default_commands().dispatch("/permission_mode plan", ctx)

        assert "已经是" in capsys.readouterr().out


class TestStatus:
    def test_it_reports_where_it_runs_and_under_what(self, capsys, tmp_path):
        default_commands().dispatch("/status", make_ctx("plan", cwd=tmp_path))

        out = capsys.readouterr().out
        assert f"工作目录：{tmp_path}" in out
        assert "权限档位：plan" in out
        assert "只读" in out  # 档位附了它意味着什么

    @pytest.mark.parametrize("allow_shell, expected", [(True, "开"), (False, "关")])
    def test_it_says_whether_commands_can_run(self, capsys, allow_shell, expected):
        default_commands().dispatch("/status", make_ctx(allow_shell=allow_shell))

        assert f"命令执行：{expected}" in capsys.readouterr().out

    def test_it_reports_the_session_and_how_much_history_there_is(self, capsys):
        session = Session(
            id="abcd", path=Path("proj/.agent/sessions/abcd.jsonl"), opened=True, written=7
        )
        ctx = make_ctx(session=session, messages=[{"role": "user", "content": "x" * 100}])

        default_commands().dispatch("/status", ctx)

        out = capsys.readouterr().out
        assert "会话文件：.agent" in out  # 相对 cwd 印，不占半屏
        assert "abcd.jsonl" in out
        assert "落盘 7 条消息" in out
        assert "历史大小：1 条" in out
        # 没有真用过接口（没锚点），所以只能说"估算"，再报出窗口和阈值
        assert "token（估算）" in out
        assert f"窗口 {window_for(None):,}" in out
        assert f"到 {int(window_for(None) * COMPACT_AT):,} 才压" in out

    def test_it_says_when_the_number_is_real_not_estimated(self, capsys):
        """锚点是上一发真请求报的——那就不是估的，这个字得改，不然这个数没法信。"""
        ctx = make_ctx(
            messages=[{"role": "user", "content": "x"}],
            context_tokens=120_000,
            context_covered=1,
        )

        default_commands().dispatch("/status", ctx)

        assert "token（真数）" in capsys.readouterr().out

    def test_it_says_the_session_has_not_landed_yet(self, capsys):
        """会话是懒建的：还没说过话，盘上就没有那个文件——不该印一个不存在的路径配"落盘 0 条"。"""
        not_yet = Session(id="s1", path=Path("proj/.agent/sessions/s1.jsonl"))

        default_commands().dispatch("/status", make_ctx(session=not_yet))

        out = capsys.readouterr().out
        assert "还没落盘" in out
        assert "s1.jsonl" not in out

    def test_it_says_when_the_history_has_been_compacted(self, capsys):
        """压过没有，光看对话是看不出来的——但它决定了开头那些事模型还记不记得住。"""
        session = Session(id="s1", path=Path("s1.jsonl"), has_summary=True)

        default_commands().dispatch("/status", make_ctx(session=session))

        assert "已经压成摘要了" in capsys.readouterr().out

    def test_it_says_nothing_about_summaries_when_there_were_none(self, capsys):
        default_commands().dispatch("/status", make_ctx())

        assert "摘要" not in capsys.readouterr().out


class TestCompact:
    def test_it_runs_the_compaction(self):
        calls: list[bool] = []
        ctx = make_ctx(compact=lambda: calls.append(True) or None)

        default_commands().dispatch("/compact", ctx)

        assert calls == [True]

    def test_it_stays_quiet_when_the_compaction_has_nothing_to_add(self, capsys):
        """压成功那几行由 _compact 自己打（它知道切在第几条），命令不重复说一遍。"""
        default_commands().dispatch("/compact", make_ctx(compact=lambda: None))

        assert capsys.readouterr().out == ""

    def test_it_relays_what_the_compaction_had_to_say(self, capsys):
        default_commands().dispatch("/compact", make_ctx(compact=lambda: "没什么可压的"))

        assert "没什么可压的" in capsys.readouterr().out

    def test_an_argument_is_refused(self, capsys):
        calls: list[bool] = []
        ctx = make_ctx(compact=lambda: calls.append(True) or None)

        default_commands().dispatch("/compact 随便写点什么", ctx)

        assert calls == []  # 参数不对就别压——压一次是一次要花钱的模型调用
        assert "不接受参数" in capsys.readouterr().out


class TestGrants:
    def test_it_lists_what_has_been_approved(self, capsys):
        default_commands().dispatch("/grants", make_ctx(grants=approved("src/a.py", "src/b.py")))

        out = capsys.readouterr().out
        assert "write_file" in out
        # 键里的路径是归一化过的（Windows 上是反斜杠），照着那个形式断言
        assert os.path.normpath("src/a.py") in out
        assert os.path.normpath("src/b.py") in out

    def test_it_says_so_when_nothing_has_been_approved(self, capsys):
        default_commands().dispatch("/grants", make_ctx())

        assert "还没有批准过" in capsys.readouterr().out

    def test_clear_empties_it(self, capsys):
        """清的是循环里那个 Grants 本身，不是它的一份副本——所以下一轮真的会重新问。"""
        grants = approved("src/a.py")

        default_commands().dispatch("/grants clear", make_ctx(grants=grants))

        assert grants.entries == set()
        assert "清掉了 1 条" in capsys.readouterr().out

    def test_clear_on_an_empty_one_says_so(self, capsys):
        default_commands().dispatch("/grants clear", make_ctx())

        assert "本来就是空的" in capsys.readouterr().out

    def test_an_argument_other_than_clear_is_refused(self, capsys):
        default_commands().dispatch("/grants nope", make_ctx())

        assert "只认 clear" in capsys.readouterr().out


class TestExit:
    @pytest.mark.parametrize("line", ["/exit", "/quit"])
    def test_it_stops_the_loop(self, line):
        assert default_commands().dispatch(line, make_ctx()) == "stop"


class TestParsing:
    def test_the_argument_keeps_its_spaces(self):
        """`accept edits` 中间那个空格是名字的一部分，拆细了还得拼回去。"""
        assert split_command("/permission_mode accept edits") == (
            "permission_mode",
            "accept edits",
        )

    @pytest.mark.parametrize(
        "line",
        ["/help", "/help   ", "/  help", "/\thelp\t"],
    )
    def test_the_name_stops_at_the_first_blank(self, line):
        assert split_command(line) == ("help", "")


class TestRegistration:
    def test_a_duplicate_name_is_rejected(self):
        commands = Commands([a_command("x")])
        with pytest.raises(ValueError, match="已经注册过"):
            commands.register(a_command("x"))

    def test_an_alias_colliding_with_another_name_is_rejected(self):
        """别名和正名同权，所以撞了也一样要拦——不然查出来的是哪条全看注册顺序。"""
        commands = Commands([a_command("a", aliases=("b",))])
        with pytest.raises(ValueError, match="已经注册过"):
            commands.register(a_command("b"))

    def test_an_alias_is_not_listed_as_its_own_command(self):
        commands = Commands([a_command("a", aliases=("b",))])

        assert [command.name for command in commands.list()] == ["a"]

    def test_a_name_with_a_blank_in_it_is_rejected(self):
        """名字里带空白就永远查不到——分派时按空白切名字。"""
        with pytest.raises(ValueError, match="只能是字母"):
            Command(name="permission mode", run=lambda ctx, arg: "handled")

    def test_every_default_command_can_be_reached_by_its_aliases(self):
        commands = default_commands()
        for command in commands.list():
            for alias in command.aliases:
                assert commands.by_name[alias] is command


def a_spoken_session(tmp_path, said: str = "给我看看那个文件"):
    session = start(tmp_path, permission_mode="default", allow_shell=False)
    session.append([{"role": "user", "content": said}])
    return session


class TestSessionList:
    """会话列表的渲染。启动参数 --list 和以后的 /sessions 共用这一份。"""

    def test_it_lists_what_is_there(self, capsys, tmp_path):
        session = a_spoken_session(tmp_path)

        render_session_list(console, tmp_path)

        out = capsys.readouterr().out
        assert session.id in out
        assert "1 条" in out
        assert "给我看看那个文件" in out
        assert "--resume" in out  # 列出来就是为了让你知道该接哪个

    def test_it_says_so_when_there_has_never_been_one(self, capsys, tmp_path):
        render_session_list(console, tmp_path)

        assert "还没有过会话" in capsys.readouterr().out

    def test_the_ones_that_never_said_anything_are_counted_not_listed(self, capsys, tmp_path):
        """空壳是每次启动的副产品，逐行列出来会把列表淹掉——报个数就够了。"""
        spoken = a_spoken_session(tmp_path)
        (tmp_path / SESSIONS_DIR / "29991231-235959.jsonl").write_text(
            '{"type": "meta", "cwd": "."}', encoding="utf-8"
        )

        render_session_list(console, tmp_path)

        out = capsys.readouterr().out
        assert spoken.id in out
        assert "29991231-235959" not in out
        assert "另有 1 次没说过话的" in out

    def test_a_compacted_session_is_marked(self, capsys, tmp_path):
        session = a_spoken_session(tmp_path)
        session.record_summary(1, "压过了")

        render_session_list(console, tmp_path)

        assert "压过" in capsys.readouterr().out

    def test_a_session_that_was_never_compacted_is_not_marked(self, capsys, tmp_path):
        a_spoken_session(tmp_path)

        render_session_list(console, tmp_path)

        assert "压过" not in capsys.readouterr().out


class TestSessionsCommand:
    """/sessions：跟启动参数的 --list 是同一份渲染，多一个"我在这儿"的星号。"""

    def test_it_marks_the_session_you_are_in(self, capsys, tmp_path):
        mine = a_spoken_session(tmp_path, "我这次说的")
        theirs = a_spoken_session(tmp_path, "另一次的")

        default_commands().dispatch("/sessions", make_ctx(cwd=tmp_path, session=mine))

        out = capsys.readouterr().out
        assert f"* {mine.id}" in out
        assert f"* {theirs.id}" not in out  # 星号只打在这一次上

    def test_it_says_so_when_this_session_has_not_landed(self, capsys, tmp_path):
        """还没说过话的这次盘上没有、列不出来——那就说明一句，别让人在单子上找自己。"""
        not_yet = Session(id="s1", path=tmp_path / "s1.jsonl")  # opened 默认 False

        default_commands().dispatch("/sessions", make_ctx(cwd=tmp_path, session=not_yet))

        assert "还没落盘" in capsys.readouterr().out

    def test_an_argument_is_refused(self, capsys, tmp_path):
        default_commands().dispatch("/sessions 随便写点什么", make_ctx(cwd=tmp_path))

        assert "不接受参数" in capsys.readouterr().out
