"""斜杠命令：认出来就地办掉、改得动档位、认不出来也不发给模型。全部本地跑，不碰网络。"""

from __future__ import annotations

import pytest

from mini_code_agent.commands import (
    ChatState,
    Command,
    CommandContext,
    Commands,
    default_commands,
)
from mini_code_agent.commands.base import split_command
from mini_code_agent.commands.builtin import make_help
from mini_code_agent.console import console


def make_ctx(mode: str = "default") -> CommandContext:
    return CommandContext(console=console, state=ChatState(permission_mode=mode))


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
        for name in ("/help", "/permission_mode", "/exit"):
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
