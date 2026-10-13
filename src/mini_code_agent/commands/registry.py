"""命令表：登记命令，并把用户敲的那一行派发出去。"""

from __future__ import annotations

from collections.abc import Iterable

from mini_code_agent.commands.base import Command, CommandContext, Outcome, split_command
from mini_code_agent.commands.builtin import (
    make_compact,
    make_exit,
    make_grants,
    make_help,
    make_permission_mode,
    make_status,
)


class Commands:
    """一张名字到命令的表。就是查名字，不含别的。

    别名和正名同权：注册进同一张表，查到的都是同一条命令。`list()` 列的是正名，别名不
    重复算一个——不然 `/quit` 会被当成另一条命令列出来。
    """

    def __init__(self, commands: Iterable[Command] | None = None) -> None:
        self.by_name: dict[str, Command] = {}
        self.all: list[Command] = []
        for command in commands or ():
            self.register(command)

    def register(self, command: Command) -> Command:
        """登记一条命令。重名（包括跟别人的别名撞）直接报错，不让后来的悄悄顶掉先来的。"""
        names = (command.name, *command.aliases)
        for name in names:
            if name in self.by_name:
                raise ValueError(f"命令 {name!r} 已经注册过了")
        self.all.append(command)
        for name in names:
            self.by_name[name] = command
        return command

    def list(self) -> list[Command]:
        """按登记顺序列出所有命令。"""
        return list(self.all)

    def dispatch(self, line: str, ctx: CommandContext) -> Outcome:
        """过一遍用户敲的这一行。

        认出命令就在这儿办掉（返回 `handled` / `stop`）；**不以 `/` 开头**返回
        `not_a_command`，调用方照常把它发给模型。

        以 `/` 开头但表里没有的那种，就地报一句、返回 `handled`——**不发给模型**。以 `/`
        开头是这个命名空间的全部约定：`/hlep` 该得到的是一句"没有这个命令"，而不是当成提问
        递过去（模型多半会顺着答，用户还以为命令生效了）。代价是拿 `/` 开头的路径开头的那
        句话也会被拦下，所以报错里点明这一行没有发出去。
        """
        if not line.startswith("/"):
            return "not_a_command"
        name, argument = split_command(line)
        # 名字不计大小写——`/Help` 和 `/help` 是同一条。参数原样留着，那是用户自己的话，
        # 折不折由命令自己定（档位那边 parse_permission_mode 本来就认大小写）。
        command = self.by_name.get(name.lower())
        if command is None:
            shown = f"/{name}" if name else line
            ctx.console.print(
                f"没有 {shown} 这个命令（这一行没有发给模型）。/help 看看有哪些。",
                style="red",
                markup=False,
            )
            return "handled"
        return command.run(ctx, argument)


def default_commands() -> Commands:
    """装配一份可以直接用的命令表。

    每次调用都返回新实例，跟 default_tools() 一个道理：调用方想再注册自己的命令时，
    不会污染到别人手里那份。

    顺序就是 `/help` 里的顺序，所以 `/help` 放第一条。`/status` 跟在 `/permission_mode` 后面
    ——两条都是"我现在在哪一档"那一类，看完档位顺手下一条就是它。
    """
    commands = Commands()
    commands.register(make_help(commands))
    commands.register(make_permission_mode())
    commands.register(make_status())
    commands.register(make_compact())
    commands.register(make_grants())
    commands.register(make_exit())
    return commands
