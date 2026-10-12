"""自带的三条命令：看帮助、看/换权限档位、退出。

一条都不跑对话：这三件事要么是"改下一轮怎么跑"，要么是"把话说清楚"。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from mini_code_agent.commands.base import Command, CommandContext, CommandOutcome
from mini_code_agent.permissions import PERMISSION_MODES, parse_permission_mode

# 只在类型标注里用得上，而且那边（registry）反过来要 import 这儿——真去 import 就成环了。
# 标注本身是懒的（`from __future__ import annotations`），不 import 也跑得起来。
if TYPE_CHECKING:
    from mini_code_agent.commands.registry import Commands

# 每一档意味着什么。**得写在界面上**：档位管的是"模型要不要停下来问"，用户该当场知道
# 刚才那句话放开了什么，而不是下一次被问到（或者再也没被问）才发现。
GLOSS: dict[str, str] = {
    "default": "改文件和跑命令都问你",
    "accept edits": "改文件不问，跑命令仍然问",
    "plan": "只读：改文件直接拒（bash 仍然要问）",
    "bypass permissions": "什么都不问——问这一层整个关掉了",
}

# 会话里能切的那几档。**bypass permissions 不在里面**，理由见 known-issues F4：它是唯一
# 一档把用户整个从回路里拿掉的，只能启动时显式开（`--permission_mode bypass`），开的时候
# 命令行还会打一行红字。在会话里敲一句话就把它打开，和"必须显式开、而且要看见那行提示"
# 差得太远。
# 按 PERMISSION_MODES 推出来，以后加第五档不必记得回来改这儿。
SWITCHABLE: tuple[str, ...] = tuple(m for m in PERMISSION_MODES if m != "bypass permissions")


def make_help(commands: Commands) -> Command:
    """列出所有命令。

    收的是**那张表本身**，不是某一刻的快照：`/help` 自己就在表里，先建表再登记它，
    它闭包抓着的是那个对象，之后 register 进来的命令它一样列得出。
    """

    def run(ctx: CommandContext, arg: str) -> CommandOutcome:
        ctx.console.print("这些命令由程序处理，不会发给模型：\n", markup=False)
        for command in commands.list():
            head = f"  /{command.name}"
            if command.aliases:
                head += "".join(f"（/{alias} 同）" for alias in command.aliases)
            if command.usage:
                head += f" {command.usage}"
            ctx.console.print(head, markup=False)
            if command.summary:
                ctx.console.print(f"      {command.summary}", style="dim", markup=False)
        return "handled"

    return Command(name="help", run=run, summary="把这页打出来")


def make_permission_mode() -> Command:
    """看当前的权限档位；带参数就换一档。"""

    def run(ctx: CommandContext, arg: str) -> CommandOutcome:
        current = ctx.state.permission_mode
        if not arg:
            ctx.console.print(f"当前的权限档位：{current}", markup=False)
            ctx.console.print(f"  {GLOSS.get(current, '')}", style="dim", markup=False)
            ctx.console.print(
                f"能在这儿切的：{'、'.join(SWITCHABLE)}", style="dim", markup=False
            )
            ctx.console.print(
                "（带空格的档位还有短写法：accept = accept edits）", style="dim", markup=False
            )
            return "handled"

        try:
            mode = parse_permission_mode(arg)
        except ValueError as exc:
            # 看不懂就直接说清楚，**不动**现在的档位。档位兜底比没有档位更糟：用户会以为
            # 切过去了，实际还在原来的那一档上跑。
            ctx.console.print(str(exc), style="red", markup=False)
            return "handled"

        if mode == "bypass permissions":
            ctx.console.print(
                "bypass permissions 不能在会话里切：它是唯一一档把用户从回路里整个拿掉的，"
                "只能启动时用 --permission_mode bypass 开（见 known-issues F4）。",
                style="red",
                markup=False,
            )
            return "handled"

        if mode == current:
            ctx.console.print(f"已经是 {mode} 了。", style="dim", markup=False)
            return "handled"

        ctx.state.permission_mode = mode
        ctx.console.print(f"档位换成 {mode}：{GLOSS.get(mode, '')}（下一轮起生效）", markup=False)
        return "handled"

    return Command(
        name="permission_mode",
        run=run,
        summary="看当前的权限档位，或者换一档",
        usage="[档位]",
    )


def make_exit() -> Command:
    """退出交互循环。"""

    def run(ctx: CommandContext, arg: str) -> CommandOutcome:
        return "stop"

    return Command(name="exit", run=run, summary="退出", aliases=("quit",))
