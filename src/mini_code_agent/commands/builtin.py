"""自带的三条命令：看帮助、看/换权限档位、退出。

一条都不跑对话：这三件事要么是"改下一轮怎么跑"，要么是"把话说清楚"。
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from mini_code_agent.commands.base import Command, CommandContext, CommandOutcome
from mini_code_agent.permissions import PERMISSION_MODES, parse_permission_mode
from mini_code_agent.summarizing import MAX_HISTORY_CHARS, total_chars

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


def _relative(path: Path, cwd: Path) -> str:
    """能相对就相对。状态行里一条绝对路径要占掉半屏，而 cwd 本来就印在上面一行。"""
    try:
        return str(path.relative_to(cwd))
    except ValueError:
        # 不在 cwd 底下（接的是别的目录的会话之类）——那就照实印，别硬凑一个相对路径。
        return str(path)


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


def make_status() -> Command:
    """把这个循环现在的样子摊开：在哪儿跑、什么档位、历史多长、接的哪次会话。

    报的全是**循环自己手里**的东西。**不印模型名**：那是 provider 的，而 Provider 协议
    （providers/base.py）故意只声明了 complete 这一个方法——为了一行状态去扩那个接口，等于
    让所有假 provider（测试里那些）都跟着多背一个字段，不值当。真要看模型，命令行和 .env
    里就是。
    """

    def run(ctx: CommandContext, arg: str) -> CommandOutcome:
        session = ctx.session
        mode = ctx.state.permission_mode
        shell = "开（跑在一次性容器里，跑完就删）" if ctx.allow_shell else "关（--shell 才打开）"
        used = total_chars(ctx.state.messages)
        # 压过没有，用户看不出来——但它决定了模型还记不记得住开头的那些事。
        compacted = "；前面那段已经压成摘要了" if session.has_summary else ""

        ctx.console.print(f"工作目录：{ctx.cwd}", markup=False)
        ctx.console.print(f"权限档位：{mode}（{GLOSS.get(mode, '')}）", markup=False)
        ctx.console.print(f"命令执行：{shell}", markup=False)
        if session.opened:
            where = f"{_relative(session.path, ctx.cwd)}（落盘 {session.written} 条消息）"
        else:
            # 会话是懒建的：还没说过话，盘上就没有那个文件。这时印一个不存在的路径、配一句
            # "落盘 0 条消息"，是拿一个看起来很确切的东西骗人。
            where = "（还没落盘——说过话才建）"
        ctx.console.print(f"会话文件：{where}", markup=False)
        ctx.console.print(
            f"历史大小：{len(ctx.state.messages)} 条 / {used:,} 字符"
            f"（到 {MAX_HISTORY_CHARS:,} 才压）{compacted}",
            markup=False,
        )
        return "handled"

    return Command(name="status", run=run, summary="看在哪儿跑、什么档位、历史多长")


def make_compact() -> Command:
    """现在就把前面的对话压成一段摘要，不管它多长。

    自动那条路只看大小（`>12 万字符` 才动手，见 summarizing.py）。这一条是"我说了算"：接下来
    要问一件跟前面全无关的事，先让模型把前面那段收成一段摘要，不必背着整段历史走。

    压的是内存里那份历史，`record_summary` 会在同一步里往会话文件记一笔（`_compact` 内部
    做的），所以 `--resume` 接上来照样认——不是只活在这一次进程里。

    **一个代价**：要是原本就没几轮、总共几百字符，摘要回来可能比原文还长，这一下会让历史
    变大。不拦：手动入口的意思就是"我说了算"，而且这种情况本来就少见；真要拦，就得给
    "什么时候该压"再加一条地板，那等于把手动入口又变回自动判断。
    """

    def run(ctx: CommandContext, arg: str) -> CommandOutcome:
        if arg:
            # 参数不对就别去压：压一次是一次要花钱的模型调用。
            ctx.console.print(f"不接受参数（收到 {arg!r}）", style="red", markup=False)
            return "handled"
        note = ctx.compact()
        if note:
            # 压成功那几行由 _compact 自己打（它知道切在第几条、成没成），这里只说它没说到的。
            ctx.console.print(note, style="dim", markup=False)
        return "handled"

    return Command(name="compact", run=run, summary="现在就把前面的对话压成一段摘要")


def make_grants() -> Command:
    """看这一会话批准过什么，或者全清掉。

    **看得见自己批准过什么是它存在的全部理由**：Grants 平时只往里加、不往外看，想反悔只能
    重启进程。而"人是最后一道闸门"这个前提，靠的恰恰是不糊里糊涂地把闸门放开——尤其在
    accept edits 档下，一整个文件的改动就这么静悄悄地不再问了。
    """

    def run(ctx: CommandContext, arg: str) -> CommandOutcome:
        entries = sorted(ctx.grants.entries)
        if arg == "clear":
            if not entries:
                ctx.console.print("本来就是空的。", style="dim", markup=False)
                return "handled"
            ctx.grants.clear()
            ctx.console.print(f"清掉了 {len(entries)} 条，之后同一个文件会重新问你。", markup=False)
            return "handled"
        if arg:
            # 不兜底：参数认不出来就说一句，别猜他想干嘛。status 那条也一样。
            ctx.console.print(f"不认识的参数：{arg!r}（只认 clear）", style="red", markup=False)
            return "handled"

        if not entries:
            ctx.console.print("这一会话里还没有批准过什么。", markup=False)
            ctx.console.print(
                "（改文件时选 a 才会记在这儿；bash 从来不记——它是逐条点头的）",
                style="dim",
                markup=False,
            )
            return "handled"

        ctx.console.print("本会话里你批准过、不再问的调用（进程退出就没了）：\n", markup=False)
        # 工具名都是 ASCII，ljust 能对齐；路径那边对齐不了（中文宽度不一），不勉强。
        width = max(len(tool) for tool, _ in entries)
        for tool, path in entries:
            ctx.console.print(f"  {tool.ljust(width)}  {path}", markup=False)
        ctx.console.print("\n（/grants clear 可以全清掉）", style="dim", markup=False)
        return "handled"

    return Command(
        name="grants",
        run=run,
        summary="看这一会话批准过什么、不再问了；或者清掉",
        usage="[clear]",
    )


def make_exit() -> Command:
    """退出交互循环。"""

    def run(ctx: CommandContext, arg: str) -> CommandOutcome:
        return "stop"

    return Command(name="exit", run=run, summary="退出", aliases=("quit",))
