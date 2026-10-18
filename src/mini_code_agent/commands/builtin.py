"""自带的命令：帮助、档位、fork 开关、状态、会话、压缩、授权、退出。

一条都不跑对话：它们要么是"改下一轮怎么跑"，要么是"把话说清楚"。
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from rich.console import Console

from mini_code_agent.commands.base import Command, CommandContext, CommandOutcome
from mini_code_agent.permissions import PERMISSION_MODES, parse_permission_mode
from mini_code_agent.sessions import SESSIONS_DIR, list_sessions
from mini_code_agent.summarizing import COMPACT_AT, context_size

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


def render_session_list(console: Console, cwd: Path, *, current: str | None = None) -> None:
    """把这个目录下有过哪些会话摆出来。

    落在命令层而不是 cli.py：这是 CLI 的表现层，而且**启动参数 `--list` 和 `/sessions`
    共用同一份**——同一件事印两遍，迟早有一边先长歪。

    `current` 是"我现在这一次"的 id，会话里跑的时候（`/sessions`）才有——启动参数那次根本
    还没有会话。给了就在对应的那一行前面打个 `*`。

    没说过话的那些不逐行列出来，只在末尾报个数：它们是每次启动的副产品，占一半篇幅会让
    列表没法看。列表里也**不放档位**——`bypass permissions` 一个词就把行宽撑爆了，而档位
    本来就不跟着历史走（接上会话时那句"这次按 X 走"就是为这个），列它意义不大。
    """
    infos = list_sessions(cwd)
    spoken = [info for info in infos if info.messages]
    silent = len(infos) - len(spoken)

    if not spoken:
        console.print("这个目录下还没有过会话。（说过话才会留下会话文件）", markup=False)
        return

    where = f"{SESSIONS_DIR} 里，* 是这次" if current else f"都在 {SESSIONS_DIR} 里"
    console.print(f"这个目录下有过 {len(spoken)} 次会话（{where}）：", markup=False)
    console.print()
    for info in spoken:
        # 条数右对齐、"压过"占两格：这样这几栏能对齐（中文宽度不一，预览那栏就没法较真了）。
        star = "*" if info.id == current else " "
        packed = "压过" if info.has_summary else "    "
        row = f"{star} {info.id}  {info.messages:>3} 条  {packed}  「{info.preview}」"
        console.print(row, markup=False)
    if silent:
        console.print(f"\n（另有 {silent} 次没说过话的，略过）", style="dim", markup=False)
    console.print(
        "\n接上某一次：mca --resume <id>；mca --continue 接最近一次有内容的"
        "\n删掉某一次：mca --delete <id>（删之前会把要删的那次摆给你看，问过才动手）",
        style="dim",
        markup=False,
    )


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


def make_fork() -> Command:
    """看 fork 开没开；带参数就打开或关掉。

    **切换要付一次重建前缀的钱**：fork 工具在不在工具表里、提示词里有没有那段，都是请求开头的
    一部分，而工具表渲染在 messages 前面——改它等于整套前缀重算。所以这件事得当场说清楚，
    不能悄悄换掉。
    """

    def run(ctx: CommandContext, arg: str) -> CommandOutcome:
        current = ctx.state.allow_fork
        if not arg:
            ctx.console.print(f"fork 现在是{'开' if current else '关'}着的。", markup=False)
            ctx.console.print("  /fork on 打开、/fork off 关掉", style="dim", markup=False)
            return "handled"

        choice = arg.strip().lower()
        if choice not in ("on", "off"):
            # 看不懂就**不动**：静默取反比不切更糟，用户会以为切过去了。
            ctx.console.print(
                f"看不懂 {arg!r}——用 /fork on 或 /fork off。", style="red", markup=False
            )
            return "handled"

        wanted = choice == "on"
        if wanted == current:
            ctx.console.print(
                f"fork 已经是{'开' if current else '关'}着的了。", style="dim", markup=False
            )
            return "handled"

        ctx.state.allow_fork = wanted
        ctx.console.print(f"fork 换成了{'开' if wanted else '关'}（下一轮起生效）", markup=False)
        ctx.console.print(
            "  工具表和提示词都变了，下一轮的前缀缓存会从头算——那一发按原价。",
            style="dim",
            markup=False,
        )
        return "handled"

    return Command(
        name="fork",
        run=run,
        summary="看 fork 开没开，或者打开/关掉（会重建前缀缓存）",
        usage="[on|off]",
    )


def make_status() -> Command:
    """把这个循环现在的样子摊开：在哪儿跑、什么档位、fork 开没开、历史多长、接的哪次会话。

    报的全是**循环自己手里**的东西。**不印模型名**：那是 provider 的，而 Provider 协议
    （providers/base.py）故意只声明了 complete 这一个方法——为了一行状态去扩那个接口，等于
    让所有假 provider（测试里那些）都跟着多背一个字段，不值当。真要看模型，命令行和 .env
    里就是。
    """

    def run(ctx: CommandContext, arg: str) -> CommandOutcome:
        session = ctx.session
        mode = ctx.state.permission_mode
        shell = "开（跑在一次性容器里，跑完就删）" if ctx.allow_shell else "关（--shell 才打开）"
        fork = (
            "开（可以派继承当前对话的子 agent）"
            if ctx.state.allow_fork
            else "关（/fork on 打开）"
        )
        used = context_size(
            ctx.state.messages,
            tokens=ctx.state.context_tokens,
            covered=ctx.state.context_covered,
        )
        # 有锚点就是**真数**（上一发请求报的输入总量），没有就是按字符**估**的。这两个差着
        # 几十个百分点，不标出来，这个数就没法信。
        how = "真数" if ctx.state.context_tokens is not None else "估算"
        # 压过没有，用户看不出来——但它决定了模型还记不记得住开头的那些事。
        compacted = "；前面那段已经压成摘要了" if session.has_summary else ""

        ctx.console.print(f"工作目录：{ctx.cwd}", markup=False)
        ctx.console.print(f"权限档位：{mode}（{GLOSS.get(mode, '')}）", markup=False)
        ctx.console.print(f"命令执行：{shell}", markup=False)
        ctx.console.print(f"fork：{fork}", markup=False)
        if session.opened:
            where = f"{_relative(session.path, ctx.cwd)}（落盘 {session.written} 条消息）"
        else:
            # 会话是懒建的：还没说过话，盘上就没有那个文件。这时印一个不存在的路径、配一句
            # "落盘 0 条消息"，是拿一个看起来很确切的东西骗人。
            where = "（还没落盘——说过话才建）"
        ctx.console.print(f"会话文件：{where}", markup=False)
        ctx.console.print(
            f"历史大小：{len(ctx.state.messages)} 条 / 约 {used:,} token（{how}）"
            f"（窗口 {ctx.window:,}，到 {int(ctx.window * COMPACT_AT):,} 才压）{compacted}",
            markup=False,
        )
        return "handled"

    return Command(
        name="status", run=run, summary="看在哪儿跑、什么档位、fork 开没开、历史多长"
    )


def make_sessions() -> Command:
    """列出这个目录下有过哪些会话，当前这次打个 `*`。

    渲染和启动参数的 `--list` 是同一份（见 render_session_list）：同一件事印两遍，迟早有一边
    先长歪。区别只在于会话里跑的时候**知道"我是哪一次"**，所以能标出来。

    （列出来是为了让你知道该 `--resume` 哪个——会话里不能直接切过去，那是另一摊事。）
    """

    def run(ctx: CommandContext, arg: str) -> CommandOutcome:
        if arg:
            ctx.console.print(f"不接受参数（收到 {arg!r}）", style="red", markup=False)
            return "handled"
        # 还没落盘的会话盘上没有，列不出来、也就标不出来——那就说明一句，别让人在单子上找自己。
        current = ctx.session.id if ctx.session.opened else None
        render_session_list(ctx.console, ctx.cwd, current=current)
        if current is None:
            ctx.console.print(
                "（这次会话还没落盘——说过话才建，所以上面没有它）", style="dim", markup=False
            )
        return "handled"

    return Command(name="sessions", run=run, summary="列出这个目录下有过哪些会话")


def make_compact() -> Command:
    """现在就把前面的对话压成一段摘要，不管它多长。

    自动那条路看的是"占了这个模型的窗口多少"（到 80% 才动手，见 summarizing.py）。这一条是
    "我说了算"：接下来要问一件跟前面全无关的事，先让模型把前面那段收成一段摘要，不必背着
    整段历史走。

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
