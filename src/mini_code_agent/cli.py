"""命令行入口。"""

from __future__ import annotations

import queue
import threading
from pathlib import Path

import typer

from mini_code_agent.commands import (
    ChatState,
    CommandContext,
    default_commands,
    render_session_list,
)
from mini_code_agent.console import console
from mini_code_agent.context_window import window_for
from mini_code_agent.core.agent import AgentResult, runagent
from mini_code_agent.core.subagent import make_agent_tool
from mini_code_agent.core.tasks import TaskManager
from mini_code_agent.folding import PersistText, fold_if_stale
from mini_code_agent.interaction import PermissionBroker, ask_permission
from mini_code_agent.memory import sync_index
from mini_code_agent.permissions import PERMISSION_MODES, Grants, parse_permission_mode
from mini_code_agent.prompting import TerminalPrompter, make_ask_user
from mini_code_agent.prompts import Answer, NullPrompter, Prompter
from mini_code_agent.providers.anthropic import DEFAULT_MODEL, AnthropicProvider
from mini_code_agent.providers.base import Provider
from mini_code_agent.sessions import Session, delete, describe, load, start, summary_message
from mini_code_agent.skills import sync_index as sync_skill_index
from mini_code_agent.summarizing import KEEP_RECENT_TURNS, context_size, cut_point, summarize
from mini_code_agent.system_prompt import build_system_prompt
from mini_code_agent.tools import ToolRegistry, default_tools, spill

app = typer.Typer(add_completion=False, help="最小可用的 coding agent。")


def _compact(
    messages: list[dict],
    provider: Provider,
    session: Session,
    *,
    size: int,
    window: int,
    force: bool = False,
) -> list[dict]:
    """压一次：把旧的那段换成一段摘要，并记进会话。

    `size` 是这段历史现在有多大（token），`window` 是模型的上下文窗口——够到 `COMPACT_AT`
    才压（见 summarizing.cut_point）。`force=True` 是 `/compact`：用户点名要压，就不看大小了，
    两个参数照旧收着（调用方手边本来就有），由 cut_point 决定用不用。

    放在**轮与轮之间**做，不放 runagent 里：那边有个前提——"返回的 messages = 传进去的 +
    追加的那几条"（下面就是靠 result.messages[len(messages):] 落盘的）。压缩会改掉前缀，
    正好把这个前提破坏掉。

    **摘要调不动就当没压**，照常跑这一轮——绝不能因为压缩把会话卡死。

    **返回值是不是同一个 list 就是"压没压"的信号**：没压时原样返回传进来那个（`is` 相等），
    压了返回一个新的。调用方靠它决定要不要把锚点作废（见 ChatState.forget_context）。
    """
    cut = cut_point(messages, size=size, window=window, force=force)
    if cut is None:
        return messages
    # 这底下是一次网络往返，慢起来好几秒。自动那条路上它是背景行为，手动那条路上它是用户
    # 刚敲下去的命令——两种都不该干等着一声不吭。
    console.print("（正在把前面的对话压成摘要…）", style="dim", markup=False)
    try:
        summary = summarize(messages[:cut], provider)
    except Exception as exc:
        # 网络、限流、模型抽风——都只是"这次没压成"，不是会话的错。
        console.print(f"（前面的对话没能压成摘要：{type(exc).__name__}，这次先不压）", style="dim")
        return messages
    if not summary:
        return messages

    session.record_summary(cut, summary)
    console.print(f"（把前面 {cut} 条消息压成了一段摘要）", style="dim", markup=False)
    return [summary_message(summary), *messages[cut:]]


def _persister(cwd: Path, session: Session) -> PersistText:
    """折叠时的落盘动作：把正文写进**这次会话**的目录，回给模型一个真能用的 id。

    **`session.id` 要在回调里现读**：会话文件名撞名时 `Session._open` 会把它改掉，提前算死
    目录会把同一批文件劈到两个地方。

    回的是（给人看的路径, 落盘后的真实文件名）——后者可能不是原 id（形状不合格的会被换成
    `<工具名>-<摘要>`），占位符里必须写这一个。
    """

    def persist(tool_name: str, tool_use_id: str, text: str) -> tuple[str, str] | None:
        saved = spill.save_into(
            spill.session_dir(cwd, session.id), tool_name, tool_use_id, text
        )
        if saved is None:
            return None
        return spill.display_path(cwd, saved), saved.stem

    return persist


def _chat(
    provider: Provider,
    registry: ToolRegistry,
    session: Session,
    messages: list[dict],
    *,
    cwd: Path,
    allow_shell: bool,
    permission_mode: str,
    prompter: Prompter,
    grants: Grants,
    model: str | None = None,
    allow_fork: bool = False,
    tasks: TaskManager | None = None,
    broker: PermissionBroker | None = None,
) -> None:
    """交互循环：读一行、跑一轮、把消息接下去并落盘。

    **每轮重拼一次 system prompt**：记忆和 `AGENTS.md` 都可能在中途变（模型刚 `remember` 了
    一条），拼一次用到底的话，这一轮记下的东西要重启进程才看得见。档位和历史同理——前者
    `/permission_mode` 能换，后者 `/status` 要报它多长，所以这两个从局部量挪进了 `state`，
    和提示词一起每轮现读。传进来的 `permission_mode` 和 `messages` 只是这次运行的**起始值**。

    这一行先过命令表（见 commands/）：认出来的命令就地办掉、**不发给模型**，也不落盘——
    命令是用户对程序说的话，不是对话的一部分，接上历史时不该看见它。

    **读键盘在另一条线程上**（见下面 `read_lines`）：后台任务结束时要把这条循环从提示符上
    唤醒，而 `typer.prompt` 是阻塞的，主线程自己读就没法被叫醒。事件队列把"用户敲了一行"
    和"有个后台任务完成了"放在同一条路上。

    抽成函数是为了能测——`main_command` 里写死了 `AnthropicProvider()`，塞不进假的；这里收
    一个 provider 进来，测试就能拿剧本化的假 provider 跑两轮，看第二轮有没有带上第一轮。
    """
    state = ChatState(
        permission_mode=permission_mode, messages=messages, allow_fork=allow_fork
    )
    # 模型的窗口一次运行里定死，所以不进 ChatState 那个"会变的东西"的盒子（见它的类注释）。
    window = window_for(model)
    # 哪些工具的结果允许被折——一次运行里不变（见 tools/registry.py 的 foldable_names）。
    foldable = registry.foldable_names()
    # 其中哪些"重跑拿不回来"、折之前得先落一份（见 fold_persist_names）。
    fold_persist = registry.fold_persist_names()
    persist = _persister(cwd, session)

    def size_now() -> int:
        """这段历史现在有多大（token）——有锚点就用锚点，没有就按字符粗估。"""
        return context_size(
            state.messages, tokens=state.context_tokens, covered=state.context_covered
        )

    def compact_once(*, force: bool = False) -> bool:
        """压一次（够不够大由 force 决定），返回"真压了没有"。"""
        return state.replaced(
            _compact(
                state.messages, provider, session, size=size_now(), window=window, force=force
            )
        )

    def compact_now() -> str | None:
        """就地压一次，不管现在多长（`/compact` 用）。返回该跟用户补的一句话。

        判"有没有可压的"用的是同一个 cut_point，所以这儿的规矩和自动那条路一模一样：切点
        必须落在真正的用户轮上，最近几轮一律留着——区别只在于不看大小。
        """
        if cut_point(state.messages, size=size_now(), window=window, force=True) is None:
            return f"没什么可压的——最近 {KEEP_RECENT_TURNS} 轮一律原样留着，要压得先多聊几轮。"
        # 压在第几条、成没成，_compact 自己会说；只有它没说到的那些才由这里补。
        compact_once(force=True)
        return None

    def fold_once() -> None:
        """折一次——**缓存真冷了才折**，判据在 folding 里。

        不花钱。把内存里这份历史里旧的工具结果换成"开头一段 + 一句说明"，会话文件里原文一个字
        都不动；其中重跑拿不回来的那几类（bash / web_fetch / web_search）折之前先落一份盘，
        上下文里只留路径和 id。为什么没有手动入口，见 folding.py 的模块说明。
        """
        if not state.replaced(
            fold_if_stale(
                state.messages,
                foldable=foldable,
                path=session.path,
                persist=persist,
                persist_tools=fold_persist,
            )
        ):
            return
        console.print(
            "（把旧的工具结果折起来了；拿不回来的那几类另存了一份）",
            style="dim",
            markup=False,
        )

    def absorb(result: AgentResult) -> None:
        """把一轮的结果接回循环：落盘、换历史、更新锚点。

        普通一轮和后台任务的汇报回合共用这一段。两处各写一遍的话，迟早有一处忘了落盘，
        而那种漏是静默的。
        """
        # 长度得在 messages 被换掉**之前**取。
        session.append(result.messages[len(state.messages) :])
        state.messages = result.messages
        if result.anchor is not None:
            # 这一轮带工具的最后一发有多大——下一轮判断要不要压就看它。
            state.remember_context(result.anchor.tokens, result.anchor.covered)

    def run_turn(line: str) -> None:
        """用户说了一句话：折、压、发、接回历史、打印。"""
        # 缓存大概已经冷了就先折一次：那时候整段前缀反正要按原价重算，改历史不额外损失
        # 任何东西，是纯赚。**热的时候折是亏的**（见 known-issues D5），所以判据只有
        # "冷不冷"、不看大小——那一条在 folding 里，这儿只管排顺序。
        #
        # 折排在压缩**前面**：折不要钱、压要一次模型调用，折完可能就不用到压那一步了。
        fold_once()
        compact_once()
        system = build_system_prompt(
            cwd, state.permission_mode, model, allow_fork=state.allow_fork
        )
        result = runagent(
            line,
            provider,
            registry,
            system=system,
            cwd=cwd,
            allow_shell=allow_shell,
            permission_mode=state.permission_mode,
            prompter=prompter,
            grants=grants,
            messages=state.messages,
            # 落盘的工具结果按**这次会话**分目录（`.agent/tool-results/<id>/`），
            # 这样 `--delete` 能连同这些文件一起删。session.id 在 start() 那一刻就定死
            # 了，跑到一半不会变。
            spill_dir=spill.session_dir(cwd, session.id),
            allow_fork=state.allow_fork,
            # 交互模式才有"下一轮"可以送后台结果；一次性模式那边不传。
            allow_background=True,
            notices=tasks.take_notices if tasks is not None else None,
        )
        absorb(result)
        if result.error:
            # 这一轮没能把请求发出去（网络、限流、配置……）。**会话不结束**——用户再发一次
            # 就行；上面那句已经把这一轮**已经发生过的**工具调用落盘了，不会白丢。
            typer.echo(f"这一轮没跑成：{result.error}", err=True)
            return
        if result.final is None:
            typer.echo(
                f"用满步数也没等到最终回复，期间执行了 {len(result.trace)} 次工具调用。",
                err=True,
            )
            return
        console.print(f"\n{result.final}", markup=False)

    def report_background() -> None:
        """后台任务完成了：让模型**自己开口**告诉用户。

        只打一行程序通知是不够的——用户要的是模型知道这件事、并且用自己的话说出来。
        所以这里跑一个短回合：同一份 system、同一张工具表、同一段历史，只追加一条 user
        消息，前缀照旧命中缓存。

        `max_steps=1` + `NullPrompter`：这一回合只许说话。用户可能正停在提示符上输入，
        不能弹权限问题；也不该借"汇报"之名开始新的工作。
        """
        if tasks is None:
            return
        notice = tasks.take_notices()
        if not notice:
            # 主 agent 正在跑的那一轮可能已经把这条并进自己的上下文了；take_notices 取走
            # 即标记，所以这里拿到空就说明已经说过了，不重复开口。
            return
        fold_once()
        system = build_system_prompt(
            cwd, state.permission_mode, model, allow_fork=state.allow_fork
        )
        result = runagent(
            f"{notice}\n\n"
            "用一两句话把这件事告诉用户：哪个任务完成了、结论是什么。"
            "不要调用工具，不要开始新的工作。",
            provider,
            registry,
            system=system,
            max_steps=1,
            cwd=cwd,
            allow_shell=allow_shell,
            permission_mode=state.permission_mode,
            prompter=NullPrompter(),
            grants=grants,
            messages=state.messages,
            spill_dir=spill.session_dir(cwd, session.id),
            allow_fork=state.allow_fork,
            allow_background=False,
        )
        absorb(result)
        if result.error:
            typer.echo(f"后台任务的汇报没发出去：{result.error}", err=True)
            return
        if result.final is not None:
            console.print(f"\n{result.final}", markup=False)

    ctx = CommandContext(
        console=console,
        state=state,
        window=window,
        cwd=cwd,
        allow_shell=allow_shell,
        session=session,
        grants=grants,
        compact=compact_now,
        tasks=tasks,
        broker=broker,
    )
    # 建一次就够：命令表在循环里不变。它抓着的 ctx / state 都是上面那两个活着的对象——
    # 命令改档位、看历史、清批准，动到的都是循环这一轮真要用的那份。
    commands = default_commands()

    # 三个生产者、一个消费者：用户敲的一行、后台任务完成的事件、以及"有请求在等授权"的提醒。
    events: queue.Queue[tuple[str, object]] = queue.Queue()

    # 终端所有权：`wants_line` 是主循环在等输入，`using_terminal` 是读键盘线程正占用终端
    # （问权限或读一行）。只有读键盘线程会在 `using_terminal=True` 时碰 stdin——这是那条
    # "任何时刻只有一个读者"的不变式。
    gate = threading.Condition()
    wants_line = False
    using_terminal = False

    def service_approvals() -> None:
        """把后台任务排队等着的授权请求问掉。**只在读键盘线程里调。**

        这里用的 `prompter` 是主线程那一份真的 `TerminalPrompter`：预览格式、y/n/a、用户
        写下的拒绝理由，和前台权限提示完全一致。
        """
        if broker is None:
            return
        while (item := broker.take()) is not None:
            if item.cancel is not None and item.cancel.is_set():
                broker.resolve(item, Answer(False, note="任务已取消。"))
                continue
            console.print(f"（{item.label}后台任务请求：）", style="dim", markup=False)
            try:
                # remember=False：拿到答案之后再决定记不记——问的过程中任务可能被取消了。
                answer = ask_permission(prompter, item.request, grants, remember=False)
            except (EOFError, KeyboardInterrupt, typer.Abort):
                broker.resolve(item, Answer(False, note="没有回答，按拒绝处理。"))
                raise
            except Exception as exc:  # noqa: BLE001 - 一条处理坏了不能把读键盘线程带走
                console.print(
                    f"（处理后台授权请求失败：{type(exc).__name__}: {exc}；按拒绝处理）",
                    style="red",
                    markup=False,
                )
                broker.resolve(item, Answer(False, note=f"授权请求没处理成：{exc}"))
                continue
            if (
                answer.allowed
                and answer.remember
                and (item.cancel is None or not item.cancel.is_set())
            ):
                grants.remember(item.request)
            broker.resolve(item, answer)

    def read_lines() -> None:
        nonlocal wants_line, using_terminal
        while True:
            with gate:
                while not wants_line or using_terminal:
                    gate.wait()
                wants_line = False
                using_terminal = True
            try:
                service_approvals()  # 等在提示符后面的请求先处理掉
                line = typer.prompt("> ")
                service_approvals()  # 用户刚敲完一行：先处理请求，再把这一行交出去
            except (EOFError, KeyboardInterrupt, typer.Abort):
                events.put(("stop", None))
                return
            finally:
                with gate:
                    using_terminal = False
            events.put(("line", line))

    # 先把"任务结束了往哪儿报"接上，再开读键盘线程：顺序反过来的话，一个刚完成的任务可能
    # 落在两者中间，事件就丢了。
    if tasks is not None:
        tasks.set_on_finish(lambda record: events.put(("task", record.id)))
    if broker is not None:
        # 请求进队时如果读键盘线程正闲着，主循环会被叫醒打一句提示；用户停在 `> ` 上时
        # 只能等到下一次回车——这是行输入终端的限制。
        broker.on_defer = lambda: events.put(("approval", None))
    # daemon：进程退出时它跟着走。Ctrl+C 只会打在主线程上（主线程正阻塞在 events.get()），
    # 所以这条线程不需要自己处理中断。
    threading.Thread(target=read_lines, name="repl-input", daemon=True).start()

    try:
        while True:
            try:
                kind, payload = events.get_nowait()
            except queue.Empty:
                # 只有读键盘线程没占着终端时才让它去要输入；否则屏幕上会多一个 `> `。
                with gate:
                    if not using_terminal:
                        wants_line = True
                        gate.notify()
                kind, payload = events.get()

            if kind == "stop":
                break
            if kind == "task":
                report_background()
                continue
            if kind == "approval":
                console.print(
                    "（后台任务在等授权；按回车后处理）", style="dim", markup=False
                )
                continue

            line = str(payload).strip()
            if not line:
                continue
            outcome = commands.dispatch(line, ctx)
            if outcome == "stop":
                break
            if outcome == "handled":
                continue
            run_turn(line)
    except (typer.Abort, EOFError, KeyboardInterrupt):
        # Ctrl+C / Ctrl+D 都当正常退出。打断时提示符后面没有换行，先补一个，
        # 免得 bye 紧跟在 "^C" 后面。
        typer.echo()
    finally:
        if tasks is not None:
            # 循环没了：让还在跑的任务在下一步之前停下，再往队列里放事件只会积在那里。
            tasks.set_on_finish(None)
            tasks.cancel_all()
        if broker is not None:
            broker.on_defer = None
            broker.close("用户离开了，后台任务不能再请求授权。")


def _delete_session(cwd: Path, session_id: str, *, assume_yes: bool) -> None:
    """删掉一次会话：**先把要删的那次摆出来，问过才动手**。

    顺序是有意的。会话文件是那次对话唯一的记录——没有回收站，也不像改文件那样有
    `.agent/.history` 兜着（`Session.append` 是纯追加，从来不备份）。所以这一步要让人
    **看清删的是哪一次**再答：id 是一串数字加后缀，光看它认不出那是哪一回，最后那句人话才是
    认得出的那个钩子（和 `--list` 里那栏同一个用途）。

    `--yes` 跳过那一次问答，给脚本用。
    """
    try:
        target = describe(cwd, session_id)
    except FileNotFoundError as exc:
        raise typer.BadParameter(str(exc), param_hint="--delete") from None

    what = f"{target.messages} 条消息" if target.messages else "没说过话"
    spilled = spill.count_session(cwd, target.id)
    if spilled:
        # **要一起删的也得说出来**。只报"删掉某次会话"的话，用户不知道还端掉了一批落盘文件
        # ——那里面可能正是他想留着的东西（某条命令的完整输出）。
        what += f"，连同 {spilled} 个落盘文件"
    console.print(f"要删的是 {target.id}（{what}）", style="bold", markup=False)
    if target.preview:
        console.print(f"  最后一句：「{target.preview}」", markup=False)
    if not assume_yes and not typer.confirm(f"删掉 {target.id}？删了拿不回来"):
        console.print("没删。", markup=False)
        return
    delete(cwd, target.id)
    # 落盘的工具结果属于这次会话，跟着一起端掉（见 tools/spill.py）。
    spill.remove_session(cwd, target.id)
    console.print(f"删掉了 {target.id}。", markup=False)


@app.command()
def main_command(
    prompt: str = typer.Argument("", help="Prompt to send to the agent."),
    cwd: Path | None = typer.Option(None, "--cwd", "-C", help="agent 的工作目录，默认当前目录"),
    model: str = typer.Option(
        "",
        "--model",
        "-m",
        help="模型名；不写就用 providers/anthropic.py 里的 DEFAULT_MODEL",
    ),
    api_key: str = typer.Option(
        "",
        "--api-key",
        help="接口密钥；优先用 .env 里的 DEEPSEEK_API_KEY（命令行会留在 shell 历史里）",
    ),
    base_url: str = typer.Option(
        "",
        "--base-url",
        help="Anthropic 兼容端点；优先用 .env 里的 DEEPSEEK_ANTHROPIC_BASE_URL",
    ),
    shell: bool = typer.Option(
        False, "--shell", help="允许 agent 执行命令（跑在一次性 docker 容器里）"
    ),
    fork: bool = typer.Option(
        False,
        "--fork",
        help="允许主 agent 派 fork 类型的子 agent：继承当前对话和缓存前缀",
    ),
    permission_mode: str = typer.Option(
        "default",
        "--permission_mode",
        help=f"权限档位：{' / '.join(PERMISSION_MODES)}（accept / bypass 是短写法）",
    ),
    continue_: bool = typer.Option(False, "--continue", help="接上最近一次会话"),
    resume: str = typer.Option(
        "", "--resume", help="接上指定 id 的会话（id 就是 .agent/sessions/ 里的文件名）"
    ),
    list_: bool = typer.Option(
        False, "--list", help="列出这个目录下有过哪些会话（连同各自的 id），然后退出"
    ),
    delete_: str = typer.Option(
        "", "--delete", help="删掉指定 id 的会话（删之前会摆出来问一次），然后退出"
    ),
    yes: bool = typer.Option(False, "--yes", "-y", help="配合 --delete：不再问那一次"),
) -> None:
    """把指令交给 agent，把最终回复打印到标准输出。"""
    prompt = prompt.strip()
    if list_:
        # 只看不做的活儿，排在所有检查和构造之前：它连 provider 都不需要，所以没有 API key
        # 也跑得起来，一个打错的档位也不该把一次列目录拦下。
        render_session_list(console, cwd or Path.cwd())
        raise typer.Exit()
    if delete_:
        # 删除同理，也排在构造之前：它不是 agent 的动作，是用户对自己的文件动手——不该需要
        # provider，也不该因为顺带写错了个档位就被拦下。
        _delete_session(cwd or Path.cwd(), delete_, assume_yes=yes)
        raise typer.Exit()
    try:
        permission_mode = parse_permission_mode(permission_mode)
    except ValueError as exc:
        # 在门口就把错字拦住。跑起来才发现档位不对的话，那时已经有工具调用被放行了。
        raise typer.BadParameter(str(exc), param_hint="--permission_mode") from None

    if permission_mode == "bypass permissions":
        # 唯一一档把用户从回路里拿掉。开的时候说一声，别让它悄悄生效——它不是常驻设置，
        # 是这一次运行的选择。
        console.print(
            "⚠ bypass permissions：什么都不问。模型跑命令不再经过你，"
            "而命令能改项目里任何文件、也能把 .env 读进上下文。",
            style="bold red",
            markup=False,
        )

    # prompter 建一次、两个分支共用：它的答案以后要带"这一次会话别再问了"这种记忆，
    # 每轮新建一个的话那份记忆就丢了。
    prompter = TerminalPrompter()
    # "本会话这个别再问了"记在这儿。和 prompter 一样得建在循环外面——runagent 每轮新建，
    # 放里面的话下一轮就忘了，等于每轮都重新问一遍。
    grants = Grants()
    # ask_user 只在"有人可问"的时候才存在，所以它不在 default_tools() 里，在这儿装进来。
    registry = default_tools()
    registry.register(make_ask_user(prompter))
    workdir = cwd or Path.cwd()
    # 让磁盘上那两份索引（记忆、技能）跟目录对齐：手工加/删过的话它们会漂（不影响模型——
    # 模型看的是现算的那份）。只在真不一样时才写，目录不存在就直接返回。
    sync_index(workdir)
    sync_skill_index(workdir)
    # 会话：接着上一次，或者开一个新的。两个分支共用同一份落盘。**放在 provider 前面**——
    # 接不上的时候该报的是"没有这个会话"，而不是别的什么。
    history: list[dict] = []
    if resume or continue_:
        try:
            session, meta, history = load(workdir, resume or None)
        except FileNotFoundError as exc:
            raise typer.BadParameter(str(exc)) from None
        # 档位**不跟着历史走**：它是"我这次要多大权限"的选择，不该被上次带着跑。上次是什么
        # 档打在这儿，对不上你能看见。
        console.print(
            f"接上 {session.id}：{len(history)} 条消息，上次档位 {meta.get('permission_mode')}"
            f"（这次按 {permission_mode} 走）",
            style="bold",
            markup=False,
        )
    else:
        session = start(workdir, permission_mode=permission_mode, allow_shell=shell)
    # provider 建一次就够（一次运行一个 HTTP 客户端）。system prompt 不在这儿建：一次性那条
    # 分支现拼一份，交互那条每轮重拼——记忆会中途变，见 _chat。
    provider = AnthropicProvider(
        model=model or DEFAULT_MODEL,
        api_key=api_key or None,
        base_url=base_url or None,
    )
    # 后台任务表：`agent` 工具把跑长的活交给它，`/tasks` 和空闲时唤醒的汇报回合共用同一份。
    tasks = TaskManager()
    # 后台任务的授权请求表：后台线程往里放，读键盘线程在持有终端时取出来问用户。**要在
    # `make_agent_tool` 之前建**——那个工具的闭包会抓着它。
    broker = PermissionBroker()
    # 子 agent 那个工具在这儿装进去（`agent`），和上面的 `ask_user` 一样是 harness 才有的能力。
    # **它排在 provider 后面，不是跟 ask_user 放一起**：那个工具要 provider 才能跑（子 agent
    # 是要真发请求的），而 `registry` 建在 provider **之前**——那个顺序是有意的，见上面
    # "接会话排在 provider 前面"那句。两条路（一次性、_chat）共用同一个 registry，所以装在
    # 这儿两边都用得上。
    registry.register(
        make_agent_tool(
            provider=provider,
            prompter=prompter,
            grants=grants,
            model=provider.model,
            tasks=tasks,
            broker=broker,
        )
    )
    # fork 不单独注册：它是 `agent` 的 `type` 枚举里的一项。开没开 `--fork` 由
    # `ToolContext.fork` 判——关着的时候 `agent(type="fork")` 会回一句"没打开"。枚举是静态的，
    # 所以切换开关**不会改工具表**；变的只有系统提示里那段说明。
    if prompt:
        # 一次性模式下用户那句不显示出来的话，输出里只有工具和回答，看不出在答什么。
        console.print(f"> {prompt}", style="bold", markup=False)
        # 接上来的历史可能已经很长了，先压一次再发。**这一轮之前没有真请求**，所以没有锚点，
        # 只能整段按字符粗估——`--resume` 接上一个长历史时走的就是这条。
        #
        # 折在压前面，判据和 _chat 里那条一样（会话文件久没被写过 = 缓存大概冷了）。
        history = fold_if_stale(
            history,
            foldable=registry.foldable_names(),
            path=session.path,
            persist=_persister(workdir, session),
            persist_tools=registry.fold_persist_names(),
        )
        history = _compact(
            history,
            provider,
            session,
            size=context_size(history, tokens=None),
            window=window_for(provider.model),
        )
        result = runagent(
            prompt,
            provider,
            registry,
            system=build_system_prompt(
                workdir, permission_mode, provider.model, allow_fork=fork
            ),
            cwd=workdir,
            allow_shell=shell,
            permission_mode=permission_mode,
            prompter=prompter,
            grants=grants,
            messages=history,
            spill_dir=spill.session_dir(workdir, session.id),
            allow_fork=fork,
        )
        # 长度得在 history 被换掉之前取——和 _chat 里同一处道理。
        session.append(result.messages[len(history) :])
        if result.error:
            # 一次性模式没有"下一轮"，所以报错就退出——但退出码要是 1，别让脚本以为跑成了。
            typer.echo(f"这一轮没跑成：{result.error}", err=True)
            raise typer.Exit(code=1)
        if result.final is None:
            typer.echo(
                f"用满步数也没等到最终回复，期间执行了 {len(result.trace)} 次工具调用。",
                err=True,
            )
            raise typer.Exit(code=1)
        console.print(f"\n{result.final}", markup=False)
        return

    _chat(
        provider,
        registry,
        session,
        history,
        cwd=workdir,
        allow_shell=shell,
        permission_mode=permission_mode,
        prompter=prompter,
        grants=grants,
        # 提示词里要写"你是哪个模型"。**从这儿传进去，不是去 provider 里取**——Provider 协议
        # 只声明了 complete，认得它不等于认得 model；手上这份是真的 AnthropicProvider。
        model=provider.model,
        allow_fork=fork,
        tasks=tasks,
        broker=broker,
    )
    print("bye")


if __name__ == "__main__":
    app()
