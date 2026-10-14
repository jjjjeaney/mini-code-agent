"""命令行入口。"""

from __future__ import annotations

from pathlib import Path

import typer

from mini_code_agent.commands import (
    ChatState,
    CommandContext,
    default_commands,
    render_session_list,
)
from mini_code_agent.console import console
from mini_code_agent.core.agent import runagent
from mini_code_agent.memory import sync_index
from mini_code_agent.permissions import PERMISSION_MODES, Grants, parse_permission_mode
from mini_code_agent.prompting import TerminalPrompter, make_ask_user
from mini_code_agent.prompts import Prompter
from mini_code_agent.providers.anthropic import AnthropicProvider
from mini_code_agent.providers.base import Provider
from mini_code_agent.sessions import Session, delete, describe, load, start, summary_message
from mini_code_agent.skills import sync_index as sync_skill_index
from mini_code_agent.summarizing import KEEP_RECENT_TURNS, cut_point, summarize
from mini_code_agent.system_prompt import build_system_prompt
from mini_code_agent.tools import ToolRegistry, default_tools, spill

app = typer.Typer(add_completion=False, help="最小可用的 coding agent。")


def _compact(
    messages: list[dict], provider: Provider, session: Session, *, force: bool = False
) -> list[dict]:
    """压一次：把旧的那段换成一段摘要，并记进会话。

    `force=False`（默认，"轮与轮之间"那条路）是**太长才压**；`force=True` 是 `/compact`
    ——用户点名要压，就不看长度了。两条走同一个函数，免得落盘、记账那些事各自漂出去。

    放在**轮与轮之间**做，不放 runagent 里：那边有个前提——"返回的 messages = 传进去的 +
    追加的那几条"（下面就是靠 result.messages[len(messages):] 落盘的）。压缩会改掉前缀，
    正好把这个前提破坏掉。

    摘要调不动就当没压，照常跑这一轮——绝不能因为压缩把会话卡死。
    """
    cut = cut_point(messages, force=force)
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
) -> None:
    """交互循环：读一行、跑一轮、把消息接下去并落盘。

    **每轮重拼一次 system prompt**：记忆和 `AGENTS.md` 都可能在中途变（模型刚 `remember` 了
    一条），拼一次用到底的话，这一轮记下的东西要重启进程才看得见。档位和历史同理——前者
    `/permission_mode` 能换，后者 `/status` 要报它多长，所以这两个从局部量挪进了 `state`，
    和提示词一起每轮现读。传进来的 `permission_mode` 和 `messages` 只是这次运行的**起始值**。

    这一行先过命令表（见 commands/）：认出来的命令就地办掉、**不发给模型**，也不落盘——
    命令是用户对程序说的话，不是对话的一部分，接上历史时不该看见它。

    抽成函数是为了能测——`main_command` 里写死了 `AnthropicProvider()`，塞不进假的；这里收
    一个 provider 进来，测试就能拿剧本化的假 provider 跑两轮，看第二轮有没有带上第一轮。
    """
    state = ChatState(permission_mode=permission_mode, messages=messages)

    def compact_now() -> str | None:
        """就地压一次，不管现在多长（`/compact` 用）。返回该跟用户补的一句话。

        判"有没有可压的"用的是同一个 cut_point，所以这儿的规矩和自动那条路一模一样：切点
        必须落在真正的用户轮上，最近几轮一律留着——区别只在于不看长度。
        """
        if cut_point(state.messages, force=True) is None:
            return f"没什么可压的——最近 {KEEP_RECENT_TURNS} 轮一律原样留着，要压得先多聊几轮。"
        # 切在第几条、成没成，_compact 自己会说；只有它没说到的那些才由这里补。
        state.messages = _compact(state.messages, provider, session, force=True)
        return None

    ctx = CommandContext(
        console=console,
        state=state,
        cwd=cwd,
        allow_shell=allow_shell,
        session=session,
        grants=grants,
        compact=compact_now,
    )
    # 建一次就够：命令表在循环里不变。它抓着的 ctx / state 都是上面那两个活着的对象——
    # 命令改档位、看历史、清批准，动到的都是循环这一轮真要用的那份。
    commands = default_commands()
    try:
        while True:
            line = typer.prompt("> ").strip()
            if not line:
                continue
            outcome = commands.dispatch(line, ctx)
            if outcome == "stop":
                break
            if outcome == "handled":
                continue
            # 太长就压一次再发。放在轮与轮之间，不放进 runagent——理由见 _compact。
            state.messages = _compact(state.messages, provider, session)
            system = build_system_prompt(cwd, state.permission_mode, model)
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
            )
            # 落盘、接下历史——两边要的是同一批东西：这一轮新增的那几条。
            # 长度得在 messages 被换掉**之前**取。
            session.append(result.messages[len(state.messages) :])
            state.messages = result.messages
            if result.error:
                # 这一轮没能把请求发出去（网络、限流、配置……）。**会话不结束**——用户再发一次
                # 就行；上面那句已经把这一轮**已经发生过的**工具调用落盘了，不会白丢。
                typer.echo(f"这一轮没跑成：{result.error}", err=True)
                continue
            if result.final is None:
                typer.echo(
                    f"用满步数也没等到最终回复，期间执行了 {len(result.trace)} 次工具调用。",
                    err=True,
                )
                continue
            console.print(f"\n{result.final}", markup=False)
    except (typer.Abort, EOFError, KeyboardInterrupt):
        # Ctrl+C / Ctrl+D 都当正常退出。打断时提示符后面没有换行，先补一个，
        # 免得 bye 紧跟在 "^C" 后面。
        typer.echo()


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
    shell: bool = typer.Option(
        False, "--shell", help="允许 agent 执行命令（跑在一次性 docker 容器里）"
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
    provider = AnthropicProvider()
    if prompt:
        # 一次性模式下用户那句不显示出来的话，输出里只有工具和回答，看不出在答什么。
        console.print(f"> {prompt}", style="bold", markup=False)
        # 接上来的历史可能已经很长了，先压一次再发。
        history = _compact(history, provider, session)
        result = runagent(
            prompt,
            provider,
            registry,
            system=build_system_prompt(workdir, permission_mode, provider.model),
            cwd=workdir,
            allow_shell=shell,
            permission_mode=permission_mode,
            prompter=prompter,
            grants=grants,
            messages=history,
            spill_dir=spill.session_dir(workdir, session.id),
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
    )
    print("bye")


if __name__ == "__main__":
    app()
