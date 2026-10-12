"""命令行入口。"""

from __future__ import annotations

from pathlib import Path

import typer

from mini_code_agent.console import console
from mini_code_agent.core.agent import runagent
from mini_code_agent.permissions import PERMISSION_MODES, Grants, parse_permission_mode
from mini_code_agent.prompting import TerminalPrompter, make_ask_user
from mini_code_agent.prompts import Prompter
from mini_code_agent.providers.anthropic import AnthropicProvider
from mini_code_agent.providers.base import Provider
from mini_code_agent.sessions import Session, load, start
from mini_code_agent.system_prompt import build_system_prompt
from mini_code_agent.tools import ToolRegistry, default_tools

app = typer.Typer(add_completion=False, help="最小可用的 coding agent。")


def _chat(
    provider: Provider,
    registry: ToolRegistry,
    system: str,
    session: Session,
    messages: list[dict],
    *,
    cwd: Path,
    allow_shell: bool,
    permission_mode: str,
    prompter: Prompter,
    grants: Grants,
) -> None:
    """交互循环：读一行、跑一轮、把消息接下去并落盘。

    抽成函数是为了能测——`main_command` 里写死了 `AnthropicProvider()`，塞不进假的；这里收
    一个 provider 进来，测试就能拿剧本化的假 provider 跑两轮，看第二轮有没有带上第一轮。
    """
    try:
        while True:
            line = typer.prompt("> ").strip()
            if not line:
                continue
            if line == "/exit":
                break
            result = runagent(
                line,
                provider,
                registry,
                system=system,
                cwd=cwd,
                allow_shell=allow_shell,
                permission_mode=permission_mode,
                prompter=prompter,
                grants=grants,
                messages=messages,
            )
            # 落盘、接下历史——两边要的是同一批东西：这一轮新增的那几条。
            # 长度得在 messages 被换掉**之前**取。
            session.append(result.messages[len(messages) :])
            messages = result.messages
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
) -> None:
    """把指令交给 agent，把最终回复打印到标准输出。"""
    prompt = prompt.strip()
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
    # provider 和 system prompt 也各建一次：一次运行一个 HTTP 客户端、一份提示词。
    # 档位中途换不了，所以 system 不用每轮重拼。
    workdir = cwd or Path.cwd()
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
    # provider 和 system prompt 各建一次：一次运行一个 HTTP 客户端、一份提示词。
    # 档位中途换不了，所以 system 不用每轮重拼。
    provider = AnthropicProvider()
    system = build_system_prompt(workdir, permission_mode)
    if prompt:
        # 一次性模式下用户那句不显示出来的话，输出里只有工具和回答，看不出在答什么。
        console.print(f"> {prompt}", style="bold", markup=False)
        result = runagent(
            prompt,
            provider,
            registry,
            system=system,
            cwd=workdir,
            allow_shell=shell,
            permission_mode=permission_mode,
            prompter=prompter,
            grants=grants,
            messages=history,
        )
        # 长度得在 history 被换掉之前取——和 _chat 里同一处道理。
        session.append(result.messages[len(history) :])
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
        system,
        session,
        history,
        cwd=workdir,
        allow_shell=shell,
        permission_mode=permission_mode,
        prompter=prompter,
        grants=grants,
    )
    print("bye")


if __name__ == "__main__":
    app()
