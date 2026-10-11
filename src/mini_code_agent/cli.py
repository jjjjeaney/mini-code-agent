"""命令行入口。"""

from __future__ import annotations

from pathlib import Path

import typer

from mini_code_agent.core.agent import runagent
from mini_code_agent.permissions import parse_permission_mode
from mini_code_agent.prompting import TerminalPrompter
from mini_code_agent.providers.anthropic import AnthropicProvider

app = typer.Typer(add_completion=False, help="最小可用的 coding agent。")


@app.command()
def main_command(
    prompt: str = typer.Argument("", help="Prompt to send to the agent."),
    cwd: Path | None = typer.Option(None, "--cwd", "-C", help="agent 的工作目录，默认当前目录"),
    shell: bool = typer.Option(
        False, "--shell", help="允许 agent 执行命令（跑在一次性 docker 容器里）"
    ),
    permission_mode: str = typer.Option("default","--permission_mode", help=""),
) -> None:
    """把指令交给 agent，把最终回复打印到标准输出。"""
    prompt = prompt.strip()
    try:
        permission_mode = parse_permission_mode(permission_mode)
    except ValueError as exc:
        # 在门口就把错字拦住。跑起来才发现档位不对的话，那时已经有工具调用被放行了。
        raise typer.BadParameter(str(exc), param_hint="--permission_mode") from None

    # prompter 建一次、两个分支共用：它的答案以后要带"这一次会话别再问了"这种记忆，
    # 每轮新建一个的话那份记忆就丢了。
    prompter = TerminalPrompter()
    if prompt:
        result = runagent(
            prompt,
            AnthropicProvider(),
            cwd=cwd,
            allow_shell=shell,
            permission_mode=permission_mode,
            prompter=prompter,
        )
        if result.final is None:
            typer.echo(
                f"用满步数也没等到最终回复，期间执行了 {len(result.trace)} 次工具调用。",
                err=True,
            )
            raise typer.Exit(code=1)
        typer.echo(result.final)
        return
    else:
        try:
            while True:
                line = typer.prompt("> ").strip()
                if not line:
                    continue
                if line == "/exit":
                    break
                result = runagent(
                    line,
                    AnthropicProvider(),
                    cwd=cwd,
                    allow_shell=shell,
                    permission_mode=permission_mode,
                    prompter=prompter,
                )
                if result.final is None:
                    typer.echo(
                        f"用满步数也没等到最终回复，期间执行了 {len(result.trace)} 次工具调用。",
                        err=True,
                    )
                    continue
                typer.echo(result.final)
        except (typer.Abort, EOFError, KeyboardInterrupt):
            # Ctrl+C / Ctrl+D 都当正常退出。打断时提示符后面没有换行，先补一个，
            # 免得 bye 紧跟在 "^C" 后面。
            typer.echo()
        print("bye")


if __name__ == "__main__":
    app()
