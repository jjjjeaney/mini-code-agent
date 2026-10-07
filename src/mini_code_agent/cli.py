"""命令行入口。"""

from __future__ import annotations

import typer

from mini_code_agent.core.agent import runagent
from mini_code_agent.providers.anthropic import AnthropicProvider

app = typer.Typer(add_completion=False, help="最小可用的 coding agent。")


@app.command()
def main_command(prompt: str = typer.Argument(..., help="要交给 agent 的指令")) -> None:
    """把指令交给 agent，把最终回复打印到标准输出。"""
    result = runagent(prompt, AnthropicProvider())

    if result.final is None:
        typer.echo(
            f"用满步数也没等到最终回复，期间执行了 {len(result.trace)} 次工具调用。",
            err=True,
        )
        raise typer.Exit(code=1)

    typer.echo(result.final)


if __name__ == "__main__":
    app()
