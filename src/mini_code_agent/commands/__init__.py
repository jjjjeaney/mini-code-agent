"""斜杠命令：CLI 的表现层。

只调用 agent 的公开接口，不反向依赖 core 的内部结构。

交互循环里读到的那一行，先过 `Commands.dispatch`：认出来的命令**不发给模型**，由程序
就地办掉（`/help`、`/permission_mode`、`/fork`、`/status`、`/sessions`、`/tasks`、`/compact`、
`/grants`、`/exit` 这些）。加新的就在 builtin.py 里写一条，再登记进 `default_commands()`。
"""

from mini_code_agent.commands.base import ChatState, Command, CommandContext, Outcome
from mini_code_agent.commands.builtin import render_session_list
from mini_code_agent.commands.registry import Commands, default_commands

__all__ = [
    "ChatState",
    "Command",
    "CommandContext",
    "Commands",
    "Outcome",
    "default_commands",
    "render_session_list",
]
