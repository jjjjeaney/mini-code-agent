"""读长期记忆的工具：按名字取一条记忆的正文。

清单（名字 + 一行描述）在 system prompt 里，模型看着它挑；正文可能不短，所以按需取。

**不走 read_file**：记忆在 `.agent/` 下面，那是敏感目录，文件工具读不到——那是刻意的
（`.agent` 里还有会话流水和改前原件）。这个工具是 harness 自己开的一道门，只认 `<名字>.md`
这种形状，也就顺带挡掉了"顺着路径跑到别处去"。

只读不写。写入那一侧要经用户批准（见 `docs/memory-design.md` 第四节），是另一件事。
"""

from __future__ import annotations

from typing import Any

from mini_code_agent.memory import read
from mini_code_agent.tools.base import ToolContext

READ_MEMORY_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "name": {"type": "string", "description": "记忆的名字，就是清单里每条开头那个词"},
    },
    "required": ["name"],
}


def read_memory(kwargs: dict[str, Any], ctx: ToolContext) -> str:
    """按名字取一条记忆的正文。"""
    return read(ctx.cwd, kwargs["name"]).body
