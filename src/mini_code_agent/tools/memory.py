"""长期记忆的两个工具：读一条的正文，和记一条新的。

清单（名字 + 一行描述）在 system prompt 里，模型看着它挑；正文可能不短，所以按需取。

**都不走 read_file / write_file**：记忆在 `.agent/` 下面，那是敏感目录，文件工具够不着——那是
刻意的（`.agent` 里还有会话流水和改前原件）。这两个工具是 harness 自己开的一道门，只认
`<名字>.md` 这种形状，顺带挡掉了路径穿越。

写的那条还要过权限闸门（`permissions.py` 里"写记忆一律要问"）：记忆是**以后每次会话都生效**的
东西，和改 `AGENTS.md` 一个道理，不该悄悄写进去。
"""

from __future__ import annotations

from typing import Any

from mini_code_agent.memory import Memory, read, write
from mini_code_agent.memory.store import TYPES
from mini_code_agent.tools.base import ToolContext
from mini_code_agent.tools.safety import MAX_TOOL_CHARS, truncate_text

READ_MEMORY_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "name": {"type": "string", "description": "记忆的名字，就是清单里每条开头那个词"},
    },
    "required": ["name"],
}

REMEMBER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "name": {
            "type": "string",
            "description": (
                "记忆的名字：小写字母、数字、连字符，一眼看出是什么，比如 uv-for-tests"
            ),
        },
        "description": {
            "type": "string",
            "description": (
                "一句话说清这条是什么。它进系统提示的清单，下次只靠这一句判断要不要取正文"
            ),
        },
        "type": {
            "type": "string",
            "enum": list(TYPES),
            "description": (
                "user=关于用户的偏好习惯；project=这个项目的约定、命令、边界；"
                "reference=外部指针（文档地址、服务在哪儿）"
            ),
        },
        "body": {
            "type": "string",
            "description": (
                "正文。写能再查证的事实：具体路径、具体命令、为什么这么定，"
                "并且**点出依据**（哪份文件、哪条命令、什么时候的事）——以后好核实它还算不算数"
            ),
        },
    },
    "required": ["name", "description", "type", "body"],
}


def read_memory(kwargs: dict[str, Any], ctx: ToolContext) -> str:
    """按名字取一条记忆的正文。

    正文是模型自己写的，长度本来可控；这里卡一道只是不让它变成一条绕过所有限额的通道
    （写的时候不限长，读的时候也不限的话，那条通道就一直开着）。
    """
    return truncate_text(read(ctx.cwd, kwargs["name"]).body, MAX_TOOL_CHARS)


def remember(kwargs: dict[str, Any], ctx: ToolContext) -> str:
    """记一条长期记忆。

    `source` 由这里定死成 `agent`——记忆是谁写的这件事不该让写它的模型自己填，那是给人看的
    置信度标记。
    """
    memory = Memory(
        name=kwargs["name"],
        description=kwargs["description"],
        type=kwargs["type"],
        body=kwargs["body"],
        source="agent",
    )
    write(ctx.cwd, memory)
    return (
        f"记下了 {memory.name}（type={memory.type}）。"
        "下一轮起它就出现在清单里，同时被写进 .agent/memory/。"
    )
