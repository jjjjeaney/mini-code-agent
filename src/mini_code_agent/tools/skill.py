"""skill 的两个工具：读一份的正文，和读它的附件。

清单（名字 + 一句话）在 system prompt 里，模型看着它挑；正文和附件可能都不短，所以按需取。

**都不走 `read_file`**：skill 在 `.agent/` 底下，那是敏感目录，文件工具够不着——那是刻意的
（`.agent` 里还有会话流水和改前备份）。这两个工具是 harness 自己开的一道门，只认
"名字 + 附件相对路径"这种形状，顺带把路径穿越挡掉。

**只有读，没有写**：skill 由人写（见 `skills/store.py`）。记忆那边至少是"事实"，skill 是
"步骤"，不该让模型自己给自己立规矩。
"""

from __future__ import annotations

from typing import Any

from mini_code_agent.skills import attachments, read
from mini_code_agent.tools.base import ToolContext
from mini_code_agent.tools.safety import (
    MAX_READ_FILE_CHARS,
    ensure_within_size,
    find_symlink,
    resolve_inside,
    truncate_text,
)

READ_SKILL_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "name": {"type": "string", "description": "skill 的名字，就是清单里每条开头那个"},
    },
    "required": ["name"],
}

READ_SKILL_FILE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "name": {"type": "string", "description": "skill 的名字"},
        "path": {
            "type": "string",
            "description": "附件在那个 skill 目录里的相对路径，比如 check.sh",
        },
    },
    "required": ["name", "path"],
}


def read_skill(kwargs: dict[str, Any], ctx: ToolContext) -> str:
    """取一份 skill 的正文——也就是要照着做的那些步骤——末尾附上它有哪些附件。

    附件那段**接在截断之后**：它是元信息，被截掉的话模型就不知道有附件了（跟 read_file
    里"共 N 行"那半句同一个道理）。

    限额按**读文件**那一档走，不按普通工具结果那档：正文是"照着做"的步骤，按工具那档
    截成一段预览，等于让它照着半个步骤做事。
    """
    skill = read(ctx.cwd, kwargs["name"])
    parts = [truncate_text(skill.body, MAX_READ_FILE_CHARS)]

    files, hidden = attachments(skill)
    if files:
        listed = "\n".join(f"  {name}" for name in files)
        more = f"（还有 {hidden} 个没列出来）" if hidden else ""
        parts.append(
            f"这份 skill 的目录里还有这些文件：\n{listed}{more}\n"
            f"要哪个就用 read_skill_file(name, path) 取。"
        )
    return "\n\n".join(parts)


def read_skill_file(kwargs: dict[str, Any], ctx: ToolContext) -> str:
    """取 skill 目录里的一个附件。

    **围栏是"这个 skill 自己的目录"。** 这是本项目里第三条读文件的路径（另外两条是文件工具和
    `read_memory`），所以闸门一道都不能省：

    - 名字先过校验——`../` 那种顺着名字就跑出去了。这一步在 `skills.read` 里。
    - 附件路径再过 `resolve_inside`：解析之后必须还在这个 skill 目录里（`..`、绝对路径、
      以及指到外面的符号链接，都在这一步被拦下）。
    - 路径上再有符号链接也直接拒。这是文件工具那边一贯的口径（宁可比"解析后还在圈里"
      更严一点），理由是链接指到哪儿得看它指向的文件，而不是看它的名字。
    - 太大、或者不是 UTF-8 文本，都说清楚是什么问题——附件里放个图片模板是正常的，
      这时候该告诉模型"读不了"，而不是甩一段解码错误。
    """
    skill = read(ctx.cwd, kwargs["name"])  # 顺带确认这份 skill 真的存在
    raw = str(kwargs["path"])

    target = resolve_inside(skill.directory, raw)
    link = find_symlink(raw, skill.directory)
    if link is not None:
        raise ValueError(
            f"{raw!r} 的路径上有符号链接（{link}）。不顺着链接读——那样读到的可能是另一个文件。"
        )
    if target.is_dir():
        raise IsADirectoryError(f"{raw!r} 是个目录。附件只能逐个文件读。")
    if not target.is_file():
        raise FileNotFoundError(
            f"{raw!r} 不存在。它在 {skill.name}/ 里，路径要相对那个目录写。"
        )
    ensure_within_size(target)

    try:
        # 同样按读文件那一档：附件是脚本、模板、参考文档，和 read_file 读的东西是一类。
        return truncate_text(target.read_text(encoding="utf-8"), MAX_READ_FILE_CHARS)
    except UnicodeDecodeError:
        raise ValueError(
            f"{raw!r} 不是 UTF-8 文本（可能是图片、压缩包之类）。这个工具只读文本附件。"
        ) from None
