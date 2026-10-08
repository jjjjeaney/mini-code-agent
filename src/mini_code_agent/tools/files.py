"""文件类工具。

路径一律先过 safety 的围栏：解析、判越界、再决定要不要碰文件系统。
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from mini_code_agent.tools.base import ToolContext
from mini_code_agent.tools.safety import (
    OutOfSandboxError,
    is_within,
    resolve_within,
    should_skip,
    truncate_text,
)

LIST_FILES_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "path": {
            "type": "string",
            "description": "要列哪个目录，相对项目根目录。留空就是项目根目录本身。",
        },
    },
    "required": [],
}


def list_files(kwargs: dict[str, Any], ctx: ToolContext) -> str:
    """列出目录下的所有文件和目录，含子目录，渲染成一棵树。

    目录带结尾斜杠，一眼能看出哪些是目录。同一个目录里先排文件、再排目录，
    两组各自按名字排序，这样结构稳定、每次跑出来都一样。

    cwd 从 ctx 拿，不从参数拿——沙箱边界必须由外部指定，
    要是让模型自己传，围栏就成了摆设。
    """
    root = ctx.cwd.resolve()
    raw = kwargs.get("path") or "."

    target = resolve_within(root, raw)
    if not is_within(root, target):
        raise OutOfSandboxError(
            f"路径 {raw!r} 指向 {target}，超出了允许的范围 {root}。"
            f"只能列 {root} 里面的目录。"
        )
    if not target.exists():
        raise FileNotFoundError(f"{raw!r} 不存在。用 '.' 可以列项目根目录。")
    if not target.is_dir():
        raise NotADirectoryError(
            f"{raw!r} 是文件不是目录。list_files 只能列目录，看单个文件用 read_file。"
        )

    lines = _tree_lines(target, prefix="")
    if not lines:
        return f"{raw!r} 下面没有可列出的内容：要么是空目录，要么内容都被过滤掉了。"
    return truncate_text("\n".join(lines))


def _tree_lines(directory: Path, prefix: str) -> list[str]:
    """把 directory 画成一棵树，返回每一行。

    prefix 是这一层所有条目前面要垫的竖线：祖先里凡是"不是最后一个孩子"的，
    都得在这儿留一道 `│`，否则层级就对不上了。
    """
    with os.scandir(directory) as scanner:
        # scandir 的条目在迭代器关掉之后不保证还能用，所以在这里就把要的信息取干净。
        entries = [
            (entry.name, entry.is_dir(), entry.is_symlink())
            for entry in scanner
            if not should_skip(Path(entry.path))
        ]
    # 文件在前、目录在后，两组各自按名字排（忽略大小写）。这样结构稳定，
    # 每次跑出来顺序都一样。
    entries.sort(key=lambda entry: (entry[1], entry[0].lower()))

    lines: list[str] = []
    for index, (name, is_dir, is_symlink) in enumerate(entries):
        is_last = index == len(entries) - 1
        lines.append(f"{prefix}{'└── ' if is_last else '├── '}{name}{'/' if is_dir else ''}")

        # 不跟进符号链接：链成环会走不完。它会照常显示成目录，只是不往里展开。
        if is_dir and not is_symlink:
            lines.extend(_tree_lines(directory / name, prefix + ("    " if is_last else "│   ")))
    return lines
