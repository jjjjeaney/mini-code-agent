"""文件类工具。

路径一律先过 safety 的围栏：解析、判越界、再决定要不要碰文件系统。
"""

from __future__ import annotations

import os
from collections import deque
from dataclasses import dataclass, field
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

DEFAULT_MAX_DEPTH = 5
DEFAULT_MAX_NODES = 200

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

PROJECT_TREE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "max_depth": {
            "type": "integer",
            "description": f"最多往下展开几层，默认 {DEFAULT_MAX_DEPTH}",
        },
        "max_nodes": {
            "type": "integer",
            "description": f"最多列出多少条，默认 {DEFAULT_MAX_NODES}",
        },
    },
    "required": [],
}


def list_files(kwargs: dict[str, Any], ctx: ToolContext) -> str:
    """列出某个目录下面一层的文件和子目录，不往下钻。

    只看一层是有意的：它回答的是"这个目录里有什么"，想知道整体结构用 project_tree。
    目录带结尾斜杠，先排目录、再排文件，各自按名字排，每次跑出来都一样。

    cwd 从 ctx 拿，不从参数拿——沙箱边界必须由外部指定，
    要是让模型自己传，围栏就成了摆设。
    """
    root = ctx.cwd.resolve()
    raw = kwargs.get("path") or "."
    target = _checked_dir(root, raw)

    entries = _children(target)
    if not entries:
        return f"{raw!r} 下面没有可列出的内容：要么是空目录，要么内容都被过滤掉了。"
    return truncate_text("\n".join(_render_level(entries)))


def project_tree(kwargs: dict[str, Any], ctx: ToolContext) -> str:
    """把 cwd 整个项目画成一棵树，用来一眼看清整体结构。

    和 list_files 的区别是它会往下钻，所以拿 max_depth 和 max_nodes 两道闸控制规模：
    项目一大，无限制地列出来会把上下文塞满。撞上哪道闸都会在末尾说明。

    起点固定是 cwd，不接受路径参数——沙箱边界由外部指定，模型碰不到。
    """
    root = ctx.cwd.resolve()
    max_depth = max(1, int(kwargs.get("max_depth") or DEFAULT_MAX_DEPTH))
    max_nodes = max(1, int(kwargs.get("max_nodes") or DEFAULT_MAX_NODES))

    nodes = _build_tree(root, depth=1, max_depth=max_depth)
    if not nodes:
        return "项目里没有可列出的内容：要么是空项目，要么内容都被过滤掉了。"

    truncated = _keep_breadth_first(nodes, max_nodes)
    text = "\n".join(_render_tree(nodes))
    if truncated:
        text += f"\n\n[已截断：条目数到了上限 {max_nodes}，剩下的没列出来]"
    return text


@dataclass
class _Node:
    name: str
    is_dir: bool
    children: list[_Node] = field(default_factory=list)
    kept: bool = False


def _checked_dir(root: Path, raw: str | Path) -> Path:
    """确认 raw 指向 root 里面的一个目录，否则抛错。三种失败给三种提示。"""
    target = resolve_within(root, raw)
    if not is_within(root, target):
        raise OutOfSandboxError(
            f"路径 {str(raw)!r} 指向 {target}，超出了允许的范围 {root}。"
            f"只能列 {root} 里面的目录。"
        )
    if not target.exists():
        raise FileNotFoundError(f"{str(raw)!r} 不存在。用 '.' 可以列项目根目录。")
    if not target.is_dir():
        raise NotADirectoryError(
            f"{str(raw)!r} 是文件不是目录。list_files 只能列目录，看单个文件用 read_file。"
        )
    return target


def _children(directory: Path) -> list[tuple[str, bool, bool]]:
    """列出 directory 的直接子项，跳过规则和排序都在这儿。返回 (名字, 是不是目录, 是不是符号链接)。

    scandir 的条目对象在迭代器关掉之后不保证还能用，所以离开 with 之前就把要的信息取干净。
    """
    with os.scandir(directory) as scanner:
        entries = [
            (entry.name, entry.is_dir(), entry.is_symlink())
            for entry in scanner
            if not should_skip(Path(entry.path))
        ]
    # 目录在前、文件在后，两组各自按名字排（忽略大小写）。
    entries.sort(key=lambda entry: (not entry[1], entry[0].lower()))
    return entries


def _build_tree(directory: Path, depth: int, max_depth: int) -> list[_Node]:
    """把 directory 下面 max_depth 层建成节点树。depth 从 1 起算。"""
    nodes: list[_Node] = []
    for name, is_dir, is_symlink in _children(directory):
        children: list[_Node] = []
        # 不跟进符号链接：链成环会走不完。它照常显示成目录，只是不往里展开。
        if is_dir and not is_symlink and depth < max_depth:
            children = _build_tree(directory / name, depth + 1, max_depth)
        nodes.append(_Node(name, is_dir, children))
    return nodes


def _keep_breadth_first(nodes: list[_Node], max_nodes: int) -> bool:
    """按层挑选要显示的节点，返回是否因为撞上限被截断。

    为什么不按深度递归着挑：那样预算会全烧在第一个子树上。src/ 底下要是有几百个
    文件，你会只看见 src/ 深处的一截，tests/ 和别的顶层目录一个都不出现——而这工具
    是给人看整体结构的。按层挑能保证浅的层一定完整，被砍掉的是最深的那部分。
    """
    queue: deque[_Node] = deque(nodes)
    left = max_nodes
    while queue and left > 0:
        node = queue.popleft()
        node.kept = True
        left -= 1
        queue.extend(node.children)
    return bool(queue)


def _render_level(entries: list[tuple[str, bool, bool]]) -> list[str]:
    """把一层子项渲染成若干行。只用于 list_files，所以不需要前缀。"""
    lines: list[str] = []
    for index, (name, is_dir, _) in enumerate(entries):
        is_last = index == len(entries) - 1
        lines.append(f"{'└── ' if is_last else '├── '}{name}{'/' if is_dir else ''}")
    return lines


def _render_tree(nodes: list[_Node], prefix: str = "") -> list[str]:
    """把节点树渲染成带竖线的若干行。

    只在挑中的节点里算"是不是最后一个"，所以被截断时末尾那条也是 `└──`，
    不会留下一根悬着的 `├──` 让人以为下面还有。
    """
    kept = [node for node in nodes if node.kept]
    lines: list[str] = []
    for index, node in enumerate(kept):
        is_last = index == len(kept) - 1
        connector = "└── " if is_last else "├── "
        lines.append(f"{prefix}{connector}{node.name}{'/' if node.is_dir else ''}")
        lines.extend(_render_tree(node.children, prefix + ("    " if is_last else "│   ")))
    return lines
