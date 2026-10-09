"""文件类工具。

路径一律先过 safety 的围栏：解析、判越界、再决定要不要碰文件系统。

每个工具各自成一体：它用到的小函数都定义在它自己里面，读一个函数就能读懂一个工具。
唯一的例外是文件最底下那两个（`_checked_path` 和 `_children`），它们被多个工具共用，
留在模块这一层，理由写在那儿。"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

from mini_code_agent.tools.base import (
    BackupFailedError,
    ReplaceFailedError,
    StaleReadError,
    ToolContext,
    WriteFailedError,
)
from mini_code_agent.tools.safety import (
    EXCLUDED_DIRS,
    EXCLUDED_FILES,
    SENSITIVE_FILE_PATTERNS,
    NotTextFileError,
    OutOfSandboxError,
    SensitiveFileError,
    UndecodableTextError,
    detect_encoding,
    ensure_text_file,
    ensure_within_size,
    is_sensitive,
    is_within,
    resolve_within,
    should_skip,
    truncate_text,
)

DEFAULT_MAX_DEPTH = 5
DEFAULT_MAX_NODES = 200

# grep 最多返回多少条匹配。一条匹配就是一行，不设上限的话搜个 "e" 能把上下文塞满。
MAX_GREP_MATCHES = 200

# 判文本还是二进制，只看开头这么多字节。git 判断二进制用的也是这个量级。
SNIFF_BYTES = 8192

# 覆盖文件前，原内容往这儿留一份。是工具自己的痕迹，不该进模型的视野，
# 所以 safety 的跳过规则里也加了 .agent。
HISTORY_DIR = Path(".agent") / ".history"

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

READ_FILE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "path": {
            "type": "string",
            "description": "要读哪个文件，相对项目根目录。",
        },
        "start_line": {
            "type": "integer",
            "description": "从第几行开始读，从 1 数起，默认 1",
        },
        "end_line": {
            "type": "integer",
            "description": "读到第几行（含这一行），默认读到文件末尾",
        },
    },
    "required": ["path"],
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

GLOB_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "pattern": {
            "type": "string",
            "description": (
                "匹配模式，相对项目根目录。不含 / 时按文件名在整个项目里递归找，"
                "比如 *.py；含 / 时按相对路径匹配，比如 src/*.py。"
                "支持 *（不跨目录）、**（跨任意层）和 ?。"
            ),
        },
    },
    "required": ["pattern"],
}

GREP_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "pattern": {
            "type": "string",
            "description": "要搜的内容，可以是普通文本也可以是正则表达式",
        },
        "path": {
            "type": "string",
            "description": "搜哪个目录或文件，相对项目根目录，默认整个项目",
        },
        "glob": {
            "type": "string",
            "description": "限定只搜哪类文件，比如 *.py。省略就是所有文本文件",
        },
        "ignore_case": {
            "type": "boolean",
            "description": "是否忽略大小写，默认 false",
        },
    },
    "required": ["pattern"],
}

WRITE_FILE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "path": {
            "type": "string",
            "description": "要写到哪个文件，相对项目根目录",
        },
        "content": {
            "type": "string",
            "description": "文件的完整内容。是整篇，不是片段；是替换，不是追加",
        },
    },
    "required": ["path", "content"],
}

EDIT_FILE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "path": {
            "type": "string",
            "description": "改哪个文件，相对项目根目录",
        },
        "old_string": {
            "type": "string",
            "description": "要被换掉的那段原文，必须和文件里一模一样，包括缩进和空白",
        },
        "new_string": {
            "type": "string",
            "description": "换成什么。留空字符串就是把那段删掉",
        },
        "replace_all": {
            "type": "boolean",
            "description": "old_string 在文件里出现多次时是否全部替换。默认 false，只换第一处",
        },
    },
    "required": ["path", "old_string", "new_string"],
}

REPLACE_LINES_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "path": {
            "type": "string",
            "description": "改哪个文件，相对项目根目录",
        },
        "start_line": {
            "type": "integer",
            "description": "从第几行开始（含这一行），从 1 数起",
        },
        "end_line": {
            "type": "integer",
            "description": (
                "到第几行结束（含这一行）。可以等于 start_line-1，"
                "那表示不替换任何行、只在那个位置插入"
            ),
        },
        "new_text": {
            "type": "string",
            "description": "替换成什么。留空字符串就是把这几行删掉",
        },
    },
    "required": ["path", "start_line", "end_line", "new_text"],
}


def list_files(kwargs: dict[str, Any], ctx: ToolContext) -> str:
    """列出某个目录下面一层的文件和子目录，不往下钻。

    只看一层是有意的：它回答的是"这个目录里有什么"，想知道整体结构用 project_tree。
    目录带结尾斜杠，先排目录、再排文件，各自按名字排，每次跑出来都一样。

    cwd 从 ctx 拿，不从参数拿——沙箱边界必须由外部指定，
    要是让模型自己传，围栏就成了摆设。
    """

    def render_level(entries: list[tuple[str, bool, bool]]) -> list[str]:
        """把一层子项渲染成若干行。只排一层，所以不需要竖线前缀。"""
        lines: list[str] = []
        for index, (name, is_dir, _) in enumerate(entries):
            is_last = index == len(entries) - 1
            lines.append(f"{'└── ' if is_last else '├── '}{name}{'/' if is_dir else ''}")
        return lines

    root = ctx.cwd.resolve()
    raw = kwargs.get("path") or "."
    target = _checked_path(root, raw)
    if not target.is_dir():
        raise NotADirectoryError(
            f"{raw!r} 是文件不是目录。list_files 只能列目录，看单个文件用 read_file。"
        )

    entries = _children(target)
    if not entries:
        return f"{raw!r} 下面没有可列出的内容：要么是空目录，要么内容都被过滤掉了。"
    return truncate_text("\n".join(render_level(entries)))


def project_tree(kwargs: dict[str, Any], ctx: ToolContext) -> str:
    """把 cwd 整个项目画成一棵树，用来一眼看清整体结构。

    和 list_files 的区别是它会往下钻，所以拿 max_depth 和 max_nodes 两道闸控制规模：
    项目一大，无限制地列出来会把上下文塞满。撞上哪道闸都会在末尾说明。

    起点固定是 cwd，不接受路径参数——沙箱边界由外部指定，模型碰不到。
    """

    @dataclass
    class Node:
        name: str
        is_dir: bool
        children: list[Node] = field(default_factory=list)
        kept: bool = False

    def build(directory: Path, depth: int) -> list[Node]:
        """把 directory 下面 max_depth 层建成节点树。depth 从 1 起算。"""
        nodes: list[Node] = []
        for name, is_dir, is_symlink in _children(directory):
            children: list[Node] = []
            # 不跟进符号链接：链成环会走不完。它照常显示成目录，只是不往里展开。
            if is_dir and not is_symlink and depth < max_depth:
                children = build(directory / name, depth + 1)
            nodes.append(Node(name, is_dir, children))
        return nodes

    def keep_breadth_first(nodes: list[Node]) -> bool:
        """按层挑选要显示的节点，返回是否因为撞上限被截断。

        为什么不按深度递归着挑：那样预算会全烧在第一个子树上。src/ 底下要是有几百个
        文件，你会只看见 src/ 深处的一截，tests/ 和别的顶层目录一个都不出现——而这工具
        是给人看整体结构的。按层挑能保证浅的层一定完整，被砍掉的是最深的那部分。
        """
        queue: deque[Node] = deque(nodes)
        left = max_nodes
        while queue and left > 0:
            node = queue.popleft()
            node.kept = True
            left -= 1
            queue.extend(node.children)
        return bool(queue)

    def render(nodes: list[Node], prefix: str = "") -> list[str]:
        """把节点树渲染成带竖线的若干行。

        只在挑中的节点之间算"是不是最后一个"，所以被截断时末尾那条也是 `└──`，
        不会留下一根悬着的 `├──` 让人以为下面还有。
        """
        kept = [node for node in nodes if node.kept]
        lines: list[str] = []
        for index, node in enumerate(kept):
            is_last = index == len(kept) - 1
            connector = "└── " if is_last else "├── "
            lines.append(f"{prefix}{connector}{node.name}{'/' if node.is_dir else ''}")
            lines.extend(render(node.children, prefix + ("    " if is_last else "│   ")))
        return lines

    root = ctx.cwd.resolve()
    max_depth = max(1, int(kwargs.get("max_depth") or DEFAULT_MAX_DEPTH))
    max_nodes = max(1, int(kwargs.get("max_nodes") or DEFAULT_MAX_NODES))

    nodes = build(root, depth=1)
    if not nodes:
        return "项目里没有可列出的内容：要么是空项目，要么内容都被过滤掉了。"

    truncated = keep_breadth_first(nodes)
    text = "\n".join(render(nodes))
    if truncated:
        text += f"\n\n[已截断：条目数到了上限 {max_nodes}，剩下的没列出来]"
    return text


def read_file(kwargs: dict[str, Any], ctx: ToolContext) -> str:
    """读一个文本文件，可以只读其中一段。

    start_line / end_line 从 1 数起、两头都含，得是 ≥1 的整数。分段读长文件就靠它：
    先读一段，看末尾注明的"共 N 行"决定下一段从哪开始。

    检查顺序是有讲究的：先判路径（围栏、符号链接、敏感文件），再判类型和大小（便宜，
    而且能挡住大文件），然后开文件嗅一下是不是文本，最后才解码。任何一步不通过，
    内容都不会被读进来。
    """

    def slice_lines(text: str, start_line: Any, end_line: Any) -> str:
        """按行号切一段。行号从 1 数起，两头都含。"""
        lines = text.splitlines()
        total = len(lines)

        asked_start = None if start_line is None else _line_number(start_line, "start_line")
        asked_end = None if end_line is None else _line_number(end_line, "end_line")
        start = 1 if asked_start is None else asked_start

        # 只拿用户自己给的数比。没给 end_line 的时候不能拿"文件末尾"去跟 start 比——
        # 那样"读一个 2 行文件的第 50 行"会报成范围不成立，而真正的原因是文件没那么长。
        if asked_end is not None and asked_end < start:
            raise ValueError(f"end_line({asked_end}) 比 start_line({start}) 还小，这个范围不成立。")

        end = total if asked_end is None else min(asked_end, total)
        selected = lines[start - 1 : end]
        if not selected:
            return f"{raw!r} 只有 {total} 行，从第 {start} 行开始没有内容了。"

        body = "\n".join(selected)
        # 只读了一段才注明范围。整篇读下来时不用多说，免得平白多一行噪音。
        if start > 1 or end < total:
            body += f"\n\n[这是第 {start}-{end} 行，文件共 {total} 行]"
        return body

    root = ctx.cwd.resolve()
    raw = kwargs["path"]
    target = _checked_path(root, raw)

    # 不顺着符号链接读。指向 cwd 外面的链接，围栏那一步已经拦下了；指向 cwd 里面的
    # 拦不住——顺着它读到的可能是另一个文件，模型以为读的是 A，实际读的是 B。
    link = _find_symlink(raw, root)
    if link is not None:
        raise ValueError(
            f"{raw!r} 的路径上有符号链接（{link}）。read_file 不顺着链接读——"
            f"那样读到的可能是另一个文件，你以为读的是 A，实际读的是 B。"
        )

    # 拦在最前面，连开都不开。.env 就躺在项目根目录，一次 read 就能让 key 随下一轮
    # 对话发出去，而且全程没有任何报错——这是这个工具最需要挡住的一件事。
    if is_sensitive(target):
        raise SensitiveFileError(f"{raw!r} 是敏感文件（可能装着密钥或凭据），不能读。")

    if target.is_dir():
        raise IsADirectoryError(f"{raw!r} 是目录不是文件。想看目录里有什么就用 list_files。")
    if not target.is_file():
        raise ValueError(f"{raw!r} 不是普通文件（可能是管道、设备文件之类），读不了。")

    text, mtime_ns = _read_text(target, raw)

    body = truncate_text(slice_lines(text, kwargs.get("start_line"), kwargs.get("end_line")))

    # 读成功了才记：路径 -> (读之前的 mtime, 整篇的长度)。
    # 长度记的是**整篇**，不是返回给模型的那一段——后面改文件的工具要拿它跟磁盘上的现状
    # 比，两边都得是"文件自己的属性"，掺进"模型看到多少"就没法比了。
    # 放在 slice_lines 后面是有意的：行号范围写反会抛错，那一次模型什么都没拿到，
    # 不该留下"读过"的痕迹。
    ctx.read_state.record(target, text, mtime_ns)

    return body


def write_file(kwargs: dict[str, Any], ctx: ToolContext) -> str:
    """用完整内容写一个文件：不存在就新建，存在就整个替换。

    传的是**整篇**内容，不是片段；改局部用 edit_file。

    这个工具只管写。要不要批准、要不要问用户是 harness 的事，这里不做。

    闸门在 _checked_writable 里（围栏、符号链接、敏感路径），后面的写临时文件、核对
    状态、备份、原子顶替在 _commit_change 里——两条都和 edit_file 共用同一份。
    """
    root = ctx.cwd.resolve()
    raw = kwargs["path"]
    content = kwargs["content"]
    if not isinstance(content, str):
        raise ValueError(f"content 得是字符串，收到的是 {type(content).__name__}。")

    target = _checked_writable(root, raw)
    saved = _commit_change(root, target, raw, content, target.exists(), ctx)

    if saved is None:
        return f"新建了 {raw}（{len(content.splitlines())} 行）。"
    return (
        f"覆盖了 {raw}（{len(content.splitlines())} 行）。"
        f"原内容备份在 {saved.relative_to(root).as_posix()}。"
    )


def edit_file(kwargs: dict[str, Any], ctx: ToolContext) -> str:
    """把文件里的一段文本替换掉。

    old_string 必须和文件里**一模一样**，包括缩进和空白——差一个空格就找不到，报错里
    会说清。它在文件里出现多次时：replace_all=True 全部替换，否则只替换**第一处**，
    并在返回里注明原本有几处，好让模型自己判断改的地方对不对。

    新内容是在内存里算好的，往后跟 write_file 共用同一套：写临时文件、核对状态、备份、
    原子顶替。核对放在顶替之前是有必要的——从读文件到真正落盘之间隔着算替换、写临时
    文件，那段时间里别人改的东西要是不核对就被直接盖掉了。
    """
    root = ctx.cwd.resolve()
    raw = kwargs["path"]
    old = kwargs["old_string"]
    new = kwargs["new_string"]
    replace_all = bool(kwargs.get("replace_all"))

    for name, value in (("old_string", old), ("new_string", new)):
        if not isinstance(value, str):
            raise ValueError(f"{name} 得是字符串，收到的是 {type(value).__name__}。")
    if not old:
        raise ValueError("old_string 不能为空——空文本在哪儿都匹配得上，替换出什么没法预料。")
    if old == new:
        raise ValueError("old_string 和 new_string 一模一样，这一改等于什么都没做。")

    target = _checked_writable(root, raw)
    if not target.exists():
        raise FileNotFoundError(f"{raw!r} 不存在。edit_file 只改已有的文件，新建用 write_file。")
    if not target.is_file():
        raise ValueError(f"{raw!r} 不是普通文件（可能是管道、设备文件之类），改不了。")

    current, _ = _read_text(target, raw)

    count = current.count(old)
    if count == 0:
        raise ValueError(
            f"{raw!r} 里找不到 old_string。它得和文件里的原文一模一样，包括缩进和空白——"
            f"先 read_file 看一眼再照着写。"
        )
    updated = current.replace(old, new) if replace_all else current.replace(old, new, 1)

    saved = _commit_change(root, target, raw, updated, True, ctx)

    done = count if replace_all else 1
    message = f"改了 {raw}（替换 {done} 处，文件现在 {len(updated.splitlines())} 行）。"
    if count > 1 and not replace_all:
        message += f" 注意：old_string 在文件里出现了 {count} 次，这次只替换了第一处。"
    if saved is not None:
        message += f" 原内容备份在 {saved.relative_to(root).as_posix()}。"
    return message


def replace_lines(kwargs: dict[str, Any], ctx: ToolContext) -> str:
    """按行号替换文件里的一段：换掉 start_line 到 end_line（两头都含）。

    增删改都用它——new_text 比原来长就是在插内容，短就是在删，空串就是整段删掉。
    和 edit_file 的分工：那个按内容找位置（适合"把某个东西改掉"），这个按行号定位
    （适合"把这几行换掉"，不用把原文一字不差地抄出来）。

    end_line 可以等于 start_line - 1，那是个**空区间**：不替换任何行，只在这个位置插入。
    在文件末尾追加就写 start_line = 总行数+1、end_line = 总行数。没有这个空区间的话，
    想在文件头尾插内容就得把旁边那行原样重抄一遍——那正是它该替你省掉的事。

    new_text 不是空的话会自动补一个换行。不补的话，拼出来的下一行会和它粘在一起
    （变成 `新内容旧行` 一行）；补上才能保证它自己占整行。空串不补——那是"删掉这段"，
    本来就没有内容需要占行。

    写法和 edit_file 一样：整篇读进内存、在内存里拼好，再交给同一套（写临时文件、
    核对状态、备份、原子顶替）。所以 read-before-write 那几道闸和它完全一致。
    """
    root = ctx.cwd.resolve()
    raw = kwargs["path"]
    new_text = kwargs["new_text"]
    if not isinstance(new_text, str):
        raise ValueError(f"new_text 得是字符串，收到的是 {type(new_text).__name__}。")

    start = _line_number(kwargs["start_line"], "start_line")
    end = _line_number(kwargs["end_line"], "end_line", minimum=0)

    target = _checked_writable(root, raw)
    if not target.exists():
        raise FileNotFoundError(
            f"{raw!r} 不存在。replace_lines 只改已有的文件，新建用 write_file。"
        )
    if not target.is_file():
        raise ValueError(f"{raw!r} 不是普通文件（可能是管道、设备文件之类），改不了。")

    current, _ = _read_text(target, raw)
    # keepends=True：每行连自己的换行一起留着，这样区间外的内容能原样拼回去，
    # 不会因为我们重新拼一遍就把文件尾部的换行、或者 CRLF 之类的换行风格改掉。
    lines = current.splitlines(keepends=True)
    total = len(lines)

    if start > total + 1:
        raise ValueError(f"start_line({start}) 超出了文件：{raw!r} 一共 {total} 行。")
    if end > total:
        raise ValueError(f"end_line({end}) 超出了文件：{raw!r} 一共 {total} 行。")
    if end < start - 1:
        raise ValueError(f"end_line({end}) 比 start_line({start}) 小太多，这个范围不成立。")

    body = new_text if not new_text or new_text.endswith("\n") else new_text + "\n"
    updated = "".join(lines[: start - 1]) + body + "".join(lines[end:])

    saved = _commit_change(root, target, raw, updated, True, ctx)

    removed = max(0, end - start + 1)
    added = len(body.splitlines())
    if removed == 0:
        what = f"在第 {start} 行前面插了 {added} 行"
    elif added == 0:
        what = f"删掉了第 {start}-{end} 行（{removed} 行）"
    else:
        what = f"第 {start}-{end} 行换成了 {added} 行"

    message = f"改了 {raw}（{what}，文件现在 {len(updated.splitlines())} 行）。"
    if saved is not None:
        message += f" 原内容备份在 {saved.relative_to(root).as_posix()}。"
    return message


def glob(kwargs: dict[str, Any], ctx: ToolContext) -> str:
    """按文件名或路径模式，递归查找项目里的文件和目录。

    模式里不含 `/` 时按文件名匹配、整棵树递归找——`*.py` 找得到 src/core/agent.py；
    含 `/` 时按相对路径匹配。要跨层就用 `**`。

    匹配用的是 `_compile_glob`（和 grep 共用同一套 glob 语义，所以两个工具不会各说各话）。
    匹配对象是相对路径，结果按构造就不可能跑出 cwd——`..` 和绝对路径压根匹配不上任何东西。
    """

    def collect(directory: Path, relative: PurePosixPath) -> None:
        """递归走一遍，边走边挑。"""
        for name, is_dir, is_symlink in _children(directory):
            child = relative / name
            if matcher.match(child.as_posix()):
                found.append((child, is_dir))
            # 不跟进符号链接：链成环会走不完。
            if is_dir and not is_symlink:
                collect(directory / name, child)

    root = ctx.cwd.resolve()
    pattern = (kwargs["pattern"] or "").strip()
    if not pattern:
        raise ValueError("pattern 不能为空。想找所有文件可以写 *。")

    # 这两种写法匹配不上任何相对路径。与其返回空结果让人猜，不如直接说清楚。
    if Path(pattern).is_absolute() or ".." in Path(pattern).parts:
        raise ValueError(f"模式 {pattern!r} 要相对项目根目录写，不能用绝对路径，也不能有 ..。")

    matcher = _compile_glob(pattern)

    found: list[tuple[PurePosixPath, bool]] = []
    collect(root, PurePosixPath("."))
    if not found:
        return f"没有匹配 {pattern!r} 的文件或目录。"

    found.sort(key=lambda item: item[0].as_posix().lower())
    lines = [f"{path.as_posix()}{'/' if is_dir else ''}" for path, is_dir in found]
    return truncate_text("\n".join(lines))


def grep(kwargs: dict[str, Any], ctx: ToolContext) -> str:
    """按内容搜索，返回每条匹配的位置：`文件:行号:那一行`。

    优先用 ripgrep，找不到就纯 Python 整棵树走一遍。两条路共用同一套跳过规则和同一种
    glob 语义，所以换不换 rg，结果应该一致。

    pattern 按正则处理（跟 rg 一样）。**这个工具会读到文件内容，所以敏感文件的过滤放在
    结果这一层**——不管 rg 认不认那套排除规则，最后都从结果里剔掉。
    """

    def rg_excludes() -> list[str]:
        """把跳过规则翻译成 rg 的 --glob 排除项。

        这只影响速度和"要不要真去读那个文件"，正确性由 keep() 保证。两件事分开写，
        免得以后有人以为光靠这个就够了。
        """
        args: list[str] = []
        for name in sorted(EXCLUDED_DIRS):
            args += ["--glob", f"!**/{name}/**"]
        for name in sorted(EXCLUDED_FILES) + list(SENSITIVE_FILE_PATTERNS):
            args += ["--glob", f"!**/{name}"]
        return args

    def run_rg(exe: str) -> list[tuple[str, int, str]] | None:
        """跑 rg 并解析输出。拉不起来、或 rg 自己出错（退出码 2）就返回 None 交给兜底。"""
        nonlocal truncated

        args = [
            exe,
            "--line-number",
            "--with-filename",
            "--no-heading",
            "--color=never",
            "--no-messages",
            "--hidden",
            "--no-ignore",
        ]
        if ignore_case:
            args.append("--ignore-case")
        if glob_pattern:
            args += ["--glob", glob_pattern]
        args += rg_excludes()
        args += ["-e", pattern, target_arg]

        # cwd 设成 root、目标传相对路径，这样 rg 吐出来的路径也是相对的：解析时按前两个
        # 冒号切就行。要是传绝对路径，Windows 上那个盘符冒号会把切分带错位。
        try:
            process = subprocess.Popen(
                args,
                cwd=root,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
        except OSError:
            return None

        found: list[tuple[str, int, str]] = []
        with process:
            for raw in process.stdout or ():
                if len(found) >= MAX_GREP_MATCHES:
                    truncated = True
                    process.kill()
                    break
                parts = raw.rstrip("\n").split(":", 2)
                if len(parts) != 3 or not parts[1].isdigit():
                    continue
                # rg 在 Windows 上吐的是 `.\src\a.py`（反斜杠 + 点斜杠前缀）。统一成
                # `src/a.py`：不然两条后端给出的路径长得不一样，而且反斜杠路径喂回别的
                # 工具也难看。只在 Windows 上替换——Linux 上反斜杠是合法的文件名字符。
                path = parts[0].replace("\\", "/") if os.name == "nt" else parts[0]
                found.append((path.removeprefix("./"), int(parts[1]), parts[2]))
            process.wait()
        # rg 退出码 2 表示它自己出错（比如正则写坏了）。这时候空结果不是"没有匹配"，
        # 得让纯 Python 那条路去跑，它会给出说得清楚的报错。
        if not found and process.returncode not in (0, 1):
            return None
        return found

    def scan(directory: Path, relative: PurePosixPath) -> None:
        """纯 Python 兜底：整棵树走一遍，一个文件一个文件搜。"""
        for name, is_dir, is_symlink in _children(directory):
            if len(matches) >= MAX_GREP_MATCHES:
                return
            path = directory / name
            child = relative / name
            if is_dir:
                # 不跟进符号链接：链成环会走不完。
                if not is_symlink:
                    scan(path, child)
                continue
            if matcher is not None and not matcher.match(child.as_posix()):
                continue
            search_in(path)

    def search_in(path: Path) -> None:
        try:
            ensure_within_size(path)
        except ValueError:
            return  # 太大的跳过，不为一个文件中断整次搜索
        with path.open("rb") as fh:
            sample = fh.read(SNIFF_BYTES)
            if not ensure_text_file(sample):
                return  # 二进制跳过
            fh.seek(0)
            data = fh.read()
        try:
            text = data.decode(detect_encoding(sample) or "utf-8")
        except UnicodeDecodeError:
            return  # 编码解不开也跳过——grep 不该因为一两个文件整体失败
        for number, line in enumerate(text.splitlines(), start=1):
            if regex.search(line):
                matches.append((path.relative_to(root).as_posix(), number, line))
                if len(matches) >= MAX_GREP_MATCHES:
                    return

    def keep(relative: str) -> bool:
        """结果层的过滤：敏感文件和噪音文件一律不进结果。

        rg 那条路的输出也要过这一道。它用的是自己的排除规则，而排除规则只管"搜不搜"，
        管不了"搜到了也不许说"。这里是最后一道，也是唯一保证得住的一道。
        """
        path = root / relative
        return not should_skip(path) and not is_sensitive(path)

    root = ctx.cwd.resolve()
    pattern = kwargs["pattern"]
    if not pattern:
        raise ValueError("pattern 不能为空。")

    ignore_case = bool(kwargs.get("ignore_case"))
    glob_pattern = (kwargs.get("glob") or "").strip()
    matcher: re.Pattern[str] | None = None
    if glob_pattern:
        if Path(glob_pattern).is_absolute() or ".." in Path(glob_pattern).parts:
            raise ValueError(
                f"glob {glob_pattern!r} 要相对项目根目录写，不能用绝对路径，也不能有 ..。"
            )
        matcher = _compile_glob(glob_pattern)

    try:
        regex = re.compile(pattern, re.IGNORECASE if ignore_case else 0)
    except re.error as exc:
        raise ValueError(f"pattern 不是合法的正则表达式：{exc}") from exc

    target = _checked_path(root, kwargs.get("path") or ".")
    if not (target.is_dir() or target.is_file()):
        raise ValueError(f"{target} 既不是目录也不是普通文件，搜不了。")
    # 目标传相对路径给 rg，输出里的路径才会是相对的（见 run_rg 里的说明）。
    target_arg = target.relative_to(root).as_posix()

    matches: list[tuple[str, int, str]] = []
    truncated = False

    exe = shutil.which("rg")
    outcome = run_rg(exe) if exe else None
    if outcome is None:
        scan(target, PurePosixPath("."))
        truncated = len(matches) >= MAX_GREP_MATCHES
    else:
        # run_rg 返回的就是匹配列表本身；truncated 它是通过 nonlocal 直接改的。
        matches = outcome

    matches = [item for item in matches if keep(item[0])]
    if not matches:
        return f"没有找到包含 {pattern!r} 的地方。"

    # 排一下序。两条后端吐出来的顺序本来不一样（rg 按自己的目录遍历，纯 Python 走的是
    # 目录优先的深度优先），不排的话同一份代码换台机器结果顺序就变了。
    matches.sort(key=lambda item: (item[0].lower(), item[1]))

    body = "\n".join(f"{path}:{number}:{text}" for path, number, text in matches)
    if truncated:
        body += f"\n\n[已截断：匹配数到了上限 {MAX_GREP_MATCHES}，后面还有]"
    return truncate_text(body)


# ——— 下面是几个工具共用的，所以没嵌进任何一个工具里 ———


def _compile_glob(pattern: str) -> re.Pattern[str]:
    r"""把 glob 翻译成正则，锚在字符串开头。

    `**/` 跨任意层（包括零层），`*` 和 `?` 都不跨 `/`。不含 `/` 的模式前面补一个 `**/`，
    意思就是"在任意层找这个名字"——`*.py` 该匹配整棵树里所有 .py。

    没用 pathlib 是因为 `PurePath.match` 拿模式尾部跟路径尾部对齐：
    `PurePath("deep/src/cli.py").match("src/*.py")` 返回 True，那不是"相对某个目录"
    的意思。实测见 tests/test_files.py 里那条对照用例。

    glob 和 grep 共用这一份：两个工具必须对"*"是什么意思给出同一个答案。
    """
    plain = pattern.removeprefix("./")
    if "/" not in plain:
        plain = f"**/{plain}"

    out: list[str] = []
    index = 0
    while index < len(plain):
        if plain.startswith("**/", index):
            out.append("(?:[^/]+/)*")
            index += 3
        elif plain.startswith("**", index):
            out.append(".*")
            index += 2
        elif plain[index] == "*":
            out.append("[^/]*")
            index += 1
        elif plain[index] == "?":
            out.append("[^/]")
            index += 1
        else:
            out.append(re.escape(plain[index]))
            index += 1
    return re.compile("".join(out) + r"\Z")


def _resolve_inside(root: Path, raw: str | Path) -> Path:
    """解析路径并确认它落在 root 里面。存不存在、是不是目录由调用方各自判。

    这是安全闸门，几个工具都要走。抄成多份的话，以后补一条校验只会补在其中一份上——
    安全校验有多份，就等于只会被修好一份。写文件那个工具要允许"还不存在"，
    所以围栏和"必须存在"是分开的两件事。
    """
    target = resolve_within(root, raw)
    if not is_within(root, target):
        raise OutOfSandboxError(
            f"路径 {str(raw)!r} 指向 {target}，超出了允许的范围 {root}。"
            f"只能访问 {root} 里面的东西。"
        )
    return target


def _find_symlink(raw: str | Path, root: Path) -> Path | None:
    """沿着**没解析过**的路径一段段看，返回第一段符号链接。

    为什么不能拿解析后的 target 去判 `is_symlink()`：链接在 resolve_within 那一步就
    被展开了，等拿到 target，"这是链接"这个信息已经没了。只能顺着原始路径重新走一遍。

    和围栏一样，这是安全校验——将来 edit_file 也要走同一份。
    """
    unresolved = Path(raw)
    if unresolved.is_absolute():
        try:
            unresolved = unresolved.relative_to(root)
        except ValueError:
            return None  # 压根不在 root 下面，围栏那一步会另行拒绝
    current = root
    for part in unresolved.parts:
        current = current / part
        if current.is_symlink():
            return current
    return None


def _line_number(value: Any, name: str, minimum: int = 1) -> int:
    """把模型给的行号参数转成整数，低于 minimum 就报错。

    校验得严一点。直接 int() 的话，模型传 0（以为是从 0 数起）会被悄悄当成第 1 行、
    传 "abc" 只会得到一句 invalid literal for int()——两种都太含糊，模型都不知道
    自己错在哪。布尔值也要挡：bool 是 int 的子类，放进去 True 就成了第 1 行。
    浮点也不收：int(2.7) 会静默截成 2，一样是把错误藏起来。

    minimum 默认是 1（行号从 1 数起）；只有"区间终点"这种允许空区间的场合才传 0。
    """
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise ValueError(f"{name} 得是整数，收到的是 {type(value).__name__}：{value!r}")
    try:
        number = int(value)
    except ValueError:
        raise ValueError(f"{name} 得是整数，收到的是 {value!r}") from None
    if number < minimum:
        if minimum == 0:
            raise ValueError(f"{name} 不能是负数，收到的是 {number}")
        raise ValueError(f"{name} 从 {minimum} 数起，不能是 {number}")
    return number


def _read_text(target: Path, raw: str) -> tuple[str, int]:
    """把文件读成文本，顺带带出"读之前"量到的 mtime。

    大小、二进制、编码三道检查都在这儿——read_file 和 edit_file 都要走，各写一份的话
    以后补一道检查只会补在其中一份上。mtime 也在这儿量：读完再量就和内容错开版本了。
    """
    ensure_within_size(target)

    mtime_ns = target.stat().st_mtime_ns

    with target.open("rb") as fh:
        sample = fh.read(SNIFF_BYTES)
        if not ensure_text_file(sample):
            raise NotTextFileError(f"{raw!r} 看着是二进制文件，没法当文本读。")
        encoding = detect_encoding(sample) or "utf-8"
        # 采样那一段不浪费：seek 回开头，一个句柄把整篇读完。
        fh.seek(0)
        data = fh.read()

    # 读完再量一次。两次不一样，说明读的过程中有人动过这个文件，手上这份可能是撕裂的。
    if target.stat().st_mtime_ns != mtime_ns:
        raise ValueError(f"{raw!r} 在读的过程中被改过了，这次拿到的内容不一定完整，请重读一次。")

    try:
        return data.decode(encoding), mtime_ns
    except UnicodeDecodeError as exc:
        raise UndecodableTextError(
            f"{raw!r} 不是 {encoding} 编码（可能是 GBK 之类），没法当文本读。"
        ) from exc


def _checked_writable(root: Path, raw: str | Path) -> Path:
    """写类工具的路径闸门：围栏、符号链接、敏感路径、不是目录、上级目录存在。

    不查存在性——write_file 要能新建还不存在的文件，那件事留给调用方各自判。
    """
    target = _resolve_inside(root, raw)

    link = _find_symlink(raw, root)
    if link is not None:
        raise ValueError(
            f"{raw!r} 的路径上有符号链接（{link}）。不顺着链接写——"
            f"那样改的是另一个文件，你以为改的是 A，实际改的是 B。"
        )

    if is_sensitive(target):
        raise SensitiveFileError(
            f"{raw!r} 是敏感位置（可能装着密钥或凭据，或者本身就是 .git 这种目录），"
            f"不能写。覆盖它可能毁掉凭据；在里面新建同样危险——"
            f"比如 .git/hooks/ 下放个脚本，会在你下次提交时被执行。"
        )

    if target.is_dir():
        raise IsADirectoryError(f"{raw!r} 是个目录。只能写文件，不能写目录。")
    if not target.parent.is_dir():
        raise FileNotFoundError(f"{raw!r} 的上级目录不存在。不会替你建目录——路径是不是写错了？")
    return target


def _commit_change(
    root: Path, target: Path, raw: str, new_content: str, existed: bool, ctx: ToolContext
) -> Path | None:
    """改文件的公共后半段：写临时文件 -> 核对状态 -> 备份 -> 原子顶替 -> 记 read_state。

    返回备份路径（本来没有这个文件就是 None）。write_file 和 edit_file 只有"新内容是
    怎么算出来的"不同，从这里往后一模一样，所以只留一份。

    顺序是刻意的：**状态核对放在顶替之前**，不是函数开头。它前面隔着建临时文件、写内容、
    备份这几件事，搁开头的话，核对和顶替之间就空出一整轮文件操作的时间。
    """

    def default_file_mode() -> int:
        """新文件该用的权限：0o666 去掉进程的 umask，也就是系统默认的那一档。

        mkstemp 建出来的是 0o600（只有自己能读写），对新建的源码文件来说太紧。
        Python 没有 getumask，只能设一次读一次再把旧的设回去——窗口是进程级的，
        但只有这一瞬间。
        """
        current = os.umask(0o022)
        os.umask(current)
        return 0o666 & ~current

    def match_permissions(temp_file: Path) -> None:
        """让临时文件的权限和"将来那个目标文件该有的"一致。

        覆盖时沿用原文件的（把一个 0o644 的文件换成 0o600 就是悄悄收紧了权限），
        新建时给系统默认那一档。都是**尽量**：拷不过来或者设不上也照样往下写，
        丢权限总比丢内容好。
        """
        try:
            if existed:
                shutil.copymode(target, temp_file)
            else:
                os.chmod(temp_file, default_file_mode())
        except OSError:
            pass

    def ensure_state_agrees() -> None:
        """顶替之前核对状态：文件还在不在、读过没有、读过之后动过没有。"""
        if target.exists() != existed:
            what = "冒出来了" if not existed else "不见了"
            raise StaleReadError(
                f"{raw!r} 在准备写入的过程中{what}，状态和开始时对不上，这次先不写。"
            )
        if not existed:
            return
        recorded = ctx.read_state.entries.get(target)
        if recorded is None:
            raise StaleReadError(
                f"{raw!r} 已经存在，但你还没读过它。先读一遍再改——"
                f"不然会把一份你从没看过的内容整个盖掉。"
            )
        recorded_mtime, _ = recorded
        if target.stat().st_mtime_ns != recorded_mtime:
            raise StaleReadError(
                f"{raw!r} 在你读过之后被改过了（可能是别的程序，也可能是你自己另一次操作）。"
                f"先重读一遍，再改。"
            )

    def backup_existing() -> Path:
        """把原文件留一份。名字带时间戳，同一个文件改多次就留下多份历史。"""
        relative = target.relative_to(root)
        name = f"{relative.name}.{time.strftime('%Y%m%d-%H%M%S')}"
        saved = root / HISTORY_DIR / relative.parent / name
        # 撞名就往后排，**绝不覆盖**：先来的那份才是更早的原始内容，正是要留住的。
        # 不指望"时间戳够细"来避免撞名——Windows 上 time 的精度可能粗到毫秒，
        # 同一瞬间连着改两次会拿到一模一样的时间戳（实跑里撞到过，还因此丢过一份备份）。
        counter = 1
        while saved.exists():
            saved = saved.with_name(f"{name}-{counter}")
            counter += 1
        try:
            saved.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(target, saved)
        except OSError as exc:
            raise BackupFailedError(f"{raw!r} 没被改动；备份原文件失败：{exc}") from exc
        return saved

    try:
        handle_fd, temp_name = tempfile.mkstemp(
            dir=target.parent, prefix=f".{target.name}.", suffix=".tmp"
        )
    except OSError as exc:
        raise WriteFailedError(f"{raw!r} 没被改动；建临时文件失败：{exc}") from exc

    temp = Path(temp_name)
    try:
        try:
            # newline="" 表示不做换行转换：给什么就写什么。默认行为在 Windows 上会把
            # \n 翻成 \r\n，那会让一个 LF 的文件每写一次就整体变一次行尾。
            with os.fdopen(handle_fd, "w", encoding="utf-8", newline="") as fh:
                fh.write(new_content)
            match_permissions(temp)
            # mtime 在顶替**之前**量。改名不改时间戳，所以这就是新内容将来在目标上的
            # 时间戳，而且不受"顶替之后、记录之前"那点窗口影响。
            new_mtime_ns = temp.stat().st_mtime_ns
        except OSError as exc:
            raise WriteFailedError(f"{raw!r} 没被改动；写临时文件失败：{exc}") from exc

        ensure_state_agrees()
        saved = backup_existing() if existed else None

        try:
            os.replace(temp, target)
        except OSError as exc:
            raise ReplaceFailedError(f"{raw!r} 没被改动；顶替那一步失败：{exc}") from exc
    finally:
        # 顶替成功后 temp 已经被改走、不存在了，这里是 no-op；失败时把临时文件收掉。
        # 收尾自己出的错不该盖掉真正的错因。
        try:
            temp.unlink(missing_ok=True)
        except OSError:
            pass

    ctx.read_state.record(target, new_content, new_mtime_ns)
    return saved


def _checked_path(root: Path, raw: str | Path) -> Path:
    """在 _resolve_inside 之上再要求它确实存在。读类工具都用这个。"""
    target = _resolve_inside(root, raw)
    if not target.exists():
        raise FileNotFoundError(f"{str(raw)!r} 不存在。路径要相对项目根目录写。")
    return target


def _children(directory: Path) -> list[tuple[str, bool, bool]]:
    """列出 directory 的直接子项，跳过规则和排序都在这儿。返回 (名字, 是不是目录, 是不是符号链接)。

    同样留在这儿：跳过规则和排序是"两个工具必须一致"的东西，只有一份定义才谈得上
    一致。scandir 的条目对象在迭代器关掉之后不保证还能用，所以离开 with 之前就把
    要的信息取干净。
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



