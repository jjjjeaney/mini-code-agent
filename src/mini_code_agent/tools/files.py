"""文件类工具。

路径一律先过 safety 的围栏：解析、判越界、再决定要不要碰文件系统。

每个工具各自成一体：它用到的小函数都定义在它自己里面，读一个函数就能读懂一个工具。
例外是几个被多个工具共用的（`_slice_lines`、`_number_lines`、`_checked_path`、
`_children`），它们留在模块这一层，理由写在那儿。"""

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
    MAX_READ_FILE_CHARS,
    MAX_TOOL_CHARS,
    SENSITIVE_FILE_PATTERNS,
    NotTextFileError,
    SensitiveFileError,
    UndecodableTextError,
    detect_encoding,
    ensure_text_file,
    ensure_within_size,
    find_symlink,
    is_sensitive,
    resolve_inside,
    should_skip,
    truncate_text,
)
from mini_code_agent.tools.spill import TOOL_RESULTS_DIR

DEFAULT_MAX_DEPTH = 5
DEFAULT_MAX_NODES = 200

# project_tree 那两道闸的**上限**（上面两个是默认值）。没有上限的话，旋钮就攥在模型手里了
# ——传个 max_nodes: 100000，它就能把整棵树摊进上下文。别的工具的限都在代码里，这里不能是
# 例外。
MAX_TREE_DEPTH = 20
MAX_TREE_NODES = 1000

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

READ_TOOL_RESULT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "id": {
            "type": "string",
            "description": (
                "落盘结果的名字。某个工具输出太长时，它的结果里会写明存在哪儿、id 是什么，"
                "照抄那个 id；带不带 .txt 后缀都行"
            ),
        },
        "start_line": {
            "type": "integer",
            "description": "从第几行开始读，从 1 数起，默认 1",
        },
        "end_line": {
            "type": "integer",
            "description": "读到第几行（含这一行），默认读到末尾",
        },
    },
    "required": ["id"],
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
    return truncate_text("\n".join(render_level(entries)), MAX_TOOL_CHARS)


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
    # 两道闸都夹在默认值和上限之间：模型给得比默认小就听它的（少列一点是它的自由），
    # 给得比上限大就按上限来（那是它拿不到的自由）。
    max_depth = max(1, min(int(kwargs.get("max_depth") or DEFAULT_MAX_DEPTH), MAX_TREE_DEPTH))
    max_nodes = max(1, min(int(kwargs.get("max_nodes") or DEFAULT_MAX_NODES), MAX_TREE_NODES))

    nodes = build(root, depth=1)
    if not nodes:
        return "项目里没有可列出的内容：要么是空项目，要么内容都被过滤掉了。"

    truncated = keep_breadth_first(nodes)
    text = "\n".join(render(nodes))
    if truncated:
        # 说明放**前面**，不放末尾：下面还要过一道字符数闸，而它的说明按 truncate_text 的
        # 规矩挂在末尾——两条一起响的时候，末尾这句会被它连同树尾一起截掉。条目数这句更该
        # 留住（它说的是"树本身就不全"，不是"显示不下"）。
        text = f"[已截断：条目数到了上限 {max_nodes}，剩下的没列出来]\n\n{text}"
    # 条目数那道上限和字符数这道是两回事，都得有：一千条短名字也可能把上下文撑满，
    # 而两百个长名字未必。
    return truncate_text(text, MAX_TOOL_CHARS)


def _number_lines(text: str, start: int) -> str:
    """给每行标上它在**文件里**的真实行号。

    是文件里的行号，不是 1..N——只读一段时这两个数不一样，而 `replace_lines` 要的是前者。
    宽度按最后一行对齐，不然第 9 行和第 10 行的正文会差一格。

    分隔符固定是 ` | `。这段输出回头会被模型拿去当 `edit_file` 的 old_string（见那里的描述），
    那时得照着这个分隔符把前缀削掉——所以它不能变来变去。
    """
    lines = text.split("\n")
    width = len(str(start + len(lines) - 1))
    return "\n".join(
        f"{number:>{width}} | {line}" for number, line in enumerate(lines, start)
    )


def _slice_lines(text: str, start_line: Any, end_line: Any) -> tuple[list[str], int, int]:
    """按行号切一段。返回（选中那几行, 起始行号, 文件总行数）。行号从 1 数起、两头都含。

    read_file 和 read_tool_result 共用——两边的行号语义必须是同一个，"按行号取一段"
    这件事不该有两套规矩。
    """
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
    return lines[start - 1 : end], start, total


def read_file(kwargs: dict[str, Any], ctx: ToolContext) -> str:
    """读一个文本文件，可以只读其中一段。

    start_line / end_line 从 1 数起、两头都含，得是 ≥1 的整数。分段读长文件就靠它：
    先读一段，看末尾注明的"共 N 行"决定下一段从哪开始。

    **每行前面带着它在文件里的行号**（`12 | 正文`）。这是为了 `replace_lines`——它按位置改，
    而模型自己数行很容易错（每改一次，后面的行号全变）。行号不是文件内容的一部分，拿着这段
    正文去改文件时要去掉。

    检查顺序是有讲究的：先判路径（围栏、符号链接、敏感文件），再判类型和大小（便宜，
    而且能挡住大文件），然后开文件嗅一下是不是文本，最后才解码。任何一步不通过，
    内容都不会被读进来。
    """
    root = ctx.cwd.resolve()
    raw = kwargs["path"]
    target = _checked_path(root, raw)

    # 不顺着符号链接读。指向 cwd 外面的链接，围栏那一步已经拦下了；指向 cwd 里面的
    # 拦不住——顺着它读到的可能是另一个文件，模型以为读的是 A，实际读的是 B。
    link = find_symlink(raw, root)
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

    text, mtime_ns, size = _read_text(target, raw)

    selected, start, total = _slice_lines(text, kwargs.get("start_line"), kwargs.get("end_line"))

    if not selected:
        # 空文件，或者要从末尾之后开始读。这**不算失败**——只是没有内容可给；下面"读过"
        # 那笔照样要记，不记的话模型连一个空文件都改不了（改之前必须先读过）。
        body = f"{raw!r} 只有 {total} 行，从第 {start} 行开始没有内容了。"
    else:
        # **先编号、再截断**，顺序不能反：编号之后行号就是正文的一部分了，得跟着一起受
        # MAX_READ_FILE_CHARS 管着——这样总数仍然守在限额以内，代价只是同样长的文件现在
        # 少给几行。反过来（先截再编号）会超出限额，而且超多少取决于文件有多少行，那就没边了。
        #
        # 这一档比别的工具宽（见 safety.py 里那三个常量的说明）。宽不代表鼓励读满：
        # 它自带的 start_line / end_line 才是读长文件的正当姿势。
        body = truncate_text(_number_lines("\n".join(selected), start), MAX_READ_FILE_CHARS)
        # "共几行"这句接在**截断之后**：它是元信息，不是文件里的一行（所以不带行号），也不该
        # 被截断吃掉（以前它就拼在正文里，长文件一截就把它一起吞了）。
        # 只读了一段才说——整篇读下来时，行号本身已经说清了范围，多这句是噪音。
        if start > 1 or start + len(selected) - 1 < total:
            body += f"\n\n[文件共 {total} 行]"

    # 读成功了才记：路径 -> (读之前的 mtime, 整篇的字节数)。
    # 长度记的是**整篇**，不是返回给模型的那一段——后面改文件的工具要拿它跟磁盘上的现状
    # 比，两边都得是"文件自己的属性"，掺进"模型看到多少"就没法比了。
    # 放在 slice_lines 后面是有意的：行号范围写反会抛错，那一次模型什么都没拿到，
    # 不该留下"读过"的痕迹。
    ctx.read_state.record(target, size, mtime_ns)

    return body


def read_tool_result(kwargs: dict[str, Any], ctx: ToolContext) -> str:
    """取一条**落盘的工具结果**（谁落的盘见 tools/spill.py）。

    为什么不能直接用 read_file：落盘在 `.agent/` 底下，那是敏感目录，read_file 连开都不开。
    那道闸拦的是会话流水和改前备份，不能为了这个去撬它。所以这里是 harness 自己开的第二道
    门——和 read_memory / read_skill 一个形状：只认自己那个目录里的名字，围栏和符号链接
    照查。

    读法和 read_file 一致（带行号、可以只取一段），因为落下来的可能就是几 MB 的日志，
    不分段根本取不完。

    **不记 read_state**：这个文件改不了（`.agent` 在沙箱里是只读挂载，`is_sensitive` 也
    挡着写），没有"改之前要先读过"这回事。
    """
    directory = ctx.cwd.resolve() / TOOL_RESULTS_DIR
    raw = kwargs["id"]
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError("id 得是非空的字符串——引用消息里会给。")
    name = raw.strip()
    # 引用消息里印的路径带 .txt，模型可能整个抄过来。两种写法都收，省一次来回。
    if name.endswith(".txt"):
        name = name[: -len(".txt")]

    file_name = f"{name}.txt"
    target = resolve_inside(directory, file_name)
    link = find_symlink(file_name, directory)
    if link is not None:
        raise ValueError(
            f"{file_name!r} 的路径上有符号链接（{link}）。不顺着链接读——"
            f"那样读到的可能是另一个文件。"
        )
    if not target.is_file():
        raise FileNotFoundError(
            f"没有名为 {name!r} 的落盘结果。id 要用引用消息里给的那个（在那个目录里 "
            f"list_files 一下也行）。"
        )

    text = target.read_text(encoding="utf-8")
    selected, start, total = _slice_lines(text, kwargs.get("start_line"), kwargs.get("end_line"))

    if not selected:
        return f"{name!r} 只有 {total} 行，从第 {start} 行开始没有内容了。"

    body = truncate_text(_number_lines("\n".join(selected), start), MAX_READ_FILE_CHARS)
    # 和 read_file 同一句，只是措辞按"一个文件"来——这里没有文件名可印，名字就是 id。
    if start > 1 or start + len(selected) - 1 < total:
        body += f"\n\n[共 {total} 行]"
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

    # 只取内容：mtime 和长度这里用不上，核对状态那一步比的是 read_state 里模型那次
    # read_file 留下的记录。
    current, _, _ = _read_text(target, raw)

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

    current, _, _ = _read_text(target, raw)
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
    return truncate_text("\n".join(lines), MAX_TOOL_CHARS)


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
    return truncate_text(body, MAX_TOOL_CHARS)


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


def _read_text(target: Path, raw: str) -> tuple[str, int, int]:
    """把文件读成文本，顺带带出"读之前"量到的 mtime，和这次读到的字节数。

    大小、二进制、编码三道检查都在这儿——read_file 和 edit_file 都要走，各写一份的话
    以后补一道检查只会补在其中一份上。mtime 也在这儿量：读完再量就和内容错开版本了。

    字节数给的是 len(data)，不是 st_size：它和上面这份文本是同一次读出来的，跟 mtime
    属于同一个版本。st_size 是另一次系统调用，中间被人改了的话长度就和内容错开版本了
    ——那正是这个返回值要防的事。
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

    # 读完再量一次。对不上，说明读的过程中有人动过这个文件，手上这份可能是撕裂的。
    # 两个信号一起比：只比 mtime 会漏——同一个时钟刻度里的两次写，时间戳会完全一样
    # （实测这台机器上 10ms 以内经常撞）。长度这边比的是"这次真正读到的字节数"和
    # "磁盘上现在有多少字节"，直接对上手上这份和盘上那份。
    after = target.stat()
    if after.st_mtime_ns != mtime_ns or after.st_size != len(data):
        raise ValueError(f"{raw!r} 在读的过程中被改过了，这次拿到的内容不一定完整，请重读一次。")

    try:
        return data.decode(encoding), mtime_ns, len(data)
    except UnicodeDecodeError as exc:
        raise UndecodableTextError(
            f"{raw!r} 不是 {encoding} 编码（可能是 GBK 之类），没法当文本读。"
        ) from exc


def _checked_writable(root: Path, raw: str | Path) -> Path:
    """写类工具的路径闸门：围栏、符号链接、敏感路径、不是目录、上级目录存在。

    不查存在性——write_file 要能新建还不存在的文件，那件事留给调用方各自判。
    """
    target = resolve_inside(root, raw)

    link = find_symlink(raw, root)
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
                f"{raw!r} 已经存在，但你还没读过它。你必须先使用read_file工具读一遍完整文件"
                f"不然会把一份你从没看过的内容整个盖掉。"
            )
        recorded_mtime, recorded_size = recorded
        current = target.stat()
        # 两个信号一起比。只看 mtime 会漏：两次写落在同一个时钟刻度里，时间戳会完全一样
        # （实测这台机器上间隔 10ms 以内经常撞），那种时候只有长度变了才拦得住。
        if current.st_mtime_ns != recorded_mtime or current.st_size != recorded_size:
            raise StaleReadError(
                f"{raw!r} 在你读过之后被改过了（可能是别的程序，也可能是你自己另一次操作）。"
                f"先把文件完整重读一遍，再改。"
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
            # mtime 和长度都在顶替**之前**量。改名不改时间戳，所以这就是新内容将来在
            # 目标上的时间戳，而且不受"顶替之后、记录之前"那点窗口影响。长度必须从临时
            # 文件上量，不能拿 len(new_content)：非 ASCII 时字符数不等于字节数，而闸门
            # 比的是 st_size。
            new_stat = temp.stat()
            new_mtime_ns = new_stat.st_mtime_ns
            new_size = new_stat.st_size
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

    ctx.read_state.record(target, new_size, new_mtime_ns)
    return saved


def _checked_path(root: Path, raw: str | Path) -> Path:
    """在 _resolve_inside 之上再要求它确实存在。读类工具都用这个。"""
    target = resolve_inside(root, raw)
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



