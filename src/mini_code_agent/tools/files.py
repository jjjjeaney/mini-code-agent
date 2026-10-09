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
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

from mini_code_agent.tools.base import ToolContext
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

    检查顺序是有讲究的：先判类型和大小（便宜，而且能挡住大文件），再开文件嗅一下
    是不是文本，最后才解码。任何一步不通过，内容都不会被读进来。
    """

    def as_line_number(value: Any, name: str) -> int | None:
        """把模型给的行号参数转成整数；没给（None）就返回 None。

        校验得严一点。直接 int() 的话，模型传 0（以为是从 0 数起）会被悄悄当成第 1 行、
        传 "abc" 只会得到一句 invalid literal for int()——两种都太含糊，模型都不知道
        自己错在哪。布尔值也要挡：bool 是 int 的子类，放进去 True 就成了第 1 行。
        浮点也不收：int(2.7) 会静默截成 2，一样是把错误藏起来。
        """
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, (int, str)):
            raise ValueError(f"{name} 得是整数，收到的是 {type(value).__name__}：{value!r}")
        try:
            number = int(value)
        except ValueError:
            raise ValueError(f"{name} 得是整数，收到的是 {value!r}") from None
        if number < 1:
            raise ValueError(f"{name} 从 1 数起，不能是 {number}")
        return number

    def slice_lines(text: str, start_line: Any, end_line: Any) -> str:
        """按行号切一段。行号从 1 数起，两头都含。"""
        lines = text.splitlines()
        total = len(lines)

        asked_start = as_line_number(start_line, "start_line")
        asked_end = as_line_number(end_line, "end_line")
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

    # 拦在最前面，连开都不开。.env 就躺在项目根目录，一次 read 就能让 key 随下一轮
    # 对话发出去，而且全程没有任何报错——这是这个工具最需要挡住的一件事。
    if is_sensitive(target):
        raise SensitiveFileError(f"{raw!r} 是敏感文件（可能装着密钥或凭据），不能读。")

    if target.is_dir():
        raise IsADirectoryError(f"{raw!r} 是目录不是文件。想看目录里有什么就用 list_files。")
    if not target.is_file():
        raise ValueError(f"{raw!r} 不是普通文件（可能是管道、设备文件之类），读不了。")

    # 大小先卡住：后面要把整个文件读进内存再解码，先截流比先撑爆强。
    ensure_within_size(target)

    # 读之前先量一次 mtime。不能在记的时候才 stat——那样拿到的是"记录那一刻"的时间戳，
    # 而内容属于更早的版本，两者一错开，守卫就会以为模型看到的是磁盘上最新的东西。
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
    # 报错让模型重读，绝不能记下"读过"。
    if target.stat().st_mtime_ns != mtime_ns:
        raise ValueError(f"{raw!r} 在读的过程中被改过了，这次拿到的内容不一定完整，请重读一次。")

    try:
        text = data.decode(encoding)
    except UnicodeDecodeError as exc:
        raise UndecodableTextError(
            f"{raw!r} 不是 {encoding} 编码（可能是 GBK 之类），没法当文本读。"
        ) from exc

    body = truncate_text(slice_lines(text, kwargs.get("start_line"), kwargs.get("end_line")))

    # 读成功了才记：路径 -> (读之前的 mtime, 整篇的长度)。
    # 长度记的是**整篇**，不是返回给模型的那一段——后面改文件的工具要拿它跟磁盘上的现状
    # 比，两边都得是"文件自己的属性"，掺进"模型看到多少"就没法比了。
    # 放在 slice_lines 后面是有意的：行号范围写反会抛错，那一次模型什么都没拿到，
    # 不该留下"读过"的痕迹。
    ctx.read_state.record(target, text, mtime_ns)

    return body


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


def _checked_path(root: Path, raw: str | Path) -> Path:
    """解析路径、确认落在 root 里面、且确实存在。是不是目录/文件由调用方各自判。

    留在这儿而不是嵌进工具，是因为它是**安全闸门**，两个工具都要走。抄成两份的话，
    以后补一条校验只会补在其中一份上——安全校验有三份，就等于只会被修好一份。
    """
    target = resolve_within(root, raw)
    if not is_within(root, target):
        raise OutOfSandboxError(
            f"路径 {str(raw)!r} 指向 {target}，超出了允许的范围 {root}。"
            f"只能访问 {root} 里面的东西。"
        )
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
