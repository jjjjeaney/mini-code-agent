"""长期记忆的读写：`.agent/memory/<name>.md` 一条一个文件，索引 `MEMORY.md` 自动生成。

设计见 `docs/memory-design.md`。这里只落实几件已经定下来的事：

- 每条记忆一个文件，`frontmatter + 正文`。frontmatter 是**手工可读的几行**，不引 YAML——
  值都是简单标量，split 一次冒号就够了（描述里带冒号也切得对，因为只切第一个）。
- **索引不手工维护**：每次写完重扫一遍目录重排。增量改会漂，重扫不会，而且这点量很便宜。
  排序是固定的（按名字），所以只动到变了的那几行——这对 system prompt 的缓存前缀很重要。
- **提示词里那份索引是现算的**（`index_text()` 每次扫文件），磁盘上那个 `MEMORY.md` 只是给
  人翻的副本——所以它旧了也不影响模型，倒是反过来：手工加一个文件、还没重建索引，下次
  拼提示词就已经能看见它了。
- **索引只放"名字 + 一行描述"**。它是要进 system prompt 的，正文一律让模型按需取。
- 手工改坏的文件在重建索引时**跳过**，不该让一份索引整体读不出来。

只做读写。"谁让写""写之前做什么检查"在别处（见设计文档第四、五节）。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path

# 和会话落盘、改前备份共用 .agent/：它在 .gitignore 里，也读不到（EXCLUDED_DIRS +
# SENSITIVE_DIRS）——记忆进模型是靠 system prompt 那一份，不是靠文件工具去读。
MEMORY_DIR = Path(".agent") / "memory"
INDEX_NAME = "MEMORY.md"

# 三类，按**内容**分。为什么这么分见设计文档第六节。
TYPES = ("user", "project", "reference")

# 谁写的。这不是分类，是置信度——所以是个字段，不是第四类。
SOURCES = ("user", "agent")

# 名字要能当文件名，也要能一眼看出是什么。挡掉路径分隔符和 . 就是挡掉"顺着名字跑出目录"。
_NAME = re.compile(r"^[a-z0-9][a-z0-9_-]*$")

# 索引那个文件占掉的名字。名字一律小写（上面的正则），所以拿小写比就够——而 Windows 的文件名
# 不分大小写，小写的 memory 和 MEMORY.md 是同一个文件。
_INDEX_STEM = INDEX_NAME[: -len(".md")].lower()

# 整个文件：以 --- 起、以 --- 收，后面是正文。
_FRONTMATTER = re.compile(r"^---\s*\n(?P<fields>.*?)\n---\s*\n?(?P<body>.*)$", re.DOTALL)


@dataclass
class Memory:
    name: str
    description: str
    type: str
    body: str
    source: str = "user"
    created: str = ""

    def __post_init__(self) -> None:
        if not self.created:
            self.created = date.today().isoformat()


def memory_dir(cwd: Path) -> Path:
    return cwd / MEMORY_DIR


def names(cwd: Path) -> list[str]:
    """现有的记忆名，排好序——顺序稳，索引才不会今天一个样明天一个样。"""
    directory = memory_dir(cwd)
    if not directory.is_dir():
        return []
    return sorted(path.stem for path in directory.glob("*.md") if path.name != INDEX_NAME)


def read(cwd: Path, name: str) -> Memory:
    """按名字取一条。名字先校验：不校验的话 `../` 这种能顺着跑出目录去。"""
    checked = _checked_name(name)
    path = memory_dir(cwd) / f"{checked}.md"
    if not path.is_file():
        existing = names(cwd)
        hint = f"现有的是：{'、'.join(existing)}。" if existing else "现在一条都没有。"
        raise FileNotFoundError(f"没有叫 {name!r} 的记忆。{hint}")
    return parse(path.read_text(encoding="utf-8"), where=str(path))


def write(cwd: Path, memory: Memory) -> Path:
    """写一条（覆盖同名的），然后重建索引。"""
    _validate(memory)
    directory = memory_dir(cwd)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{memory.name}.md"
    path.write_text(render(memory), encoding="utf-8")
    rebuild_index(cwd)
    return path


def rebuild_index(cwd: Path) -> None:
    """重扫目录、重写索引。没有记忆时也写（内容为空）——目录自己说明自己。"""
    directory = memory_dir(cwd)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / INDEX_NAME).write_text(index_text(cwd), encoding="utf-8")


def index_text(cwd: Path) -> str:
    """索引正文。一条记忆都没有（或者全读不出来）就返回空串——调用方据此决定要不要那一节。"""
    lines: list[str] = []
    for name in names(cwd):
        try:
            memory = read(cwd, name)
        except (OSError, ValueError):
            # 手工改坏的文件跳过：一份索引不该被一个坏文件整体拖垮。
            continue
        lines.append(f"- [{memory.name}]({memory.name}.md) — {memory.description}")
    return "\n".join(lines)


def render(memory: Memory) -> str:
    """落盘的样子。手工可读是刻意的——这是给人看、也给人改的文件。"""
    return (
        "---\n"
        f"name: {memory.name}\n"
        f"description: {memory.description}\n"
        f"type: {memory.type}\n"
        f"source: {memory.source}\n"
        f"created: {memory.created}\n"
        "---\n\n"
        f"{memory.body.strip()}\n"
    )


def parse(text: str, *, where: str) -> Memory:
    """读回来。`where` 只用在报错里——说清是哪个文件坏了。"""
    match = _FRONTMATTER.match(text)
    if match is None:
        raise ValueError(f"{where} 少了 frontmatter：文件要以 --- 开头、用 --- 收尾。")

    fields: dict[str, str] = {}
    for line in match.group("fields").splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        fields[key.strip().lower()] = value.strip()

    missing = [key for key in ("name", "description", "type") if not fields.get(key)]
    if missing:
        raise ValueError(f"{where} 的 frontmatter 缺了：{'、'.join(missing)}。")

    memory = Memory(
        name=fields["name"],
        description=fields["description"],
        type=fields["type"],
        body=match.group("body").strip(),
        source=fields.get("source") or "user",
        created=fields.get("created") or "",
    )
    _validate(memory, where=where)
    return memory


def _checked_name(name: str) -> str:
    if not isinstance(name, str) or not _NAME.match(name) or name == _INDEX_STEM:
        raise ValueError(
            f"{name!r} 不是有效的记忆名：只能用小写字母、数字、连字符和下划线，"
            f"而且 {INDEX_NAME} 是索引的名字。"
        )
    return name


def _validate(memory: Memory, *, where: str = "") -> None:
    _checked_name(memory.name)
    if memory.type not in TYPES:
        raise ValueError(
            f"{where} 的 type 得是 {'、'.join(TYPES)} 之一，收到的是 {memory.type!r}。"
        )
    if memory.source not in SOURCES:
        raise ValueError(
            f"{where} 的 source 得是 {'、'.join(SOURCES)} 之一，收到的是 {memory.source!r}。"
        )
    if not memory.description.strip():
        raise ValueError(f"{where} 少了 description——索引那一行就靠它。")
    if not memory.body.strip():
        raise ValueError(f"{where} 的正文是空的。")
