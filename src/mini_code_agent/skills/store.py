"""技能的读写：`.agent/skill/<名字>/SKILL.md` 一份一个目录，索引 `SKILLS.md` 自动生成。

和长期记忆的分界是**内容**：记忆是**事实**（拿来判断——"这个项目测试用 uv"），skill 是
**步骤**（拿来照做——"发版前按这份清单走一遍"）。形状也跟着不一样：记忆是平铺的文件，
skill 是目录，能带附件。设计见 `docs/skill-design.md`。

几条定下来的：

- **一份 skill 一个目录**，正文在 `SKILL.md`（frontmatter + 正文）。别的文件全是**附件**
  （脚本、模板、参考文档），要用 `read_skill_file` 单独取——正文里塞不下附件，塞了就是一份
  skill 吃掉一大块上下文。
- **名字就是目录名**，`SKILL.md` 的 frontmatter 只管 `description`。不放 `name` 是有意的：
  两处都写迟早会不一致，而目录名是躲不掉的。
- **只有人手写**，所以这里没有 `write()`。agent 只读——记忆至少是"事实"，skill 是"步骤"，
  不该让模型自己给自己立规矩。
- **索引不手工维护**，跟记忆一样：`sync_skill_index` 重扫一遍对齐。手工加/删一份 skill 会让它漂，
  所以启动时对一次，只在真不一样时才写。
- **提示词里那份索引是现算的**（`index_text()` 每次扫目录），磁盘上那个 `SKILLS.md` 只是给人
  翻的副本——所以手工加一份、还没重建索引，下次拼提示词就已经能看见它了。
- 手工改坏的（没 frontmatter、正文空）在重建索引时**跳过**，不该让一份索引被一个坏文件拖垮。

这里**不 import tools**：`tools/skill.py` 要用这个模块，反过来引就成了环。所以路径闸门、
截断这些都在工具那一层做。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from mini_code_agent import frontmatter

# 和会话落盘、改前备份、长期记忆共用 .agent/：它在 .gitignore 里，文件工具也够不着
# （EXCLUDED_DIRS + SENSITIVE_DIRS）——skill 进模型是靠 system prompt 那份索引 + 这个模块
# 开的门，不是靠文件工具去读。
SKILL_DIR = Path(".agent") / "skill"

# 正文那个文件。**每个 skill 目录里必须有的就它一个。**
SKILL_FILE = "SKILL.md"

INDEX_NAME = "SKILLS.md"

# 列附件时最多列几个。剩下的只报个数——附件多起来的时候，把名字全铺进上下文没意义。
MAX_LISTED_ATTACHMENTS = 20


@dataclass
class Skill:
    name: str
    description: str
    body: str
    directory: Path


def skills_dir(cwd: Path) -> Path:
    return cwd / SKILL_DIR


def names(cwd: Path) -> list[str]:
    """现有的 skill 名（就是目录名），排好序——顺序稳，索引才不会今天一个样明天一个样。

    **只算真的有 `SKILL.md` 的那些**：`.agent/skill/` 下面放个别的东西不算一份 skill，
    也不该因为目录里多一个杂物就报错。
    """
    directory = skills_dir(cwd)
    if not directory.is_dir():
        return []
    return sorted(
        path.name
        for path in directory.iterdir()
        if path.is_dir() and (path / SKILL_FILE).is_file()
    )


def read(cwd: Path, name: str) -> Skill:
    """按名字取一份。名字先校验：不校验的话 `../` 这种能顺着跑出目录去。"""
    checked = frontmatter.check_name(name, problem="不是有效的 skill 名")
    directory = skills_dir(cwd) / checked
    path = directory / SKILL_FILE
    if not path.is_file():
        existing = names(cwd)
        hint = f"现有的是：{'、'.join(existing)}。" if existing else "现在一份都没有。"
        raise FileNotFoundError(f"没有叫 {name!r} 的 skill。{hint}")

    fields, body = frontmatter.split(
        path.read_text(encoding="utf-8"), where=str(path), required=("description",)
    )
    if not body.strip():
        raise ValueError(f"{path} 的正文是空的——那才是模型要照着做的东西。")
    return Skill(name=checked, description=fields["description"], body=body, directory=directory)


def attachments(skill: Skill) -> tuple[list[str], int]:
    """这份 skill 的附件：除 `SKILL.md` 之外的文件，路径相对 skill 目录，排好序。

    返回（列出来的, 没列出来的个数）。**子目录里的也算**——附件按原样放在目录里就行，
    不必摊平。
    """
    files = sorted(
        path.relative_to(skill.directory).as_posix()
        for path in skill.directory.rglob("*")
        if path.is_file() and path != skill.directory / SKILL_FILE
    )
    return files[:MAX_LISTED_ATTACHMENTS], max(0, len(files) - MAX_LISTED_ATTACHMENTS)


def index_text(cwd: Path) -> str:
    """索引正文。一份 skill 都没有（或者全读不出来）就返回空串——调用方据此决定要不要那一节。

    每条**只有名字和一句描述**：这段要进 system prompt，而正文得让模型按需取。
    """
    lines: list[str] = []
    for name in names(cwd):
        try:
            skill = read(cwd, name)
        except (OSError, ValueError):
            # 手工改坏的跳过：一份索引不该被一个坏文件整体拖垮。
            continue
        lines.append(f"- [{skill.name}]({skill.name}/{SKILL_FILE}) — {skill.description}")
    return "\n".join(lines)


def sync_skill_index(cwd: Path) -> bool:
    """让磁盘上那份索引跟目录里实际有什么对齐；写了返回 True。

    **只在真的不一样时才写**：手工加过、删过之后它会漂（模型那边不受影响——它看的是现算的
    `index_text()`），启动时对一次就够，稳态下一个字节都不动。

    目录压根不存在就直接返回：不因为跑一次 agent，就在别人项目里造一个 `.agent/skill/`。
    """
    directory = skills_dir(cwd)
    if not directory.is_dir():
        return False
    text = index_text(cwd)
    index = directory / INDEX_NAME
    if index.is_file() and index.read_text(encoding="utf-8") == text:
        return False
    index.write_text(text, encoding="utf-8")
    return True
