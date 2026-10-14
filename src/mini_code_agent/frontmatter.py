"""`---` 包起来的那几行：拆分，以及名字的校验。

**长期记忆和 skill 共用这一份。** 两边要干的事一模一样（解析 frontmatter、挡"顺着名字跑出
目录"），各写一份迟早会漂——这个项目已经为"同一份知识写两处"付过账（见 known-issues F1）。

不引 YAML：值都是简单标量，split 一次冒号就够了（描述里带冒号也切得对，因为只切第一个）。
"""

from __future__ import annotations

import re

# 名字要能当文件名（或者目录名），也要能一眼看出是什么。挡掉路径分隔符和 `.` 就是挡掉
# "顺着名字跑出目录去"。
NAME = re.compile(r"^[a-z0-9][a-z0-9_-]*$")

# 整个文件：以 --- 起、以 --- 收，后面是正文。
_FRONTMATTER = re.compile(r"^---\s*\n(?P<fields>.*?)\n---\s*\n?(?P<body>.*)$", re.DOTALL)


def split(text: str, *, where: str, required: tuple[str, ...]) -> tuple[dict[str, str], str]:
    """拆成（frontmatter 的键值, 正文）。

    `where` 只用来报错——说清是哪个文件坏了。`required` 里列的键少一个就报错：与其让上层拿着
    一个 None 往下走，不如在这儿就说清楚缺什么。
    """
    match = _FRONTMATTER.match(text)
    if match is None:
        raise ValueError(f"{where} 少了 frontmatter：文件要以 --- 开头、用 --- 收尾。")

    fields: dict[str, str] = {}
    for line in match.group("fields").splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        fields[key.strip().lower()] = value.strip()

    missing = [key for key in required if not fields.get(key)]
    if missing:
        raise ValueError(f"{where} 的 frontmatter 缺了：{'、'.join(missing)}。")
    return fields, match.group("body").strip()


def check_name(name: object, *, problem: str, index_file: str | None = None) -> str:
    """校验一个名字，返回它。

    `problem` 是报错里那句**完整的**说法（`"不是有效的记忆名"` / `"不是有效的 skill 名"`）。
    整句让调用方写，是因为中文和拉丁词之间要不要空格不一样——在这儿拼就一定有一边拼错。

    `index_file` 是"被索引占着的那个名字"（比如 `MEMORY.md`）。只有**平铺的文件**才需要它
    ——索引和记忆挤在同一个目录里，名字撞上就是同一个文件。skill 是目录、索引是隔壁一个文件，
    不会撞，不传就行。
    """
    taken = index_file[: -len(".md")].lower() if index_file else None
    if not isinstance(name, str) or not NAME.match(name) or (taken is not None and name == taken):
        tail = f"，而且 {index_file} 是索引的名字" if index_file else ""
        raise ValueError(f"{name!r} {problem}：只能用小写字母、数字、连字符和下划线{tail}。")
    return name
