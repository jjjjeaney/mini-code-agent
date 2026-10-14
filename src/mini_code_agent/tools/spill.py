"""工具结果的落盘：太长的存成文件，上下文里只留一段预览。

为什么要有：bash 的一条长日志、web_fetch 的一整页正文，真正要用的那部分常常在末尾
（测试失败的清单就在最后）。直接截断是**有损而且不可逆**的——模型连"那儿还有东西"
都不知道，只能重跑一遍，而命令有副作用、网页下一刻会变，重跑未必等价。落盘把它留在
手边，代价只有一个文件。

**落在 `.agent/` 底下**，和长期记忆、skill 一样。那里在沙箱里是只读挂载：模型自己的
bash 改不动它。工具结果常常是它后面照着做事的依据，让它能被自己的命令改写不合适。

代价是文件工具读不了 `.agent`（那是敏感目录，刻意的），所以取回来要另开一道门——就是
files.py 里的 read_tool_result，和 read_memory / read_skill 一个形状。

存下来的东西**不自动清**：和改前备份（`.agent/.history`）一样，攒着就攒着。它们都在
.gitignore 里，也进不了模型的视野（EXCLUDED_DIRS 里有 `.agent`）。
"""

from __future__ import annotations

import uuid
from pathlib import Path

from mini_code_agent.tools.safety import PREVIEW_CHARS

# 和长期记忆、会话流水、改前备份共用 `.agent/`——它在 .gitignore 里，也被挡在模型的视野
# 之外。单独一个子目录是为了好清、好认（那几样都是"要留着"的，这一样是"这次会话的中间
# 产物"）。
TOOL_RESULTS_DIR = Path(".agent") / "tool-results"


def save(cwd: Path, tool_name: str, text: str) -> Path | None:
    """把一整条工具结果写下来，返回文件路径；写不成返回 None。

    名字是 `<工具名>-<随机>`：随机那截保证不撞名，工具名那截是为了人一眼能看出这是什么。

    写不成（磁盘满、权限不对）**不该让这次调用失败**——调用方接住 None，退回"就截断"
    那条路：少给一点，总比报个错让模型以为工具坏了强。
    """
    target = cwd / TOOL_RESULTS_DIR / f"{tool_name}-{uuid.uuid4().hex[:12]}.txt"
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
    except OSError:
        return None
    return target


def preview_message(text: str, path: Path) -> str:
    """落盘之后回给模型的那条：说清多长、存哪儿、怎么取，再带一段预览。

    **行数也要给**：真正要用的常在末尾，而模型拿到的只有开头那 1500 个字符。不告诉它
    一共多少行，它就没法直接翻到最后一段——`read_tool_result` 是按行号取的。
    """
    name = path.stem
    lines = text.count("\n") + 1
    where = (TOOL_RESULTS_DIR / path.name).as_posix()
    return "\n".join(
        [
            f"[输出太长（{len(text):,} 字符 / {lines:,} 行），完整内容存在 {where}。",
            f'要用就用 read_tool_result 取：id 传 "{name}"，'
            f"配 start_line / end_line 可以只取一段（一共 {lines} 行）。",
            "",
            f"预览（前 {PREVIEW_CHARS} 个字符）：",
            text[:PREVIEW_CHARS] + "]",
        ]
    )
