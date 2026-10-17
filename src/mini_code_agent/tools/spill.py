"""工具结果的落盘：太长的存成文件，上下文里只留一段预览。

为什么要有：bash 的一条长日志、web_fetch 的一整页正文，真正要用的那部分常常在末尾
（测试失败的清单就在最后）。直接截断是**有损而且不可逆**的——模型连"那儿还有东西"
都不知道，只能重跑一遍，而命令有副作用、网页下一刻会变，重跑未必等价。落盘把它留在
手边，代价只有一个文件。

**落在 `.agent/` 底下**，和长期记忆、skill 一样。那里在沙箱里是只读挂载：模型自己的
bash 改不动它。工具结果常常是它后面照着做事的依据，让它能被自己的命令改写不合适。

**而且按会话分目录**（`.agent/tool-results/<会话 id>/`）：这些文件属于那次对话，`mca
--delete` 删会话时要一起端掉。分目录之后"哪几个文件是哪次会话的"不用另记——目录名就是。

代价是文件工具读不了 `.agent`（那是敏感目录，刻意的），所以取回来要另开一道门——就是
files.py 里的 read_tool_result，和 read_memory / read_skill 一个形状。那道门开在**父目录**
上（能搜到任何一次会话的），不是"只读自己这次会话的"：`--resume` 接上来的会话得能取到它
自己历史里引用的那些文件，而那正是它自己那次会话的目录。

存下来的东西**不自动清**：删会话时会一起删，没被删的那些攒着就攒着（和改前备份一样）。
它们都在 `.gitignore` 里，也进不了模型的视野（EXCLUDED_DIRS 里有 `.agent`）。
"""

from __future__ import annotations

import re
import shutil
import uuid
from pathlib import Path

from mini_code_agent.tools.base import ToolContext
from mini_code_agent.tools.safety import PREVIEW_CHARS

# 和长期记忆、会话流水、改前备份共用 `.agent/`——它在 .gitignore 里，也被挡在模型的视野
# 之外。单独一个子目录是为了好清、好认（那几样都是"要留着"的，这一样是"这次会话的中间
# 产物"）。
TOOL_RESULTS_DIR = Path(".agent") / "tool-results"


def session_dir(cwd: Path, session_id: str) -> Path:
    """这次会话的落盘目录。CLI 把它交给 `runagent(..., spill_dir=...)`。"""
    return cwd / TOOL_RESULTS_DIR / session_id


# 允许直接当文件名用的 `tool_use_id`。正常的长这样 `toolu_01AbCd...`。
#
# 卡这一道不是洁癖：它是**服务端响应里的字符串**，不是我们自己造的名字。`../../x` 这种形状
# 拼进路径会跑出 tool-results/ 去，而写文件比读文件更糟——那边至少还有围栏兜着。
_SAFE_ID = re.compile(r"[A-Za-z0-9_-]{1,128}")


def _name_for(tool_name: str, tool_use_id: str) -> str:
    """落盘文件的主干：正常的 id 直接用；形状不对的退回老格式（工具名 + 随机）。"""
    if _SAFE_ID.fullmatch(str(tool_use_id or "")):
        return str(tool_use_id)
    return f"{tool_name}-{uuid.uuid4().hex[:12]}"


def save(
    ctx: ToolContext, tool_name: str, tool_use_id: str, text: str
) -> Path | None:
    """把一整条工具结果写下来，返回文件路径；写不成返回 None。

    写到 `ctx.spill_dir`（CLI 那边给的是这次会话的目录）；没给就落在共用的
    `.agent/tool-results/` 里——测试、以及不经 CLI 直接调 `runagent` 的那条路走的就是它。

    **名字就是那次工具调用的 id**。服务端保证一次调用一个，所以不撞名；而且同一个 id 反复
    落盘写的是同一个路径——**幂等**。折叠那边是从没被改过的会话文件重新算出来的，同一批内容
    接一次会话就会再折一遍，靠的正是这一条。

    代价是文件名不再能一眼看出是哪个工具（`toolu_...` 认不出来）。这是换来的：名字短、模型
    抄的出错面小，而工具名本来就写在配对的那条 assistant 消息里，不缺这一份。

    写不成（磁盘满、权限不对）**不该让这次调用失败**——调用方接住 None，退回"就截断"
    那条路：少给一点，总比报个错让模型以为工具坏了强。
    """
    directory = ctx.spill_dir if ctx.spill_dir is not None else ctx.cwd / TOOL_RESULTS_DIR
    target = directory / f"{_name_for(tool_name, tool_use_id)}.txt"
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
    except OSError:
        return None
    return target


def find_spilled(cwd: Path, name: str) -> str | None:
    """按名字找一个落盘文件，返回**相对 `tool-results/` 的路径**；没有就是 None。

    **要搜，不能只看一个地方**：文件按会话分在子目录里，而模型手上只有那个名字——它不知道
    也不该知道那是哪次会话的文件。搜的范围是整个 `tool-results/`（各会话的子目录 + 根下
    平铺的），也就是围栏那一层。

    根下那些平铺的老文件是这次改动之前留下的，照样找得到——引用它们的 transcript 还在。
    """
    root = cwd.resolve() / TOOL_RESULTS_DIR
    if not root.is_dir():
        return None
    matches = sorted(
        path.relative_to(root).as_posix()
        for path in (*root.glob(f"*/{name}.txt"), *root.glob(f"{name}.txt"))
        if path.is_file()
    )
    # 名字里那 12 位随机就是为这个准备的：正常只会有一条。真撞上（概率 2^-48）取排前面的
    # 那个——宁可给错一份，也不能因为撞名把这次调用变成报错。
    return matches[0] if matches else None


def count_session(cwd: Path, session_id: str) -> int:
    """这次会话落了几个文件。给人看的（`--delete` 动手之前摆出来）。"""
    directory = session_dir(cwd, session_id)
    if not directory.is_dir():
        return 0
    return sum(1 for _ in directory.glob("*.txt"))


def remove_session(cwd: Path, session_id: str) -> int:
    """把这次会话的落盘目录整个端掉，返回删掉的文件数。

    "落盘的东西属于那次会话"这条，最后就落在这儿：`--delete` 删的不只是一个 jsonl。

    删不掉（Windows 上文件被别的进程占着之类）就留着——那是磁盘卫生，不是数据完整性，
    不该让"删会话"这件事失败。文件夹里的东西，下一次 `--delete` 同一个 id 时也碰不到了。
    """
    directory = session_dir(cwd, session_id)
    count = count_session(cwd, session_id)
    if directory.is_dir():
        shutil.rmtree(directory, ignore_errors=True)
    return count


def preview_message(text: str, *, name: str, where: str) -> str:
    """落盘之后回给模型的那条：说清多长、存哪儿、怎么取，再带一段预览。

    **行数也要给**：真正要用的常在末尾，而模型拿到的只有开头那 1500 个字符。不告诉它
    一共多少行，它就没法直接翻到最后一段——`read_tool_result` 是按行号取的。

    `where` 是给人看的路径（含会话目录），`name` 是取回时该传的那个 id——两者不一样，
    所以分开传，不从一个 Path 里拆。
    """
    lines = text.count("\n") + 1
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
