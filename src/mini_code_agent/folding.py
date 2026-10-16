"""冷的时候把旧的工具结果折掉。

**为什么只在冷的时候折。** 这个端点的缓存以**整个请求**为单元匹配，中间改一处就几乎全不中
（实测：载荷砍一半，命中从 18,560 掉到 384，那一发贵 6 倍——见 known-issues D5）。但缓存
**已经冷了**的时候，那一发本来就要按原价重算整段前缀，这时候改历史不额外损失任何东西。

设上下文 C、折完剩 αC：

- **冷的时候折**：那一发从 `C × 原价` 变成 `αC × 原价`，直接省 `(1−α)C`；之后每一发再省
  `(1−α)C / 50`（命中价是原价的 1/50）。**哪怕只有这一发，也是赚的。**
- **热的时候折**：那一发本来只要 `C/50`，现在要 `αC`——亏，而且要之后约 50 发才回本。

所以这不是"要不要折"，是"**什么时候折**"。冷不冷由 `is_stale` 判断。

**折的是内存里的视图，不落盘。** 会话文件保持"实际发生过什么"的完整记录。盘上和内存因此会
分叉，但那个分叉**下次到冷点会被重新折平**，所以没关系。压缩必须往盘上记一笔（因为它是一次
花钱的模型调用，重算要再花一次钱），折叠重算不要钱，就不需要。
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

# 保留最近这么多条工具结果不折。**按块数，不按消息数**——同一条 assistant 回复里的多个
# tool_use 会合成一条 user 消息回填（见 core/agent.py），一条消息里可能并排放着好几个。
KEEP_RECENT_RESULTS = 5

# 折完保留开头多少个字符。
#
# **开头那一段是有用的**：落过盘的结果，"存在哪儿"和 `read_tool_result` 要的 id 就写在前几行
# （见 tools/spill.py 的 preview_message），留着这一段它还取得回来；别的结果也能看出这是什么
# （`read_file` 的正文、`grep` 的 `文件:行号:内容`）。
HEAD_CHARS = 200

# 原内容短于这个数就不折。折完那条说明自己就有两百多字符，原文比它还短的话，折了反而变长。
# 它顺带把"折过的又折一遍"挡住——折过的结果正好落在这条线以下，所以折叠是幂等的。
#
# **必须大于 HEAD_CHARS 加上那句说明的长度**，否则幂等那条就不成立了。
MIN_FOLD_CHARS = 400

# 会话文件多久没被写过，就认为缓存冷了。
#
# **这是个猜。** 要把它变成有依据的数，得量这个端点的缓存 TTL：造一段内容、隔 T 分钟发一次、
# 看 `cache_read_input_tokens` 什么时候掉到 0。
#
# 猜错的代价不对称、但方向是安全的：猜短了会在"其实还热"的时候多折一次，那一次多付 αC 的
# 原价。而**只要"真冷"的概率过半，折就是赚的**（α = 0.5 时，折对了省 0.5C、折错了亏 0.5C）
# ——所以宁可取小一点，宁可多折一次。
FOLD_AFTER_SECONDS = 10 * 60


def is_stale(
    path: Path, *, now: float | None = None, after: float = FOLD_AFTER_SECONDS
) -> bool:
    """这个会话文件是不是已经久到"缓存该冷了"。

    **为什么能用文件的 mtime**：每轮结束都会 `Session.append` 写一次，所以它的 mtime 就是
    "上一轮什么时候结束的"。这一条同时覆盖两种情形——`--resume` 接上来的会话（文件是上次
    关掉时写的）、以及开着会话走开一会儿再回来。而且它比"接会话就折"准：关掉之后马上重开，
    缓存其实还是热的，那时候折反而亏，mtime 规则不会在那儿折。

    它不需要新状态、不碰文件格式、跨进程也对。

    文件不存在（还没说过话）返回 False——没什么可折的。`stat` 失败也一样：折不了不是大事，
    不该因此把这一轮弄崩。
    """
    try:
        last = path.stat().st_mtime
    except OSError:
        return False
    return (time.time() if now is None else now) - last >= after


def fold_stale(
    messages: list[dict[str, Any]], *, keep: int = KEEP_RECENT_RESULTS
) -> list[dict[str, Any]]:
    """把最近 `keep` 条以外的工具结果折成"开头一段 + 一句说明"。

    **没折到就返回传进来那一份本身**（`is` 相等）——调用方靠这个判断"这一轮到底折没折"，
    和 `cli._compact` 用的是同一个信号。

    **只换 `tool_result` 块里的 `content`**，消息条数、块的条数、`tool_use_id`、`is_error`
    一个都不动。这一条不是洁癖：

    - **消息条数一变，`Session.record_summary` 的文件下标换算就错**——它假设内存里的下标和
      文件里的消息行一一对应（见 sessions/transcript.py 里 `covered_from` 那段）。
    - **`tool_use_id` 是请求合法性的底线**：配对错了，整个请求都不合法。

    **返回新的一份，不改传进来那份**——和 `runagent` 的契约一致。消息 dict 和它里面的块列表
    在别处是**共享引用**（`runagent` 只复制最外层那个列表），就地改会连调用方手里那份一起
    改掉。
    """
    stale = _stale_positions(messages, keep)
    if not stale:
        return messages

    replacements: dict[int, dict[int, str]] = {}
    for message_index, block_index in stale:
        folded = _fold(messages[message_index]["content"][block_index])
        if folded is not None:
            replacements.setdefault(message_index, {})[block_index] = folded
    if not replacements:
        return messages

    result = list(messages)
    for message_index, new_contents in replacements.items():
        blocks = []
        for index, block in enumerate(messages[message_index]["content"]):
            if index in new_contents:
                block = {**block, "content": new_contents[index]}
            blocks.append(block)
        result[message_index] = {**messages[message_index], "content": blocks}
    return result


def _stale_positions(
    messages: list[dict[str, Any]], keep: int
) -> list[tuple[int, int]]:
    """要折的那几条工具结果在哪儿：`[(消息下标, 块下标), ...]`。

    留最近 `keep` 个**块**（不是消息——一条消息里可能并排放着好几个），其余都是候选。
    """
    positions: list[tuple[int, int]] = []
    for message_index, message in enumerate(messages):
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for block_index, block in enumerate(content):
            if isinstance(block, dict) and block.get("type") == "tool_result":
                positions.append((message_index, block_index))
    if len(positions) <= keep:
        return []
    return positions[:-keep]


def _fold(block: dict[str, Any]) -> str | None:
    """折一条结果，返回折完的文本；不值得折就返回 None。

    **那句"不是工具出错"不能省**：模型看到结果突然变短，第一反应是"工具坏了、输出被截了"，
    然后白跑一次去确认。说清这是历史整理，它才知道该不该重新要。
    """
    content = block.get("content")
    if not isinstance(content, str) or len(content) <= MIN_FOLD_CHARS:
        return None
    dropped = len(content) - HEAD_CHARS
    return (
        f"{content[:HEAD_CHARS]}\n\n"
        f"[后面的 {dropped:,} 个字符已清除：这是历史整理，不是工具出错]"
    )
