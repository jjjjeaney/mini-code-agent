"""上下文压缩：把旧的一段对话压成一段摘要。

和 trimming.py 的分工：那边折的是工具结果的**体积**（正文），这边换掉的是整段历史的**内容**
——压成一段"我们做过什么"。折完还嫌长的时候，就该压缩了：折叠只是把流水变短，模型对"读过
什么、定了什么"的理解还是从流水里现读的；压成摘要才真正把那些留下来、把流水扔掉。

几条定下来的：

- **只看字符数**，不引 tokenizer。这是个便宜的代理量，阈值留得比窗口保守——因为压缩本身还
  要再把这段历史发一次给模型（那次请求一样占地方）。
- **只能切在真正的用户轮上**。切在 `tool_use` 和它的 `tool_result` 之间，后半段就从孤儿
  `tool_result` 开头，请求直接不合法。和 trimming 里那条规矩同源。
- **压出来的那段当成一条合成的 user 消息**，顶在历史最前面（见 sessions/transcript.py 的
  summary_message）。连着两条 user 消息是合法的——API 会把它们并成一个回合。
- **有损，所以必须记下来**。怎么做见 sessions/transcript.py：文件里一条不删（还是完整流水），
  只追加一行 summary 标注，加载时按它剪枝。
"""

from __future__ import annotations

import json
from typing import Any

from mini_code_agent.providers.base import Provider

# 历史的字符数超过这个就压。粗算：中文约 1 字符 1 token，英文代码约 4 字符 1 token，混着算
# 大概 2~3 字符 1 token——12 万字符大约 4~5 万 token，离窗口还远，留出压缩那次请求的余量。
MAX_HISTORY_CHARS = 120_000

# 最近这么多轮（真正的用户轮）原样留着，只压更早的。
KEEP_RECENT_TURNS = 6

# 让模型写"可再查的事实"：路径、决定、还剩什么。细节压掉了还能重新读文件，但"改过哪个文件"
# 这种要是丢了，它就真的找不回来了。
SUMMARY_PROMPT = """\
把上面这段对话压成一份摘要，好让接着聊的人（也是一个 coding agent）能无缝接上。

写这几样，别的都别写：
- 看过、改过哪些文件（写清楚路径），各自是什么状态
- 做过哪些决定，为什么这么做
- 还没做完的，以及下一步打算做什么
- 试过但没用的做法、踩过的坑
- 用户明确交代过的事情

写成一小段，不要复述工具输出的原文，不要编造没发生过的事。"""


def total_chars(messages: list[dict[str, Any]]) -> int:
    """整段历史有多大（字符数）。算上了块的外壳——那部分也是要发的。"""
    return sum(len(json.dumps(message, ensure_ascii=False)) for message in messages)


def cut_point(
    messages: list[dict[str, Any]],
    *,
    limit: int = MAX_HISTORY_CHARS,
    keep_turns: int = KEEP_RECENT_TURNS,
) -> int | None:
    """该压就返回切点（`messages[:cut]` 压掉，`messages[cut:]` 留着），否则 None。

    返回 None 有两种情况：还没大到该压，或者轮数不够、没有安全的切点。调用方不用区分。
    """
    if total_chars(messages) <= limit:
        return None
    return _safe_cut(messages, keep_turns)


def summarize(messages: list[dict[str, Any]], provider: Provider) -> str:
    """把这段历史压成一段摘要。

    **不带 tools**：模型手里没工具可调，就只能出文本——和用满步数收尾那次一个道理。
    """
    response = provider.complete([*messages, {"role": "user", "content": SUMMARY_PROMPT}])
    return (response.text or "").strip()


def _safe_cut(messages: list[dict[str, Any]], keep_turns: int) -> int | None:
    """最近 keep_turns 个真正的用户轮留着，切在再往前那个用户轮上。

    只能在真正的用户轮上下刀：那一批 `tool_result` 的消息跟着它前面的 assistant 走，切在中间
    就会留下没有对应 `tool_use` 的结果。
    """
    turns = [i for i, message in enumerate(messages) if _is_user_turn(message)]
    if len(turns) <= keep_turns:
        return None
    return turns[-keep_turns]


def _is_user_turn(message: dict[str, Any]) -> bool:
    """真正的用户轮——不是工具结果那一批。"""
    if message.get("role") != "user":
        return False
    content = message.get("content")
    if isinstance(content, str):
        return True
    blocks = content if isinstance(content, list) else []
    return not any(block.get("type") == "tool_result" for block in blocks)
