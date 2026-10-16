"""上下文压缩：把旧的一段对话压成一段摘要。

**能让历史变短的动作有两个，它是"有记录"的那一个。** 另一个是折叠（`folding.py`：把旧的、
长的工具结果换成"开头一段 + 一句说明"）——那个**只在缓存大概已经冷了的时候**做，而且不落盘、
不记账，因为它重算不要钱。压缩不一样：它要花一次模型调用，所以必须记进会话文件、用户也该
看得见（见 sessions/transcript.py）。

折叠被删过一次（2026-10-15）：那时它**每一步**都在请求前缀的中间改动一处，而这个端点的缓存
以**整个请求**为单位匹配，改一处几乎全不中（18,560 → 384 命中，每步贵 6 倍）。回来的这一版
只在冷点折——那时候整段前缀本来就要重算，改历史不额外损失。详见 known-issues D5。

代价是它比折叠粗——一整段流水换成一段摘要——而且它排在折叠后面：折完可能就用不到它了
（`cli.py` 里的顺序）。所以阈值要慎重。

几条定下来的：

- **按"占了窗口多少"判**，不按字符数。`size` 是上一发真请求报的输入总量，`window` 是模型的
  上下文窗口，够到 `COMPACT_AT` 就压。字符数那条老路量错了东西——窗口吃的是 token，而且
  算上工具表和系统提示词，字符数两样都漏；而且那个阈值是拍的（注释写着 12 万、常量是 20 万，
  调过一次没改理由）。
- **只能切在真正的用户轮上**。切在 `tool_use` 和它的 `tool_result` 之间，后半段就从孤儿
  `tool_result` 开头，请求直接不合法。
- **压出来的那段当成一条合成的 user 消息**，顶在历史最前面（见 sessions/transcript.py 的
  summary_message）。连着两条 user 消息是合法的——API 会把它们并成一个回合。
- **有损，所以必须记下来**。怎么做见 sessions/transcript.py：文件里一条不删（还是完整流水），
  只追加一行 summary 标注，加载时按它剪枝。
"""

from __future__ import annotations

import json
from typing import Any

from mini_code_agent.providers.base import Provider

# 占到窗口这个比例就压。
#
# 留出来的那部分（20%）不是白留的：`_compact` 是在**新敲的那一行加进来之前**跑的，所以那一
# 行没算进 size；用户要是粘了一大坨进来，靠的就是这点余量。
COMPACT_AT = 0.8

# 没有锚点时按字符粗估 token 的系数。**取 2 是保守的**（中文约 1 字符 1 token，英文代码约
# 4 字符 1 token，混着算 2~3）。估大了触发得早、估小了触发得晚，而早压比晚压安全。
CHARS_PER_TOKEN = 2

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


def rough_tokens(messages: list[dict[str, Any]]) -> int:
    """按字符粗估这段历史有多少 token。**只在没有锚点的时候用**（见 context_size）。

    算上了块的外壳（`json.dumps` 那一圈）——那部分也是要发的。
    """
    chars = sum(len(json.dumps(message, ensure_ascii=False)) for message in messages)
    return chars // CHARS_PER_TOKEN


def context_size(messages: list[dict[str, Any]], *, tokens: int | None, covered: int = 0) -> int:
    """这段历史现在有多大（token）。够不够阈值就看它。

    **有锚点就用锚点**：`tokens` 是上一发真请求报的输入总量（三个输入字段相加），`covered`
    是**那一发请求里装了多少条 messages**。两个必须来自**同一发**，而且 `covered` 不能拿
    `AgentResult.messages` 的长度去顶——那份比发出去的长（响应拿到之后才 append 进去的）。

    锚点自带工具表和系统提示词，那两块字符估算够不着，所以有锚点时不用再补，只补锚点之后
    新加的那几条。

    **没有锚点**（`tokens` 是 None：会话第一轮、`--resume` 上来还没发过请求、或者刚压完）
    就整段粗估。
    """
    if tokens is None:
        return rough_tokens(messages)
    return tokens + rough_tokens(messages[covered:])


def cut_point(
    messages: list[dict[str, Any]],
    *,
    size: int,
    window: int,
    keep_turns: int = KEEP_RECENT_TURNS,
    force: bool = False,
) -> int | None:
    """该压就返回切点（`messages[:cut]` 压掉，`messages[cut:]` 留着），否则 None。

    `size` 是这段历史现在有多大（token，见 context_size），`window` 是模型的上下文窗口；
    够到 `window * COMPACT_AT` 才压。

    **`size` 和 `window` 都没有默认值，是有意的**：默认值不管取哪一侧都会静默走错——取 0 是
    "永远不压"，取个大数是"每轮都压"，两样都不报错。必填的话忘了传直接 TypeError，那比静默
    走错强。（`window <= 0` 也判不压：`window_for` 永远给正数，到不了这儿，留着是把"阈值这
    一侧偏保守"写死。）

    `force=True` 跳过大小那一关（`/compact` 用它）：用户点名要压，就不看多大了。两个大小参数
    照旧要传（调用方手边本来就有），只是这会儿用不上。

    返回 None 有两种情况：还没大到该压（force 时不看这条），或者轮数不够、没有安全的切点
    （只有真正的用户轮上才下得了刀，见 _safe_cut）。调用方不用区分。
    """
    if not force and (window <= 0 or size <= window * COMPACT_AT):
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
