"""冷的时候把旧的工具结果折掉。

**为什么只在冷的时候折。** 这个端点的缓存以**整个请求**为单元匹配，中间改一处就几乎全不中
（实测：载荷砍一半，命中从 18,560 掉到 384，那一发贵 6 倍——见 known-issues D5）。但缓存
**已经冷了**的时候，那一发本来就要按原价重算整段前缀，这时候改历史不额外损失任何东西。

设上下文 C、折完剩 αC：

- **冷的时候折**：那一发从 `C × 原价` 变成 `αC × 原价`，直接省 `(1−α)C`；之后每一发再省
  `(1−α)C / 50`（命中价是原价的 1/50）。**哪怕只有这一发，也是赚的。**
- **热的时候折**：那一发本来只要 `C/50`，现在要 `αC`——亏，而且要之后约 50 发才回本。

所以这不是"要不要折"，是"**什么时候折**"。冷不冷由 `is_stale` 判断。

**折的是内存里的视图，会话文件不动。** 盘上和内存因此会分叉，但那个分叉**下次到冷点会被
重新折平**，所以没关系。压缩必须往会话文件里记一笔（因为它是一次花钱的模型调用，重算要再
花一次钱），折叠重算不要钱，就不需要。

**但"重跑拿不回来"的那几类要先落盘**（`persist_tools`，目前是 `bash` / `web_fetch` /
`web_search`）：命令有副作用、网页会变、搜索要花钱。折之前把完整内容写进
`.agent/tool-results/<会话 id>/<tool_use_id>.txt`，上下文里只留开头一段加上路径和 id。
写不成就不折这一条——宁可让它继续占着上下文，也不能给模型一个指向空处的引用。
`read_file` / `grep` / `glob` / `list_files` 不走这条路：重跑免费，而且拿到的还是当前内容。

**不是所有工具的结果都能折**：只有标了 `Tool.foldable` 的才进候选（见 tools/base.py）。子
agent 交回来的结论就是"方向"本身、项目概览是"我现在在哪儿"的参照、模型特意去取回来的记忆和
技能——那些折了就真丢了（而且重跑一遍可能是几分钟和几毛钱）。

**没有手动入口**，这是有意的：它划不划算完全取决于"缓存冷不冷"，而那件事用户看不见（我们
自己也只能靠文件 mtime 猜）。让人敲一条命令去赌，赌错了就是多付一次全价——真嫌上下文长，
该敲的是 `/compact`。
"""

from __future__ import annotations

import time
from collections.abc import Callable, Collection
from pathlib import Path
from typing import Any

from mini_code_agent.tools import spill

# 折叠前"把正文留一份到盘上"的那个动作：(工具名, tool_use_id, 正文) → (给人看的路径,
# 落盘后的**真实文件名**)，存不下来就是 None。
#
# **必须回传真实文件名**：形状不合格的 id 会被换成 `<工具名>-<摘要>`，占位符里写的 id 得是
# 模型取的时候真能用的那个。
PersistText = Callable[[str, str, str], tuple[str, str] | None]

# 保留最近这么多条工具结果不折。**按块数，不按消息数**——同一条 assistant 回复里的多个
# tool_use 会合成一条 user 消息回填（见 core/agent.py），一条消息里可能并排放着好几个。
KEEP_RECENT_RESULTS = 5

# 折完保留开头多少个字符。
#
# **开头那一段是有用的**：落过盘的结果，"存在哪儿"和 `read_tool_result` 要的 id 就写在前几行
# （见 tools/spill.py 的 preview_message），留着这一段它还取得回来；别的结果也能看出这是什么
# （`read_file` 的正文、`grep` 的 `文件:行号:内容`）。
HEAD_CHARS = 200

# 原内容短于这个数就不折：折完那条说明自己就有两百多字符，原文比它还短的话，折了反而变长。
#
# **折完的结果必须落在这条线以下**（`_fold` 里有一道显式的闸）。这是幂等的保证：折过的结果
# 下一次会被开头那道早退挡住。路径和 id 让说明变长了，所以不能再假设"说明一定短"。
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


def fold_if_stale(
    messages: list[dict[str, Any]],
    *,
    foldable: Collection[str],
    path: Path,
    keep: int = KEEP_RECENT_RESULTS,
    now: float | None = None,
    persist: PersistText | None = None,
    persist_tools: Collection[str] = frozenset(),
) -> list[dict[str, Any]]:
    """缓存大概已经冷了（这个会话文件久没被写过）就折一次；不然原样返回（`is` 相等）。

    "冷不冷"和"怎么折"合成一次调用，是为了让调用方那边只剩一行——**判据不该由调用方自己
    拼**：两个常量（阈值和折叠规则）分在两处，改一个忘一个就是悄悄走错，而这条路上走错的
    表现是"在缓存还热的时候折"，也就是钱。

    没有折到就返回传进来那一份本身，和 `fold_stale` 一个信号。
    """
    if not is_stale(path, now=now):
        return messages
    return fold_stale(
        messages,
        foldable=foldable,
        keep=keep,
        persist=persist,
        persist_tools=persist_tools,
    )


def fold_stale(
    messages: list[dict[str, Any]],
    *,
    foldable: Collection[str],
    keep: int = KEEP_RECENT_RESULTS,
    persist: PersistText | None = None,
    persist_tools: Collection[str] = frozenset(),
) -> list[dict[str, Any]]:
    """把最近 `keep` 条以外的工具结果折成"开头一段 + 一句说明"。

    **只有 `foldable` 里那些工具的结果进候选**——子 agent 的结论、项目概览、模型特意去取的
    记忆和技能，一个都不动。而且 `keep` 数的也是**候选**里的，不是"最近几条工具结果"：不可
    折的那些既不占名额、也不会被动。

    `foldable` **没有默认值是有意的**：默认取空集等于"什么都不折"（功能悄悄没了），默认取
    "全都折"更糟（子 agent 的结论会被折掉）。忘了传直接 TypeError，比静默走错强。

    `persist_tools` 里的工具**折之前先经 `persist` 落一份盘**（结果重跑未必等价）。两个都
    给了才有意义——只给名单不给回调是**配置错误**，会直接报出来，不能悄悄退回"重跑"措辞。

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
    if persist_tools and persist is None:
        raise ValueError(
            "persist_tools 非空时必须给 persist——否则那些结果会被当成能重跑的折掉"
        )
    stale = _stale_positions(messages, foldable, keep)
    if not stale:
        return messages

    replacements: dict[int, dict[int, str]] = {}
    for message_index, block_index, name in stale:
        folded = _fold(
            messages[message_index]["content"][block_index],
            name,
            persist=persist,
            persist_tools=persist_tools,
        )
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


def _tool_names(messages: list[dict[str, Any]]) -> dict[str, str]:
    """`tool_use_id` → 工具名。

    **`tool_result` 块自己不带工具名**——它只有 `tool_use_id`，名字在配对的那条 assistant
    消息的 `tool_use` 块里。所以要认出一条结果属于哪个工具，得先把整份历史扫一遍。
    """
    names: dict[str, str] = {}
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if isinstance(block, dict) and block.get("type") == "tool_use":
                names[str(block.get("id"))] = str(block.get("name"))
    return names


def _stale_positions(
    messages: list[dict[str, Any]], foldable: Collection[str], keep: int
) -> list[tuple[int, int, str]]:
    """要折的那几条在哪儿、各属于哪个工具：`[(消息下标, 块下标, 工具名), ...]`。

    留最近 `keep` 个**候选**（不是"最近几条工具结果"，也不是最近几条消息——一条 assistant
    回复里的多个 tool_use 会合成一条 user 消息回填，一条消息里可能并排放着好几个），其余的
    都是要折的。

    认不出工具名的（理论上不该有：配对错了请求本身就不合法）**当不可折处理**——宁可留着。
    """
    names = _tool_names(messages)
    candidates: list[tuple[int, int, str]] = []
    for message_index, message in enumerate(messages):
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for block_index, block in enumerate(content):
            if not isinstance(block, dict) or block.get("type") != "tool_result":
                continue
            name = names.get(str(block.get("tool_use_id")))
            if name in foldable:
                candidates.append((message_index, block_index, name))
    if len(candidates) <= keep:
        return []
    return candidates[:-keep]


def _fold(
    block: dict[str, Any],
    name: str,
    *,
    persist: PersistText | None,
    persist_tools: Collection[str],
) -> str | None:
    """折一条结果，返回折完的文本；不值得折（或者不该折）就返回 None。

    **工具名要写进去**：`tool_result` 自己不带它，模型看到一段没头没尾的说明，判不出被清掉的
    是什么、也就判不出该不该重新要——那正是"丢了方向"的起点。

    **"不是工具出错"和"怎么拿回来"两半也不能省**：模型看到结果突然变短，第一反应是"工具坏
    了、输出被截了"，然后白跑一次去确认。

    **拿不回来的那几类先落盘**（`persist_tools`）。写不成就不折这一条——折了就真丢了，宁可
    让它继续占着上下文，也不能给模型一个指向空处的引用。已经落过盘的（正文是那条预览）不再
    写第二次：真正的原文就在盘上。
    """
    content = block.get("content")
    if not isinstance(content, str) or len(content) <= MIN_FOLD_CHARS:
        return None

    dropped = len(content) - HEAD_CHARS
    tool_use_id = str(block.get("tool_use_id") or "")
    note = f"\n\n[{name} 的结果：后面 {dropped:,} 个字符已清除（历史整理，不是工具出错）。"

    if name in persist_tools:
        if spill.is_preview(content):
            # 超过 MAX_TOOL_CHARS 的那条早就落过盘了，开头那段里带着取回的 id。
            note += "原文已经落过盘，取法在保留的开头那段里。]"
        else:
            saved = persist(name, tool_use_id, content) if persist is not None else None
            if saved is None:
                return None
            where, saved_name = saved
            note += (
                f"完整内容存在 {where}。"
                f'要就用 read_tool_result 取：id 传 "{saved_name}"。]'
            )
    else:
        note += "要原文可以重跑一次那个工具。]"

    folded = content[:HEAD_CHARS] + note
    # 折完必须比原文短、而且落在 MIN_FOLD_CHARS 以下。后者是**幂等的保证**：折过的结果下一次
    # 会被开头那道早退挡住。路径和 id 让说明变长了，不能再假设"说明一定短"。
    if len(folded) >= len(content) or len(folded) > MIN_FOLD_CHARS:
        return None
    return folded
