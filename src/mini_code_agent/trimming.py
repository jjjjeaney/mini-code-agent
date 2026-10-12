"""上下文裁剪：把旧的、长的工具结果折叠掉。

为什么需要：每次请求都要把**整个** messages 重发一遍，而里面最占地方的就是工具结果——
read_file 一次能带一万两千字符，bash 每条流最多八千字节。多轮一开，这些结果只增不减，
迟早撞上上下文上限（那时 API 会明确报错，不是静默出错，但这一轮就废了）。

做法，几条都是有意的：

- **只动工具结果**。它最胖，也最容易重新拿到——需要就重读一遍文件、重跑一条命令。assistant
  说过的话和用户的话都留着。
- **只折旧的、大的**。最近几条完整留着；短的（"3 passed" 这种）也留着——折它省不下地方，
  却丢了有用的细节。
- **折的是内容，不是块**。`tool_result` 块原样保留（`tool_use_id` 一个都不能少），只把
  `content` 换成一句说明。API 要求每个 `tool_use` 都有对应的 `tool_result`，删块会让整个
  请求不合法。
- **只影响这一次请求**。传进来的那份不动，返回的是折过的新一份。写进会话文件的、以及
  `result.messages` 里的，始终是完整的。

留一句说明而不是悄悄删，是刻意的：静默截断最糟——模型会以为自己从没读过那个文件。这条
`truncate_text` 里写过，Claude Code 的 microcompact 也正因为静默被人开 issue 抱怨过。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# 最近这么多条工具结果不动。再往前的、而且够长的才折。
KEEP_RECENT_RESULTS = 8

# 短于这个长度的不折：省不下地方，却丢了细节。
MIN_FOLD_CHARS = 2000


@dataclass
class Folded:
    """一次裁剪的结果：折过的 messages，以及折了几条、腾出多少字符。"""

    messages: list[dict[str, Any]]
    count: int = 0
    chars: int = 0


def fold_old_tool_results(
    messages: list[dict[str, Any]],
    *,
    keep_recent: int = KEEP_RECENT_RESULTS,
    min_chars: int = MIN_FOLD_CHARS,
) -> Folded:
    """把旧的、长的工具结果折成一句说明。

    返回新的一份（传进来那份不动）。没有可折的就原样返回，count 为 0。
    """
    names = _tool_names(messages)
    # 先按顺序数出所有工具结果——"最近 keep_recent 条"要从后往前数。
    results = [
        (i, j, block)
        for i, message in enumerate(messages)
        for j, block in enumerate(_blocks(message))
        if block.get("type") == "tool_result"
    ]
    targets = {
        (i, j): block
        for i, j, block in results[: max(0, len(results) - keep_recent)]
        if isinstance(block.get("content"), str) and len(block["content"]) >= min_chars
    }
    if not targets:
        return Folded(messages=list(messages))

    touched = {i for i, _ in targets}
    out: list[dict[str, Any]] = []
    freed = 0
    for i, message in enumerate(messages):
        if i not in touched:
            out.append(message)
            continue
        blocks = []
        for j, block in enumerate(_blocks(message)):
            target = targets.get((i, j))
            if target is None:
                blocks.append(block)
                continue
            freed += len(target["content"])
            blocks.append(
                {
                    **target,
                    "content": _placeholder(names.get(str(target.get("tool_use_id"))), target),
                }
            )
        out.append({**message, "content": blocks})
    return Folded(messages=out, count=len(targets), chars=freed)


def _placeholder(tool: str | None, block: dict[str, Any]) -> str:
    """折成什么样。带上工具名和原长度：模型得知道丢的是什么、丢了多少。"""
    who = tool or "一个工具"
    return f"[折叠的 {who} 输出（{len(block['content'])} 字符）；需要就重读]"


def _tool_names(messages: list[dict[str, Any]]) -> dict[str, str]:
    """tool_use_id -> 工具名。工具结果本身不带名字，得从前面那条 assistant 里找。"""
    names: dict[str, str] = {}
    for message in messages:
        for block in _blocks(message):
            if block.get("type") == "tool_use" and isinstance(block.get("id"), str):
                names[block["id"]] = str(block.get("name") or "工具")
    return names


def _blocks(message: dict[str, Any]) -> list[dict[str, Any]]:
    """消息里的内容块。content 是字符串（用户那句话就是这么存的）就当没有块。"""
    content = message.get("content")
    return content if isinstance(content, list) else []
