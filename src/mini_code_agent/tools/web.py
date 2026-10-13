"""联网工具：搜索 + 取网页正文，都基于 Tavily。

需要 .env 里配好 TAVILY_API_KEY，否则调用时会抛一句能照着做的错误。

**分成两个工具是有意的**：搜索回答的是"有哪些页面"，取正文回答的是"这一页写了什么"。
合成一个的话只有两种下场——要么每条结果都拖一整篇正文进来（九成用不上，还把上下文占掉），
要么只给摘要（要照着做的步骤、要抄的代码，恰恰都在正文里）。拆开之后读多深由模型自己定。
"""

from __future__ import annotations

import os
from typing import Any

from dotenv import load_dotenv
from tavily import TavilyClient

from mini_code_agent.tools.base import ToolContext
from mini_code_agent.tools.safety import truncate_text

load_dotenv()

# 结果条数上限。模型要 50 条就会把上下文撑爆，这里硬性截断。
MAX_RESULTS_LIMIT = 10
DEFAULT_MAX_RESULTS = 5

WEB_SEARCH_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "query": {"type": "string", "description": "要搜什么，用一句自然语言描述"},
        "max_results": {
            "type": "integer",
            "description": f"返回几条结果，默认 {DEFAULT_MAX_RESULTS}，最多 {MAX_RESULTS_LIMIT} 条",
        },
    },
    "required": ["query"],
}

WEB_FETCH_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "url": {
            "type": "string",
            "description": "要打开的网页地址（http/https），一般是 web_search 返回的链接之一",
        },
    },
    "required": ["url"],
}


def _client() -> TavilyClient:
    """按需建客户端。缺 key 时抛一句能照着做的话，而不是让 KeyError 冒出去。"""
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        raise RuntimeError(
            "缺少 TAVILY_API_KEY，没法联网搜索。"
            "请在项目根目录的 .env 里配上 TAVILY_API_KEY 再试。"
        )
    return TavilyClient(api_key=api_key)


def web_search(kwargs: dict[str, Any], ctx: ToolContext) -> str:
    """联网搜一下，返回若干条结果的标题、链接和摘要。

    只取 title / url / content；Tavily 还会回 images、request_id、
    raw_content（整页正文）这些，对模型没用还占上下文，一律丢掉。
    """
    query = kwargs["query"]
    # 模型可能显式传 null。.get 的默认值只管"键不存在"，不管"值是 null"，所以用 or 兜。
    max_results = kwargs.get("max_results") or DEFAULT_MAX_RESULTS
    limit = max(1, min(int(max_results), MAX_RESULTS_LIMIT))
    response = _client().search(query, max_results=limit)

    results = response.get("results") or []
    if not results:
        return f"没有搜到和「{query}」相关的结果，换个说法再试。"

    blocks: list[str] = []
    for index, item in enumerate(results, 1):
        lines = [f"[{index}] {item.get('title') or '(无标题)'}"]
        if url := item.get("url"):
            lines.append(f"    {url}")
        if snippet := (item.get("content") or "").strip():
            lines.append(f"    {snippet}")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def web_fetch(kwargs: dict[str, Any], ctx: ToolContext) -> str:
    """打开一个网页，把正文取回来。

    取回来的是 **Tavily 服务器去取的**，不是你本机去取的。这一点值得记住：模型可能被文件或
    页面里的内容带偏、去取 localhost 或者内网地址——那一下请求发生在 Tavily 那边，够不着你的
    机器。所以这个工具不构成一条"从本机打内网"的路子。

    正文可能很长，按 truncate_text 的限额截断，并在正文里注明截了、原文多长。静默截断最糟：
    模型会以为页面就这么长，然后照着这个错的认知做事。

    一次一个链接。要读好几页就多调几次——同一条回复里可以并列好几个 tool_use，比一次塞一堆
    再让模型自己在正文里找边界强。
    """
    url = str(kwargs["url"]).strip()
    response = _client().extract([url])

    # 取不回来的链接在 failed_results 里，不在 results 里——只看 results 的话，死链会被
    # 报成"页面是空的"，那就指错方向了。
    if failed := response.get("failed_results") or []:
        reason = failed[0].get("error") or "取不回来"
        return f"没取回 {url}：{reason}。换个链接，或者用 web_search 再找找。"

    results = response.get("results") or []
    if not results:
        return f"没取回 {url} 的内容。换个链接，或者用 web_search 再找找。"

    item = results[0]
    title = item.get("title") or "(无标题)"
    body = (item.get("raw_content") or "").strip()
    if not body:
        return (
            f"取回了 {url}（{title}），但正文是空的——可能是纯图片页、要登录，"
            "或者正文全靠脚本渲染。换个来源试试。"
        )
    return f"# {title}\n{item.get('url') or url}\n\n{truncate_text(body)}"
