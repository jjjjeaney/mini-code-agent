"""联网搜索工具，基于 Tavily。

需要 .env 里配好 TAVILY_API_KEY，否则调用时会抛一句能照着做的错误。
"""

from __future__ import annotations

import os
from typing import Any

from dotenv import load_dotenv
from tavily import TavilyClient

from mini_code_agent.tools.base import ToolContext

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
