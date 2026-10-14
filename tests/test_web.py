"""tools/web.py 的单元测试：搜索和取正文各返回什么、失败怎么说。

全部离线——把 `_client()` 换成假客户端，一个请求都不发。
"""

from __future__ import annotations

import pytest

from mini_code_agent.tools import web
from mini_code_agent.tools.base import ToolContext


class FakeTavily:
    """按剧本回答，记下每次问了什么。"""

    def __init__(self, *, search: dict | None = None, extract: dict | None = None) -> None:
        self.search_response = search or {}
        self.extract_response = extract or {}
        self.searches: list[tuple] = []
        self.extracts: list[list[str]] = []

    def search(self, query, **kwargs):
        self.searches.append((query, kwargs))
        return self.search_response

    def extract(self, urls, **kwargs):
        self.extracts.append(list(urls))
        return self.extract_response


@pytest.fixture
def ctx(tmp_path):
    return ToolContext(cwd=tmp_path)


def use(monkeypatch, client: FakeTavily) -> FakeTavily:
    monkeypatch.setattr(web, "_client", lambda: client)
    return client


def a_page(**overrides) -> dict:
    """一条 extract 结果，形状照实测的来（url / title / raw_content）。"""
    item = {"url": "https://a.example/x", "title": "一篇文档", "raw_content": "正文在这儿"}
    return {"results": [{**item, **overrides}]}


class TestWebSearch:
    def test_it_numbers_the_results(self, monkeypatch, ctx):
        client = use(
            monkeypatch,
            FakeTavily(
                search={
                    "results": [
                        {"title": "第一条", "url": "https://a.example", "content": "摘要一"},
                        {"title": "第二条", "url": "https://b.example", "content": "摘要二"},
                    ]
                }
            ),
        )

        out = web.web_search({"query": "怎么怎么样"}, ctx)

        assert client.searches == [("怎么怎么样", {"max_results": web.DEFAULT_MAX_RESULTS})]
        assert "[1] 第一条" in out
        assert "[2] 第二条" in out
        assert "摘要二" in out

    def test_nothing_found_says_so(self, monkeypatch, ctx):
        use(monkeypatch, FakeTavily(search={"results": []}))

        out = web.web_search({"query": "没有的东西"}, ctx)

        assert "没有搜到" in out
        assert "没有的东西" in out

    def test_the_number_of_results_is_capped(self, monkeypatch, ctx):
        """模型要 50 条就会把上下文撑爆，所以硬性截断。"""
        client = use(monkeypatch, FakeTavily(search={"results": []}))

        web.web_search({"query": "q", "max_results": 999}, ctx)

        assert client.searches == [("q", {"max_results": web.MAX_RESULTS_LIMIT})]


class TestWebFetch:
    def test_it_returns_the_title_the_url_and_the_body(self, monkeypatch, ctx):
        client = use(monkeypatch, FakeTavily(extract=a_page()))

        out = web.web_fetch({"url": "https://a.example/x"}, ctx)

        assert client.extracts == [["https://a.example/x"]]  # 一次一个链接
        assert "# 一篇文档" in out
        assert "https://a.example/x" in out
        assert "正文在这儿" in out

    def test_the_url_is_stripped_before_it_goes_out(self, monkeypatch, ctx):
        client = use(monkeypatch, FakeTavily(extract=a_page()))

        web.web_fetch({"url": "  https://a.example/x  "}, ctx)

        assert client.extracts == [["https://a.example/x"]]

    def test_a_url_that_could_not_be_fetched_says_why(self, monkeypatch, ctx):
        """死链在 failed_results 里，不在 results 里——只看 results 会把它报成"页面是空的"，
        那就把模型指错方向了。"""
        use(
            monkeypatch,
            FakeTavily(
                extract={
                    "results": [],
                    "failed_results": [
                        {"url": "https://dead.example", "error": "Failed to fetch url"}
                    ],
                }
            ),
        )

        out = web.web_fetch({"url": "https://dead.example"}, ctx)

        assert "Failed to fetch url" in out  # 原始原因照抄给模型看
        assert "https://dead.example" in out

    def test_a_page_with_no_text_says_so(self, monkeypatch, ctx):
        use(monkeypatch, FakeTavily(extract=a_page(raw_content="   ")))

        out = web.web_fetch({"url": "https://a.example/x"}, ctx)

        assert "正文是空的" in out

    def test_an_empty_response_says_so(self, monkeypatch, ctx):
        use(monkeypatch, FakeTavily(extract={}))

        assert "没取回" in web.web_fetch({"url": "https://a.example/x"}, ctx)

    def test_a_long_page_comes_back_whole(self, monkeypatch, ctx):
        """这个工具**自己不截断**：太长的那份整篇落盘，上下文里只留预览（见 tools/spill.py）。

        截在这儿的话，落下去的也只是截过的那一份——而这一页是花钱取回来的，再要就得重新
        取一次，网页还可能已经变了。
        """
        use(monkeypatch, FakeTavily(extract=a_page(raw_content="x" * 50_000)))

        out = web.web_fetch({"url": "https://a.example/x"}, ctx)

        assert "已截断" not in out
        # 拿末尾那截来比，不数全文的 x：抬头那行 URL 里就有一个 x。
        assert out.endswith("x" * 50_000)  # 一个字都没少

    def test_a_long_page_keeps_its_title_and_link(self, monkeypatch, ctx):
        """抬头那两行得留着——不然模型认不出这坨正文是哪一页的（落盘那份也一样）。"""
        use(monkeypatch, FakeTavily(extract=a_page(raw_content="x" * 50_000)))

        out = web.web_fetch({"url": "https://a.example/x"}, ctx)

        assert out.startswith("# 一篇文档\nhttps://a.example/x")


def test_a_missing_key_says_what_to_do(monkeypatch):
    """缺 key 的时候要抛一句能照着做的话，而不是让 KeyError 冒出去。"""
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)

    with pytest.raises(RuntimeError, match="TAVILY_API_KEY"):
        web._client()
