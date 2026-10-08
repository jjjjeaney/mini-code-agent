"""Tool / ToolResult / ToolRegistry 的单元测试，全部本地执行，不发网络请求。"""

from __future__ import annotations

import json

import pytest

from mini_code_agent.providers.base import ToolCall
from mini_code_agent.tools import Tool, ToolRegistry, ToolResult, default_tools, web
from mini_code_agent.tools.builtin import echo as builtin_echo

ADD_SCHEMA = {
    "type": "object",
    "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
    "required": ["a", "b"],
}


def add(a: int, b: int) -> int:
    return a + b


def boom() -> str:
    raise RuntimeError("工具内部炸了")


def echo(payload: dict | None = None) -> dict:
    return payload or {}


def make_add() -> Tool:
    return Tool(name="add", description="把两个整数相加", fn=add, params=ADD_SCHEMA)


@pytest.fixture
def registry() -> ToolRegistry:
    return ToolRegistry(
        [
            make_add(),
            Tool(name="boom", description="总是抛异常", fn=boom),
            Tool(name="echo", description="原样返回参数", fn=echo),
        ]
    )


class TestTool:
    def test_to_anthropic_schema_uses_api_field_names(self):
        assert make_add().to_anthropic_schema() == {
            "name": "add",
            "description": "把两个整数相加",
            "input_schema": ADD_SCHEMA,
        }

    def test_missing_params_becomes_empty_object_schema(self):
        tool = Tool(name="now", description="返回当前时间", fn=lambda: "now")
        assert tool.to_anthropic_schema()["input_schema"] == {
            "type": "object",
            "properties": {},
            "required": [],
        }

    def test_missing_description_is_omitted_instead_of_sent_as_null(self):
        tool = Tool(name="now", fn=lambda: "now")
        assert "description" not in tool.to_anthropic_schema()

    def test_empty_name_is_rejected(self):
        with pytest.raises(ValueError, match="不能为空"):
            Tool(name="", description="x", fn=add)

    def test_non_callable_fn_is_rejected(self):
        with pytest.raises(TypeError, match="必须可调用"):
            Tool(name="add", description="x", fn="不是一个函数")


class TestToolResult:
    def test_block_keeps_string_content_as_is(self):
        result = ToolResult(id="toolu_1", name="add", result="3")
        assert result.to_anthropic_block() == {
            "type": "tool_result",
            "tool_use_id": "toolu_1",
            "content": "3",
        }

    def test_block_json_encodes_structured_content(self):
        result = ToolResult(id="toolu_2", name="echo", result={"ok": True})
        assert json.loads(result.to_anthropic_block()["content"]) == {"ok": True}

    def test_block_marks_errors(self):
        result = ToolResult(id="toolu_3", name="boom", result="炸了", is_error=True)
        assert result.to_anthropic_block()["is_error"] is True

    def test_successful_block_has_no_is_error_key(self):
        assert "is_error" not in ToolResult(id="t", name="add", result="3").to_anthropic_block()

    def test_name_is_local_metadata_and_stays_off_the_wire(self):
        assert "name" not in ToolResult(id="t", name="add", result="3").to_anthropic_block()


class TestToolRegistry:
    def test_list_preserves_registration_order(self, registry):
        assert [t.name for t in registry.list()] == ["add", "boom", "echo"]

    def test_register_returns_the_tool(self):
        tool = make_add()
        assert ToolRegistry().register(tool) is tool

    def test_duplicate_name_is_rejected(self, registry):
        with pytest.raises(ValueError, match="已经注册过"):
            registry.register(make_add())

    def test_run_calls_the_underlying_function(self, registry):
        assert registry.run("add", {"a": 2, "b": 3}) == 5

    def test_run_without_arguments(self, registry):
        assert registry.run("echo") == {}

    def test_run_unknown_tool_raises_key_error(self, registry):
        with pytest.raises(KeyError, match="nope"):
            registry.run("nope")

    def test_schemas_lists_every_tool(self, registry):
        assert [schema["name"] for schema in registry.schemas()] == ["add", "boom", "echo"]

    def test_dispatch_wraps_the_return_value_in_a_tool_result(self, registry):
        call = ToolCall(id="toolu_1", name="add", arguments={"a": 1, "b": 2})

        result = registry.dispatch(call)

        assert result == ToolResult(id="toolu_1", name="add", result=3)
        assert result.to_anthropic_block()["content"] == "3"

    def test_dispatch_turns_tool_errors_into_results(self, registry):
        result = registry.dispatch(ToolCall(id="toolu_3", name="boom", arguments={}))

        assert result.is_error is True
        assert "RuntimeError" in result.result

    def test_dispatch_unknown_tool_is_a_result_not_an_exception(self, registry):
        result = registry.dispatch(ToolCall(id="toolu_4", name="nope", arguments={}))

        assert result.is_error is True
        assert "KeyError" in result.result


class TestBuiltinEcho:
    def test_prints_the_text_and_returns_a_confirmation(self, capsys):
        result = builtin_echo("你好")

        assert capsys.readouterr().out == "你好\n"
        assert "你好" in result


class TestDefaultTools:
    def test_registers_the_builtin_tools(self):
        assert [tool.name for tool in default_tools().list()] == ["echo", "web_search"]

    def test_echo_schema_requires_text(self):
        schema = default_tools().schemas()[0]

        assert schema["name"] == "echo"
        assert schema["input_schema"]["required"] == ["text"]

    def test_each_call_returns_a_fresh_registry(self):
        assert default_tools() is not default_tools()

    def test_echo_runs_through_the_registry(self, capsys):
        default_tools().run("echo", {"text": "hi"})

        assert capsys.readouterr().out == "hi\n"


class FakeTavilyClient:
    """顶掉真的 Tavily 客户端：不发网络请求，同时记下收到的参数。"""

    def __init__(self) -> None:
        self.response: dict = {"results": []}
        self.error: Exception | None = None
        self.calls: list[dict] = []

    def search(self, query, **kwargs):
        self.calls.append({"query": query, **kwargs})
        if self.error is not None:
            raise self.error
        return self.response


@pytest.fixture
def fake_tavily(monkeypatch) -> FakeTavilyClient:
    client = FakeTavilyClient()
    monkeypatch.setattr(web, "_client", lambda: client)
    return client


class TestWebSearch:
    def test_formats_title_url_and_snippet(self, fake_tavily):
        fake_tavily.response = {
            "results": [
                {
                    "title": "Lionel Messi - Wikipedia",
                    "url": "https://en.wikipedia.org/wiki/Lionel_Messi",
                    "content": "Argentine footballer.",
                    "score": 0.95,
                    "images": ["nope.jpg"],
                }
            ]
        }

        text = web.web_search("Leo Messi")

        assert "[1] Lionel Messi - Wikipedia" in text
        assert "https://en.wikipedia.org/wiki/Lionel_Messi" in text
        assert "Argentine footballer." in text

    def test_drops_the_noise_tavily_returns(self, fake_tavily):
        fake_tavily.response = {
            "request_id": "abc",
            "response_time": 1.23,
            "images": ["nope.jpg"],
            "results": [{"title": "t", "url": "u", "content": "c"}],
        }

        text = web.web_search("x")

        assert "abc" not in text
        assert "nope.jpg" not in text

    def test_passes_query_and_max_results_to_tavily(self, fake_tavily):
        web.web_search("python 3.13", max_results=3)

        assert fake_tavily.calls == [{"query": "python 3.13", "max_results": 3}]

    def test_clamps_max_results_to_the_cap(self, fake_tavily):
        web.web_search("x", max_results=999)

        assert fake_tavily.calls[0]["max_results"] == web.MAX_RESULTS_LIMIT

    def test_says_so_when_nothing_matched(self, fake_tavily):
        assert "没有搜到" in web.web_search("绝不可能存在的词")

    def test_missing_api_key_gives_an_actionable_error(self, monkeypatch):
        monkeypatch.delenv("TAVILY_API_KEY", raising=False)

        with pytest.raises(RuntimeError, match="TAVILY_API_KEY"):
            web._client()

    def test_client_transport_error_reaches_the_model_as_a_result(self, fake_tavily):
        fake_tavily.error = RuntimeError("rate limited")

        result = default_tools().dispatch(
            ToolCall(id="t1", name="web_search", arguments={"query": "x"})
        )

        assert result.is_error is True
        assert "rate limited" in result.result
