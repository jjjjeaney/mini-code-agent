"""Tool / ToolResult / ToolRegistry 的单元测试，全部本地执行，不发网络请求。"""

from __future__ import annotations

import json

import pytest

from mini_code_agent.providers.base import ToolCall
from mini_code_agent.tools import Tool, ToolRegistry, ToolResult

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
