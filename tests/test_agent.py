"""runagent 主循环的单元测试：剧本化的假 provider，全部本地执行，不发网络请求。"""

from __future__ import annotations

import pytest

from mini_code_agent.core.agent import runagent
from mini_code_agent.providers.base import ModelResponse, ToolCall
from mini_code_agent.tools import Tool, ToolRegistry

ADD_SCHEMA = {
    "type": "object",
    "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
    "required": ["a", "b"],
}


def add(a: int, b: int) -> int:
    return a + b


class ScriptedProvider:
    """按剧本依次吐出预设响应，同时记下每次调用收到了什么。"""

    def __init__(self, responses: list[ModelResponse]) -> None:
        self.responses = list(responses)
        self.calls: list[dict] = []

    def complete(self, messages, tools=None, system=None) -> ModelResponse:
        self.calls.append(
            {
                "messages": [dict(m) for m in messages],
                "tools": tools,
                "system": system,
            }
        )
        return self.responses.pop(0)


def tool_turn(call_id: str, name: str, arguments: dict) -> ModelResponse:
    """构造一个「模型要求调用工具」的响应。"""
    return ModelResponse(
        text=None,
        tool_calls=[ToolCall(id=call_id, name=name, arguments=arguments)],
        assistant_content=[
            {"type": "tool_use", "id": call_id, "name": name, "input": arguments}
        ],
        stop_reason="tool_use",
    )


def text_turn(text: str) -> ModelResponse:
    return ModelResponse(
        text=text,
        assistant_content=[{"type": "text", "text": text}],
        stop_reason="end_turn",
    )


@pytest.fixture
def registry() -> ToolRegistry:
    return ToolRegistry([Tool(name="add", description="求和", fn=add, params=ADD_SCHEMA)])


def test_stops_after_a_plain_reply(registry):
    provider = ScriptedProvider([text_turn("你好")])

    result = runagent("你好", provider, registry)

    assert result.final == "你好"
    assert result.trace == ["你好"]
    assert [m["role"] for m in result.messages] == ["user"]
    assert len(provider.calls) == 1


def test_trace_renders_the_call_its_result_and_the_final_answer(registry):
    provider = ScriptedProvider([tool_turn("toolu_1", "add", {"a": 1, "b": 2}), text_turn("3")])

    result = runagent("1+2 等于几", provider, registry)

    assert result.trace == ['→ add({"a": 1, "b": 2})', "← add: 3", "3"]


def test_no_argument_call_and_structured_result_are_rendered():
    registry = ToolRegistry([Tool(name="ls", description="列目录", fn=lambda: ["a.py", "b.py"])])
    provider = ScriptedProvider([tool_turn("toolu_1", "ls", {}), text_turn("两个文件")])

    result = runagent("列一下", provider, registry)

    assert result.trace == ['→ ls()', '← ls: ["a.py", "b.py"]', "两个文件"]


def test_tool_call_is_executed_and_fed_back(registry):
    provider = ScriptedProvider([tool_turn("toolu_1", "add", {"a": 1, "b": 2}), text_turn("3")])

    result = runagent("1+2 等于几", provider, registry)

    assert result.final == "3"
    assert [m["role"] for m in result.messages] == ["user", "assistant", "user"]


def test_tool_result_points_back_at_the_tool_use(registry):
    provider = ScriptedProvider([tool_turn("toolu_1", "add", {"a": 1, "b": 2}), text_turn("3")])

    result = runagent("1+2 等于几", provider, registry)

    assert result.messages[1]["content"][0]["id"] == "toolu_1"
    block = result.messages[2]["content"][0]
    assert block["type"] == "tool_result"
    assert block["tool_use_id"] == "toolu_1"
    assert block["content"] == "3"


def test_assistant_content_is_replayed_untouched(registry):
    response = tool_turn("toolu_1", "add", {"a": 1, "b": 2})
    provider = ScriptedProvider([response, text_turn("3")])

    result = runagent("1+2", provider, registry)

    assert result.messages[1]["content"] == response.assistant_content


def test_every_request_carries_the_tool_schemas(registry):
    provider = ScriptedProvider([tool_turn("toolu_1", "add", {"a": 1, "b": 2}), text_turn("3")])

    runagent("1+2", provider, registry)

    assert provider.calls[0]["tools"] == registry.schemas()
    assert provider.calls[1]["tools"] == registry.schemas()


def test_system_prompt_is_forwarded_on_every_step(registry):
    provider = ScriptedProvider([tool_turn("toolu_1", "add", {"a": 1, "b": 2}), text_turn("3")])

    runagent("1+2", provider, registry, system="你是一个计算器。")

    assert [call["system"] for call in provider.calls] == ["你是一个计算器。"] * 2


def test_without_a_registry_no_tools_are_sent():
    provider = ScriptedProvider([text_turn("你好")])

    runagent("你好", provider)

    assert not provider.calls[0]["tools"]


def test_multiple_tool_rounds_interleave_calls_and_results(registry):
    provider = ScriptedProvider(
        [
            tool_turn("toolu_1", "add", {"a": 1, "b": 2}),
            tool_turn("toolu_2", "add", {"a": 3, "b": 4}),
            text_turn("7"),
        ]
    )

    result = runagent("算两次", provider, registry)

    assert result.trace == [
        '→ add({"a": 1, "b": 2})',
        "← add: 3",
        '→ add({"a": 3, "b": 4})',
        "← add: 7",
        "7",
    ]
    assert result.final == "7"
    assert len(provider.calls) == 3


def test_a_failing_tool_shows_up_as_an_error_line(registry):
    provider = ScriptedProvider([tool_turn("toolu_1", "nope", {}), text_turn("换个办法")])

    result = runagent("调用一个不存在的工具", provider, registry)

    assert result.trace[0] == "→ nope()"
    assert result.trace[1].startswith("✗ nope: ")
    assert "KeyError" in result.trace[1]
    assert result.final == "换个办法"


def test_running_out_of_steps_ends_with_a_result_not_an_answer(registry):
    script = [tool_turn(f"toolu_{i}", "add", {"a": i, "b": i}) for i in range(3)]
    provider = ScriptedProvider(script)

    result = runagent("没完没了", provider, registry, max_steps=3)

    assert result.final is None
    assert len(result.trace) == 6
    assert result.trace[-1].startswith("← add: ")
