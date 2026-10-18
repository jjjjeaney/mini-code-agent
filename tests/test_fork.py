"""fork 类型：继承父 agent 的上下文、复用缓存前缀。全部本地执行。

fork 不再是单独的工具——它是 `agent` 的 `type` 枚举里的一项，和 explore / plan /
general-purpose 同级。
"""

from __future__ import annotations

import pytest

from mini_code_agent.core.agent import runagent
from mini_code_agent.core.subagent import (
    AGENT_SCHEMA,
    AGENT_TYPES,
    make_agent_tool,
)
from mini_code_agent.core.tasks import TaskManager
from mini_code_agent.permissions import Grants
from mini_code_agent.prompts import Answer
from mini_code_agent.providers.base import ModelResponse, ProviderError, ToolCall
from mini_code_agent.tools import (
    ForkContext,
    ReadFileState,
    Tool,
    ToolContext,
    ToolRegistry,
    default_tools,
)


class ScriptedProvider:
    """按剧本回答，记下每次收到了什么。"""

    def __init__(self, responses: list[ModelResponse]) -> None:
        self.responses = list(responses)
        self.calls: list[dict] = []

    def complete(self, messages, tools=None, system=None) -> ModelResponse:
        self.calls.append(
            {"messages": [dict(m) for m in messages], "tools": tools, "system": system}
        )
        return self.responses.pop(0)


def tool_turn(call_id: str, name: str, arguments: dict) -> ModelResponse:
    return ModelResponse(
        text=None,
        tool_calls=[ToolCall(id=call_id, name=name, arguments=arguments)],
        assistant_content=[
            {"type": "tool_use", "id": call_id, "name": name, "input": arguments}
        ],
        stop_reason="tool_use",
    )


def fork_turn(call_id: str = "f1", *, description: str = "试试", prompt: str = "做事"):
    """主 agent 派一个 fork：还是 `agent` 这个工具，只是 type 换成 fork。"""
    return tool_turn(
        call_id,
        "agent",
        {"type": "fork", "description": description, "prompt": prompt},
    )


def text_turn(text: str) -> ModelResponse:
    return ModelResponse(
        text=text,
        assistant_content=[{"type": "text", "text": text}],
        stop_reason="end_turn",
    )


class RecordingPrompter:
    def __init__(self, *, allow: bool = True) -> None:
        self.allow = allow
        self.edits: list = []

    def confirm_edit(self, request):
        self.edits.append(request)
        return Answer(self.allow)

    def confirm_tool(self, request):
        return Answer(self.allow)

    def single_choice(self, question, options):  # pragma: no cover - 不该走到
        raise AssertionError("fork 的测试里不该出现选择题")


def the_tool(
    provider, prompter=None, *, tasks=None, background_after=120.0
) -> Tool:
    return make_agent_tool(
        provider=provider,
        prompter=prompter or RecordingPrompter(),
        grants=Grants(),
        model="test-model",
        tasks=tasks,
        background_after=background_after,
    )


def call_fork(
    tool: Tool,
    ctx: ToolContext,
    *,
    description: str = "试试",
    prompt: str = "做事",
    **extra,
) -> str:
    return tool.fn(
        {"type": "fork", "description": description, "prompt": prompt, **extra}, ctx
    )


def a_fork_context(
    messages: list[dict],
    *,
    system: str | None = "同一份系统提示",
    registry: ToolRegistry | None = None,
    depth: int = 0,
) -> ForkContext:
    return ForkContext(
        messages=messages,
        system=system,
        registry=registry if registry is not None else ToolRegistry([]),
        depth=depth,
    )


class TestTheType:
    def test_fork_is_one_of_the_agent_types(self):
        """和 explore / plan / general-purpose 同级——模型在同一个 `type` 里挑。"""
        assert "fork" in AGENT_TYPES
        assert "fork" in AGENT_SCHEMA["properties"]["type"]["enum"]
        assert AGENT_TYPES["fork"].fork is True

    def test_the_other_types_are_not_forks(self):
        others = (t for name, t in AGENT_TYPES.items() if name != "fork")
        assert all(not agent_type.fork for agent_type in others)

    def test_the_conclusion_still_spills_and_never_folds(self):
        tool = the_tool(ScriptedProvider([]))

        assert tool.spill is True
        assert tool.foldable is False


class TestTheGuards:
    def test_a_nested_fork_is_refused_with_a_plain_result(self, tmp_path):
        """递归靠深度在运行时拒：`agent` 一直挂在表上（删掉会改前缀、把缓存打没）。"""
        provider = ScriptedProvider([])
        ctx = ToolContext(cwd=tmp_path, fork=a_fork_context([], depth=1))

        result = call_fork(the_tool(provider), ctx)

        assert "不能再次 fork" in result
        assert provider.calls == []

    def test_an_empty_prompt_is_refused(self, tmp_path):
        ctx = ToolContext(cwd=tmp_path, fork=a_fork_context([]))

        with pytest.raises(ValueError, match="非空"):
            call_fork(the_tool(ScriptedProvider([])), ctx, prompt="  ")

    def test_a_missing_system_prompt_says_so(self, tmp_path):
        ctx = ToolContext(cwd=tmp_path, fork=a_fork_context([], system=None))

        result = call_fork(the_tool(ScriptedProvider([])), ctx)

        assert "没有 system prompt" in result

    def test_no_fork_context_says_the_flag_is_off(self, tmp_path):
        result = call_fork(
            the_tool(ScriptedProvider([])), ToolContext(cwd=tmp_path)
        )

        assert "--fork" in result


class TestTheCachePrefix:
    """**这一组是这个功能存在的理由。**"""

    def test_the_fork_request_is_the_parents_last_request_plus_the_task(self, tmp_path):
        """父 agent 上一步发出去的是 `messages[:-1]`；fork 在它后面只追加一条任务消息。"""
        provider = ScriptedProvider(
            [fork_turn(prompt="换个做法"), text_turn("fork 的结论"), text_turn("好")]
        )
        registry = ToolRegistry([the_tool(provider)])

        runagent(
            "先看看这个仓库",
            provider,
            registry,
            system="同一份系统提示",
            cwd=tmp_path,
            allow_fork=True,
        )

        parent_request = provider.calls[0]
        fork_request = provider.calls[1]

        assert (
            fork_request["messages"][: len(parent_request["messages"])]
            == parent_request["messages"]
        )
        assert fork_request["messages"][-1]["role"] == "user"
        assert "换个做法" in fork_request["messages"][-1]["content"]
        assert fork_request["system"] == parent_request["system"]
        assert fork_request["tools"] == parent_request["tools"]

    def test_the_fork_gets_the_parents_exact_tool_list(self, tmp_path):
        """父 agent 的工具表在开头算一次；fork 不能再算——中间变了就前功尽弃。"""
        state = {"on": True}

        class FlippingProvider(ScriptedProvider):
            def complete(self, messages, tools=None, system=None):
                response = super().complete(messages, tools, system)
                state["on"] = False  # 第一发之后，那个工具就该"消失"了
                return response

        provider = FlippingProvider(
            [fork_turn(prompt="换个做法"), text_turn("fork 的结论"), text_turn("好")]
        )
        registry = ToolRegistry(
            [
                the_tool(provider),
                Tool(
                    name="flaky",
                    fn=lambda kwargs, ctx: "",
                    available=lambda _: state["on"],
                ),
            ]
        )

        runagent("做点事", provider, registry, system="s", cwd=tmp_path, allow_fork=True)

        assert "flaky" in {schema["name"] for schema in provider.calls[0]["tools"]}
        assert provider.calls[1]["tools"] == provider.calls[0]["tools"]


class TestWhatTheForkSees:
    def test_it_sees_the_parent_history_but_not_the_current_tool_turn(self, tmp_path):
        provider = ScriptedProvider(
            [fork_turn(prompt="接着做"), text_turn("结论"), text_turn("好")]
        )
        history = [
            {"role": "user", "content": "背景一"},
            {"role": "assistant", "content": [{"type": "text", "text": "记下了"}]},
        ]

        runagent(
            "现在做 X",
            provider,
            ToolRegistry([the_tool(provider)]),
            system="s",
            cwd=tmp_path,
            messages=history,
            allow_fork=True,
        )

        fork_messages = provider.calls[1]["messages"]

        # 父 agent 上一步的请求 + 一条 fork 任务
        assert fork_messages[:-1] == provider.calls[0]["messages"]
        assert fork_messages[-1]["role"] == "user"
        assert "接着做" in fork_messages[-1]["content"]
        # 派发中的那条 assistant tool_use 不在里面
        assert not any(
            isinstance(message.get("content"), list)
            and any(
                block.get("type") == "tool_use"
                for block in message["content"]
                if isinstance(block, dict)
            )
            for message in fork_messages
        )

    def test_sibling_calls_in_the_same_batch_are_invisible(self, tmp_path):
        """整条 assistant 消息被丢掉，所以同一批里的兄弟调用和结果它都看不见。"""
        provider = ScriptedProvider(
            [
                ModelResponse(
                    text=None,
                    tool_calls=[
                        ToolCall(
                            id="f1",
                            name="agent",
                            arguments={
                                "type": "fork",
                                "description": "试试",
                                "prompt": "做事",
                            },
                        ),
                        ToolCall(id="s1", name="sibling", arguments={}),
                    ],
                    assistant_content=[
                        {
                            "type": "tool_use",
                            "id": "f1",
                            "name": "agent",
                            "input": {
                                "type": "fork",
                                "description": "试试",
                                "prompt": "做事",
                            },
                        },
                        {"type": "tool_use", "id": "s1", "name": "sibling", "input": {}},
                    ],
                    stop_reason="tool_use",
                ),
                text_turn("结论"),
                text_turn("好"),
            ]
        )
        registry = ToolRegistry(
            [the_tool(provider), Tool(name="sibling", fn=lambda kwargs, ctx: "兄弟结果")]
        )

        runagent("做点事", provider, registry, system="s", cwd=tmp_path, allow_fork=True)

        fork_messages = provider.calls[1]["messages"]
        assert "兄弟结果" not in str(fork_messages)
        assert "sibling" not in str(fork_messages)


class TestTheResult:
    def test_the_tally_counts_only_the_forks_own_calls(self, tmp_path):
        """`result.messages` 带着父 agent 的整段历史，计数必须从 `len(base)` 往后切。"""
        provider = ScriptedProvider(
            [
                fork_turn(),
                tool_turn("p1", "probe", {}),
                text_turn("结论"),
                text_turn("好"),
            ]
        )
        registry = ToolRegistry(
            [the_tool(provider), Tool(name="probe", fn=lambda kwargs, ctx: "探过")]
        )
        history = [
            {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "id": "old", "name": "probe", "input": {}}
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "old", "content": "旧结果"}
                ],
            },
        ]

        runagent(
            "做点事",
            provider,
            registry,
            system="s",
            cwd=tmp_path,
            messages=history,
            allow_fork=True,
        )

        # 父 agent 第二发里那条 tool_result 就是 fork 交回来的结论
        block = provider.calls[3]["messages"][-1]["content"][0]
        assert block["type"] == "tool_result"
        assert "1 次工具调用（probe×1）" in block["content"]

    def test_it_says_so_when_the_fork_could_not_run(self, tmp_path):
        class FailingProvider:
            def complete(self, messages, tools=None, system=None):
                raise ProviderError("连不上模型服务。")

        ctx = ToolContext(
            cwd=tmp_path,
            fork=a_fork_context([{"role": "user", "content": "背景"}]),
        )

        result = call_fork(the_tool(FailingProvider()), ctx)

        assert "没跑成" in result
        assert "连不上" in result

    def test_it_says_so_when_the_fork_never_concluded(self, tmp_path, monkeypatch):
        import mini_code_agent.core.subagent as subagent_module

        monkeypatch.setattr(subagent_module, "FORK_MAX_STEPS", 1)
        provider = ScriptedProvider(
            [tool_turn("p1", "probe", {}), tool_turn("p2", "probe", {})]
        )
        registry = ToolRegistry([Tool(name="probe", fn=lambda kwargs, ctx: "探过")])
        ctx = ToolContext(
            cwd=tmp_path,
            fork=a_fork_context([{"role": "user", "content": "背景"}], registry=registry),
        )

        result = call_fork(the_tool(provider), ctx)

        assert "没给出结论" in result


class TestBackground:
    def test_a_fork_can_go_background_too(self, tmp_path):
        """它和别的类型同级：`run_in_background` 对它一样有效。"""
        manager = TaskManager()
        provider = ScriptedProvider([text_turn("结论")])
        ctx = ToolContext(
            cwd=tmp_path,
            allow_background=True,
            fork=a_fork_context([{"role": "user", "content": "背景"}]),
        )

        result = call_fork(
            the_tool(provider, tasks=manager, background_after=0.0),
            ctx,
            run_in_background=True,
        )

        assert "已派发" in result
        assert manager.wait(manager.list()[0].id, timeout=2)


class TestReadState:
    """**fork 拿的是副本**：进来时继承，出去之后一条都不回流。"""

    def a_read_file(self, tmp_path):
        path = tmp_path / "a.txt"
        path.write_text("原始内容\n", encoding="utf-8")
        stat = path.stat()
        return path, stat

    def test_the_fork_inherits_what_the_parent_read(self, tmp_path):
        path, stat = self.a_read_file(tmp_path)
        state = ReadFileState()
        state.record(path, stat.st_size, stat.st_mtime_ns)
        provider = ScriptedProvider(
            [
                tool_turn(
                    "e1",
                    "edit_file",
                    {"path": "a.txt", "old_string": "原始内容", "new_string": "改过的"},
                ),
                text_turn("改好了"),
            ]
        )
        ctx = ToolContext(
            cwd=tmp_path,
            read_state=state,
            fork=a_fork_context([{"role": "user", "content": "背景"}], registry=default_tools()),
        )

        call_fork(
            the_tool(provider, RecordingPrompter(allow=True)),
            ctx,
            description="改一下",
            prompt="把 a.txt 改掉",
        )

        assert path.read_text(encoding="utf-8") == "改过的\n"
        # fork 的写入只落在副本上：父 agent 的记录原封不动（下次改会被告知"重读"）
        assert state.entries == {path: (stat.st_mtime_ns, stat.st_size)}

    def test_the_forks_reads_do_not_flow_back(self, tmp_path):
        self.a_read_file(tmp_path)
        provider = ScriptedProvider(
            [tool_turn("r1", "read_file", {"path": "a.txt"}), text_turn("看过了")]
        )
        ctx = ToolContext(
            cwd=tmp_path,
            fork=a_fork_context([], registry=default_tools()),
        )

        call_fork(the_tool(provider), ctx, description="看看", prompt="读一下 a.txt")

        assert ctx.read_state.entries == {}

    def test_a_plan_mode_fork_cannot_write(self, tmp_path):
        self.a_read_file(tmp_path)
        provider = ScriptedProvider(
            [
                tool_turn(
                    "e1",
                    "edit_file",
                    {"path": "a.txt", "old_string": "原始内容", "new_string": "改过的"},
                ),
                text_turn("没改成"),
            ]
        )
        ctx = ToolContext(
            cwd=tmp_path,
            permission_mode="plan",
            fork=a_fork_context([], registry=default_tools()),
        )

        call_fork(the_tool(provider), ctx, description="改一下", prompt="改掉")

        assert (tmp_path / "a.txt").read_text(encoding="utf-8") == "原始内容\n"
        assert "plan" in str(provider.calls[1]["messages"])

    def test_the_permission_prompt_says_fork(self, tmp_path, capsys):
        self.a_read_file(tmp_path)
        prompter = RecordingPrompter(allow=True)
        provider = ScriptedProvider(
            [
                tool_turn(
                    "e1",
                    "edit_file",
                    {"path": "a.txt", "old_string": "原始内容", "new_string": "改过的"},
                ),
                text_turn("改好了"),
            ]
        )
        ctx = ToolContext(
            cwd=tmp_path,
            fork=a_fork_context([], registry=default_tools()),
        )

        call_fork(the_tool(provider, prompter), ctx, description="改一下", prompt="改掉")

        assert prompter.edits  # 真问过
        assert "[fork]" in capsys.readouterr().out
