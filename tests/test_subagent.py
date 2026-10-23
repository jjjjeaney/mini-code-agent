"""core/subagent.py 的单元测试：类型表、工具池、那一行过程、带标记的 prompter。

全部本地执行。这一步还没有调用方——`make_agent_tool` 在下一步——所以这里测的都是零件。
"""

from __future__ import annotations

import threading
from collections.abc import Sequence

import pytest

from mini_code_agent.core.subagent import (
    AGENT_SCHEMA,
    AGENT_TYPES,
    AgentType,
    _fork_task_message,
    _LabeledPrompter,
    _pool,
    _tally,
    make_agent_tool,
)
from mini_code_agent.core.tasks import TaskManager
from mini_code_agent.permissions import Grants, PermissionRequest
from mini_code_agent.prompts import Answer, BackgroundPrompter, NullPrompter
from mini_code_agent.providers.base import ModelResponse, ProviderError, ToolCall
from mini_code_agent.system_prompt import _SUBAGENT_STATIC, SUBAGENT_HEADS
from mini_code_agent.tools import Tool, ToolContext
from mini_code_agent.usage import UsageMeter


def names(agent_type: AgentType) -> set[str]:
    return {tool.name for tool in _pool(agent_type).list()}


def request(tool: str = "write_file") -> PermissionRequest:
    return PermissionRequest(tool_name=tool, tool_args={"path": "src/a.py"})


class RecordingPrompter:
    """记下每一次问的是什么，按构造参数回答。"""

    def __init__(self, answer: Answer | None = None, choice: str = "第一个") -> None:
        self.answer = answer if answer is not None else Answer(True)
        self.choice = choice
        self.asked: list = []

    def confirm_tool(self, req: PermissionRequest) -> Answer:
        self.asked.append(("tool", req))
        return self.answer

    def confirm_edit(self, req: PermissionRequest) -> Answer:
        self.asked.append(("edit", req))
        return self.answer

    def single_choice(self, question: str, options: Sequence[str]) -> str:
        self.asked.append(("choice", question, tuple(options)))
        return self.choice


class TestAgentTypes:
    def test_the_table_matches_the_prompt_heads(self):
        """表和提示词必须一一对应。

        对不上的后果是**跑到一半才炸**：`build_subagent_prompt` 认不出那个名字会抛
        ValueError，而那时子 agent 已经派出去了、钱已经花了。所以在这儿钉死。
        """
        assert {name for name, t in AGENT_TYPES.items() if not t.fork} == set(SUBAGENT_HEADS)

    def test_the_key_is_the_name(self):
        """同一个名字写了两次（字典键和 `name` 字段）。冗余就会漂，所以钉一条。"""
        for key, agent_type in AGENT_TYPES.items():
            assert key == agent_type.name

    def test_every_type_has_a_summary_for_the_model(self):
        """那段 summary 是模型挑类型的唯一依据——空着等于让它猜。"""
        for agent_type in AGENT_TYPES.values():
            assert agent_type.summary.strip()

    def test_the_schema_lets_the_model_see_the_types(self):
        """枚举写进 schema：模型看得见有哪些可选，不用猜（猜错要花掉一步）。"""
        assert AGENT_SCHEMA["properties"]["type"]["enum"] == list(AGENT_TYPES)

    def test_the_schema_requires_all_three(self):
        assert AGENT_SCHEMA["required"] == ["type", "description", "prompt"]


class TestTheToolPool:
    def test_the_read_only_types_only_get_read_only_tools(self):
        """explore / plan **只能读**——这不是靠提示词求它别改，是池子里压根没有能改的东西。"""
        for name in ("explore", "plan"):
            assert [t.name for t in _pool(AGENT_TYPES[name]).list() if not t.read_only] == []

    def test_the_read_only_types_still_get_the_reading_tools(self):
        """池子不能空，也不能少——空了它什么都干不了，只会白烧步数。"""
        for name in ("explore", "plan"):
            assert {"read_file", "grep", "glob", "list_files", "project_tree"} <= names(
                AGENT_TYPES[name]
            )

    def test_the_general_one_can_act(self):
        assert {"write_file", "edit_file", "replace_lines", "bash"} <= names(
            AGENT_TYPES["general-purpose"]
        )

    def test_no_type_can_spawn_another(self):
        """**递归护栏。**

        它是**结构性**的：池子从 `default_tools()` 现筛一份新的，而 `agent` 是在 CLI 层
        另外 `registry.register()` 进去的，两者根本不相干。也就是说这条护栏靠"新增工具
        一律进 `default_tools()`"这个隐含约定撑着——所以它值得一条测试。
        """
        for agent_type in AGENT_TYPES.values():
            if agent_type.fork:
                # fork 用的是父 agent 的工具表，不经过 _pool——它另有运行时的深度护栏。
                with pytest.raises(ValueError, match="父 agent"):
                    _pool(agent_type)
                continue
            assert "agent" not in names(agent_type)
            # 以前那个独立的 fork 工具也不该借尸还魂。
            assert "fork" not in names(agent_type)

    def test_no_type_can_ask_the_user(self):
        """子 agent 问不了用户：该问的事它写进总结、由主 agent 去问。

        顺带这也是 `NobodyToAskError` 在子 agent 里触发不了的原因——那个只从
        `single_choice` 抛，而 `single_choice` 只有 `ask_user` 会调。
        """
        for agent_type in AGENT_TYPES.values():
            if agent_type.fork:
                continue  # 用的是父 agent 的工具表，不在这条断言的范围里
            assert "ask_user" not in names(agent_type)

    def test_no_type_can_switch_the_permission_mode(self):
        """切档是主 agent 的事。池子从 `default_tools()` 重建，这个工具本来就不在里面。"""
        for agent_type in AGENT_TYPES.values():
            if agent_type.fork:
                continue
            assert "set_permission_mode" not in names(agent_type)

    def test_no_type_can_write_long_term_memory(self):
        """`remember` 是"以后每次会话都生效"的东西——该由知道整段对话的主 agent 来记。"""
        for agent_type in AGENT_TYPES.values():
            if agent_type.fork:
                continue
            assert "remember" not in names(agent_type)

    def test_each_call_builds_a_fresh_registry(self):
        """池子是按类型现建的，不是共用一份——共用的话一个类型筛过，另一个类型就跟着变。"""
        assert _pool(AGENT_TYPES["explore"]) is not _pool(AGENT_TYPES["explore"])


def a_tool_call(name: str, call_id: str = "t1") -> dict:
    return {
        "role": "assistant",
        "content": [{"type": "tool_use", "id": call_id, "name": name, "input": {}}],
    }


class TestTheTally:
    def test_it_counts_by_tool_name(self):
        messages = [
            a_tool_call("read_file"),
            a_tool_call("read_file"),
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1"}]},
            a_tool_call("grep"),
        ]

        assert _tally(messages) == "3 次工具调用（read_file×2、grep×1）"

    def test_it_sorts_by_count(self):
        messages = [a_tool_call("grep"), a_tool_call("read_file"), a_tool_call("read_file")]

        assert _tally(messages) == "3 次工具调用（read_file×2、grep×1）"

    def test_it_says_when_nothing_was_called(self):
        """一次没调过（子 agent 直接答了）也要说一声——那一行是要拼进结果里的，不能空。"""
        assert _tally([{"role": "user", "content": "问一句"}]) == "没调过工具"

    def test_it_ignores_text_and_results(self):
        """只数 `tool_use`。工具结果那一批里没有 `tool_use` 块，不该被算进去。"""
        messages = [
            {"role": "assistant", "content": [{"type": "text", "text": "我看看"}]},
            a_tool_call("read_file"),
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1"}]},
        ]

        assert _tally(messages) == "1 次工具调用（read_file×1）"

    def test_a_long_breakdown_is_cut_off(self):
        """它进的是**主 agent 的上下文**——那一行只是"查得多深"的信号，不是流水账。"""
        messages = [a_tool_call(name) for name in "abcdefg"]

        assert _tally(messages) == "7 次工具调用（a×1、b×1、c×1、d×1、e×1 等 7 类）"

    def test_a_dict_without_content_is_skipped(self):
        """历史里可能有没带 content 的消息（比如错误路径上攒出来的），别在那儿炸。"""
        assert _tally([{"role": "assistant"}, {"role": "user", "content": "字符串"}]) == (
            "没调过工具"
        )


class TestTheLabeledPrompter:
    def test_it_says_which_subagent_is_asking(self, capsys):
        """不标的话，用户看到的提示和主 agent 一模一样——配上"Grants 两份共用"，
        他分不清"本会话别再问"批给了谁、批的是什么。"""
        _LabeledPrompter(RecordingPrompter(), "explore").confirm_edit(request())

        assert "（[explore] 子 agent 请求：）" in capsys.readouterr().out

    def test_it_hands_every_method_through(self, capsys):
        """三个方法都要转发——协议要完整，少一个就是把那句话吞了。"""
        inner = RecordingPrompter()
        labeled = _LabeledPrompter(inner, "plan")

        labeled.confirm_tool(request())
        labeled.confirm_edit(request())
        labeled.single_choice("选哪个？", ["a", "b"])

        assert [entry[0] for entry in inner.asked] == ["tool", "edit", "choice"]

    def test_the_request_goes_through_untouched(self, capsys):
        """原样转发：内层收到的是**同一个** request 对象，一个字段都没改。"""
        inner = RecordingPrompter()
        asked = request("edit_file")

        _LabeledPrompter(inner, "explore").confirm_edit(asked)

        assert inner.asked == [("edit", asked)]

    def test_the_answer_comes_back_untouched(self, capsys):
        inner = RecordingPrompter(Answer(False, note="这个文件先别动"))

        answer = _LabeledPrompter(inner, "explore").confirm_edit(request())

        assert answer is inner.answer
        assert answer.allowed is False
        assert answer.note == "这个文件先别动"

    def test_the_choice_comes_back_too(self, capsys):
        inner = RecordingPrompter(choice="第二个")

        assert _LabeledPrompter(inner, "explore").single_choice("选哪个？", ["a", "b"]) == "第二个"
        assert inner.asked == [("choice", "选哪个？", ("a", "b"))]


class ScriptedProvider:
    """按剧本依次回答，记下每次收到了什么。子 agent 发的请求也走这一个对象。"""

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


class ProviderThatFails:
    def complete(self, messages, tools=None, system=None) -> ModelResponse:
        raise ProviderError("连不上模型服务。")


def tool_turn(call_id: str, name: str, arguments: dict) -> ModelResponse:
    return ModelResponse(
        text=None,
        tool_calls=[ToolCall(id=call_id, name=name, arguments=arguments)],
        assistant_content=[{"type": "tool_use", "id": call_id, "name": name, "input": arguments}],
        stop_reason="tool_use",
    )


def text_turn(text: str) -> ModelResponse:
    return ModelResponse(
        text=text,
        assistant_content=[{"type": "text", "text": text}],
        stop_reason="end_turn",
    )


def the_tool(
    provider,
    *,
    prompter=None,
    grants=None,
    tasks=None,
    background_after=120.0,
    broker=None,
) -> Tool:
    # `make_agent_tool` 在派子 agent 时现读 provider.model，所以模型名挂在 provider 上，
    # 不再作为工具参数冻结。
    provider.model = "test-model"
    return make_agent_tool(
        provider=provider,
        prompter=prompter if prompter is not None else NullPrompter(),
        grants=grants if grants is not None else Grants(),
        tasks=tasks,
        background_after=background_after,
        broker=broker,
    )


def call_the_tool(
    tool: Tool,
    ctx: ToolContext,
    *,
    kind: str = "explore",
    prompt: str = "找到压缩那段代码",
    **extra,
) -> str:
    return tool.fn(
        {"type": kind, "description": "找压缩逻辑", "prompt": prompt, **extra}, ctx
    )


class TestTheAgentToolShape:
    """这个工具长什么样——模型是照着它决定用不用、怎么用的。"""

    def test_it_is_named_agent(self):
        assert the_tool(ScriptedProvider([])).name == "agent"

    def test_the_description_offers_every_type(self):
        """模型挑类型的唯一依据就是这段说明——漏一个它就永远不会选那个。"""
        description = the_tool(ScriptedProvider([])).description

        for name, agent_type in AGENT_TYPES.items():
            assert name in description
            assert agent_type.summary in description

    def test_the_description_warns_about_plan_mode(self):
        """`type` 的枚举是静态的、收不窄，所以"plan 档下别派 general-purpose"只能在文字里拦。"""
        assert "plan 档" in the_tool(ScriptedProvider([])).description

    def test_it_spills_long_summaries(self):
        """总结没有天然上限。不封顶的话，"只把结论拿回来"这句话就是空的。"""
        assert the_tool(ScriptedProvider([])).spill is True

    def test_the_model_name_is_read_when_the_subagent_starts(self, tmp_path):
        """模型名在派发时现读 provider.model——`/model` 中途换了它，下一次派发就该用新的。"""
        provider = ScriptedProvider([text_turn("结论"), text_turn("结论")])
        tool = the_tool(provider)
        ctx = ToolContext(cwd=tmp_path)

        provider.model = "model-a"
        call_the_tool(tool, ctx, prompt="第一件事")
        provider.model = "model-b"
        call_the_tool(tool, ctx, prompt="第二件事")

        assert "模型 model-a" in provider.calls[0]["system"]
        assert "模型 model-b" in provider.calls[1]["system"]

    def test_its_result_can_never_be_folded(self):
        """**子 agent 交回来的结论不许被折掉。**

        它是"方向"本身：折了模型就不知道自己在干什么了，而且它背后是一整趟子 agent——重跑
        一遍是几分钟和几毛钱。做法是**什么都不做**：`foldable` 默认就是 False，这个工具没有
        标它，折叠那边按 `Tool.foldable` 筛，自然落在外面（见 folding.py）。
        """
        assert the_tool(ScriptedProvider([])).foldable is False


class TestTheForkTaskMessage:
    def test_it_states_the_result_format(self):
        """fork 继承父 agent 的 system prompt，没法在里面加格式——只能写进追加的任务消息。"""
        text = _fork_task_message("做点什么")

        for section in ("【做了什么】", "【改了哪些文件】", "【验证结果】", "【遗留问题】"):
            assert section in text


class TestRunningASubagent:
    def test_an_unknown_type_comes_back_as_a_plain_result(self, tmp_path):
        """名字不对是**模型自己的输入**脏了，不是工具坏了——当普通结果回给它改，别抛。"""
        provider = ScriptedProvider([])

        text = call_the_tool(
            the_tool(provider), ToolContext(cwd=tmp_path), kind="explore-everything"
        )

        assert "没有名为" in text
        assert "explore" in text  # 把可选的列出来，它才知道该改成什么
        assert provider.calls == []  # 一发都没发出去

    def test_it_inherits_the_parents_sandbox_mode(self, tmp_path):
        provider = ScriptedProvider([text_turn("结论")])

        call_the_tool(the_tool(provider), ToolContext(cwd=tmp_path, sandbox="host"))

        assert "宿主机" in provider.calls[0]["system"]

    def test_its_requests_go_into_the_parents_turn_meter(self, tmp_path):
        """子 agent 花的是另一串请求，但记的是主 agent 这一轮的账。"""
        meter = UsageMeter()
        provider = ScriptedProvider(
            [
                ModelResponse(
                    text="结论",
                    assistant_content=[{"type": "text", "text": "结论"}],
                    stop_reason="end_turn",
                    input_tokens=10,
                    output_tokens=2,
                )
            ]
        )

        call_the_tool(the_tool(provider), ToolContext(cwd=tmp_path, usage=meter))

        totals = meter.snapshot()
        assert totals.requests == 1
        assert totals.input_tokens == 10
        assert totals.output_tokens == 2

    def test_an_empty_prompt_is_refused(self, tmp_path):
        with pytest.raises(ValueError, match="非空"):
            call_the_tool(the_tool(ScriptedProvider([])), ToolContext(cwd=tmp_path), prompt="   ")

    def test_the_child_gets_its_own_system_prompt(self, tmp_path):
        """它拿到的是**子 agent 那份**（身份不同、问不了用户），不是主 agent 那份。"""
        provider = ScriptedProvider([text_turn("在 src/summarizing.py:70")])

        call_the_tool(the_tool(provider), ToolContext(cwd=tmp_path), kind="explore")

        system = provider.calls[0]["system"]
        assert _SUBAGENT_STATIC in system
        assert SUBAGENT_HEADS["explore"] in system

    def test_the_child_gets_only_the_read_only_pool(self, tmp_path):
        """explore 只能读，是**池子里没有可写的东西**，不是靠提示词求它别改。"""
        provider = ScriptedProvider([text_turn("结论")])

        call_the_tool(the_tool(provider), ToolContext(cwd=tmp_path), kind="explore")

        offered = {schema["name"] for schema in provider.calls[0]["tools"]}
        assert {"read_file", "grep", "glob"} <= offered
        assert not offered & {"write_file", "edit_file", "replace_lines", "bash"}
        assert "agent" not in offered  # 不能递归
        assert "ask_user" not in offered

    def test_the_child_cannot_see_the_parent_history(self, tmp_path):
        """它的全部信息就是那段 prompt——这是"只把结论拿回来"的另一面。"""
        provider = ScriptedProvider([text_turn("结论")])

        call_the_tool(
            the_tool(provider), ToolContext(cwd=tmp_path), prompt="找到压缩那段代码"
        )

        assert provider.calls[0]["messages"] == [{"role": "user", "content": "找到压缩那段代码"}]

    def test_the_result_leads_with_the_process_line(self, tmp_path):
        """那一行在**最前面**：下面还要过长度闸，放末尾会被连同正文一起截掉。"""
        (tmp_path / "a.txt").write_text("正文\n", encoding="utf-8")
        provider = ScriptedProvider(
            [tool_turn("t1", "read_file", {"path": "a.txt"}), text_turn("这里就是结论")]
        )

        text = call_the_tool(the_tool(provider), ToolContext(cwd=tmp_path))

        assert text.startswith("[explore 子 agent：1 次工具调用（read_file×1）]")
        assert text.endswith("这里就是结论")

    def test_it_says_so_when_the_child_never_concluded(self, tmp_path):
        """每一发都在调工具、永远不给结论——用满步数，收尾那一发也是空的。"""
        (tmp_path / "a.txt").write_text("正文\n", encoding="utf-8")
        provider = ScriptedProvider(
            [tool_turn(f"t{i}", "read_file", {"path": "a.txt"}) for i in range(9)]
        )

        text = call_the_tool(the_tool(provider), ToolContext(cwd=tmp_path))

        assert "没给出结论" in text
        assert text.startswith("[explore 子 agent：")  # 过程那行照样有

    def test_it_says_so_when_the_child_could_not_run(self, tmp_path):
        """连不上是**信息**，不是"工具坏了"——不当成 is_error 抛给模型看。"""
        text = call_the_tool(the_tool(ProviderThatFails()), ToolContext(cwd=tmp_path))

        assert "没跑成" in text
        assert "连不上模型服务" in text


class TestTheGateInsideASubagent:
    """子 agent **不是绕开闸门的一条路**：它改文件照样问用户。"""

    def test_a_write_still_asks_the_user(self, tmp_path):
        prompter = RecordingPrompter(Answer(False, note="先别动"))
        provider = ScriptedProvider(
            [
                tool_turn("t1", "write_file", {"path": "新文件.txt", "content": "x"}),
                text_turn("那我不改了"),
            ]
        )

        text = call_the_tool(
            the_tool(provider, prompter=prompter),
            ToolContext(cwd=tmp_path),
            kind="general-purpose",
        )

        assert [entry[0] for entry in prompter.asked] == ["edit"]  # 改文件那一档问法
        assert not (tmp_path / "新文件.txt").exists()  # 拒了就没写
        assert text.endswith("那我不改了")

    def test_a_plan_mode_child_cannot_write(self, tmp_path):
        """档位是**继承**来的。不继承的话，plan 档下派一个 general-purpose 子 agent
        就能改文件——那是 plan 档的一个洞。"""
        provider = ScriptedProvider(
            [
                tool_turn("t1", "write_file", {"path": "新文件.txt", "content": "x"}),
                text_turn("改不了"),
            ]
        )

        call_the_tool(
            the_tool(provider),
            ToolContext(cwd=tmp_path, permission_mode="plan"),
            kind="general-purpose",
        )

        sent_back = provider.calls[1]["messages"][-1]["content"][0]
        assert sent_back["is_error"] is True
        assert "plan" in sent_back["content"]
        assert not (tmp_path / "新文件.txt").exists()

    def test_a_grant_carries_between_the_parent_and_the_child(self, tmp_path):
        """`Grants` 是两份共用的：主 agent 批过的文件，子 agent 不再问。

        它本来就是按 (工具, 文件路径) 记的会话级批准——用户批的是"这个文件、这次会话"，
        不是"谁去动它"。代价是批过的那个文件子 agent 写什么都不会再问，所以权限提示必须
        带出来源标记（见下一条）。
        """
        granted = Grants()
        granted.remember(PermissionRequest(tool_name="write_file", tool_args={"path": "a.txt"}))
        prompter = RecordingPrompter(Answer(False))  # 但凡是问到了就拒
        (tmp_path / "a.txt").write_text("原来的\n", encoding="utf-8")
        provider = ScriptedProvider(
            [
                tool_turn("t1", "read_file", {"path": "a.txt"}),
                tool_turn("t2", "write_file", {"path": "a.txt", "content": "新的\n"}),
                text_turn("改好了"),
            ]
        )

        call_the_tool(
            the_tool(provider, prompter=prompter, grants=granted),
            ToolContext(cwd=tmp_path),
            kind="general-purpose",
        )

        assert prompter.asked == []  # 一次都没问
        assert (tmp_path / "a.txt").read_text(encoding="utf-8") == "新的\n"

    def test_the_permission_prompt_says_which_subagent(self, tmp_path, capsys):
        """不标的话，用户看到的提示和主 agent 一模一样——配上上面那条"Grants 共用"，
        他分不清"本会话别再问"批给了谁。"""
        prompter = RecordingPrompter(Answer(False))
        provider = ScriptedProvider(
            [
                tool_turn("t1", "write_file", {"path": "新文件.txt", "content": "x"}),
                text_turn("好"),
            ]
        )

        call_the_tool(
            the_tool(provider, prompter=prompter),
            ToolContext(cwd=tmp_path),
            kind="general-purpose",
        )

        out = capsys.readouterr().out
        assert "（[general-purpose] 子 agent 请求：）" in out
        # 头尾两行也在，用户看得见"派了谁、去干什么、跑完调了几次工具"
        assert "▶ [general-purpose] 找压缩逻辑" in out


class TestBackground:
    """`run_in_background` 和两分钟自动交接。"""

    def test_the_schema_offers_run_in_background(self):
        assert AGENT_SCHEMA["properties"]["run_in_background"]["type"] == "boolean"
        assert "run_in_background" not in AGENT_SCHEMA["required"]

    def test_it_returns_a_task_id_instead_of_waiting(self, tmp_path):
        gate = threading.Event()

        class BlockingProvider:
            def complete(self, messages, tools=None, system=None):
                gate.wait(2)
                return text_turn("结论")

        manager = TaskManager()
        tool = the_tool(BlockingProvider(), tasks=manager, background_after=0.0)

        out = call_the_tool(
            tool,
            ToolContext(cwd=tmp_path, allow_background=True),
            run_in_background=True,
        )

        assert "已派发" in out
        assert "不要重派" in out
        assert manager.running() == 1
        gate.set()
        assert manager.wait(manager.list()[0].id, timeout=2)

    def test_a_long_foreground_task_is_moved_to_the_background(self, tmp_path):
        provider = ScriptedProvider([text_turn("结论")])
        manager = TaskManager()
        tool = the_tool(provider, tasks=manager, background_after=0.0)

        out = call_the_tool(
            tool, ToolContext(cwd=tmp_path, allow_background=True)
        )

        assert "转到后台" in out
        assert manager.wait(manager.list()[0].id, timeout=2)
        assert "结论" in manager.take_notices()

    def test_without_background_support_it_runs_synchronously(self, tmp_path):
        """没有下一轮可以送结果的回合——模型要了后台也只能同步跑完，并且说明一句。"""
        provider = ScriptedProvider([text_turn("结论")])
        tool = the_tool(provider, tasks=TaskManager(), background_after=0.0)

        out = call_the_tool(
            tool, ToolContext(cwd=tmp_path), run_in_background=True
        )

        assert "没有后台模式" in out
        assert "结论" in out
        assert len(provider.calls) == 1

    def test_the_concurrency_limit_refuses_a_third_background_task(self, tmp_path):
        gate = threading.Event()

        class BlockingProvider:
            def complete(self, messages, tools=None, system=None):
                gate.wait(2)
                return text_turn("结论")

        manager = TaskManager()
        tool = the_tool(BlockingProvider(), tasks=manager, background_after=0.0)
        ctx = ToolContext(cwd=tmp_path, allow_background=True)

        call_the_tool(tool, ctx, run_in_background=True)
        call_the_tool(tool, ctx, run_in_background=True)
        out = call_the_tool(tool, ctx, run_in_background=True)

        assert "已经有 2 个" in out
        gate.set()
        for record in manager.list():
            manager.wait(record.id, timeout=2)

    def test_without_a_broker_the_background_half_cannot_ask(self):
        """没有 broker（主要是测试）时退回兜底：需要当场批准的事写进结论。"""
        answer = BackgroundPrompter().confirm_edit(request())

        assert answer.allowed is False
        assert "后台任务" in answer.note
        assert "结论" in answer.note

    def test_a_background_task_can_ask_through_a_broker(self, tmp_path):
        """有 broker 时，后台的写文件请求进队列，而不是被兜底 prompter 直接拒掉。"""

        class FakeBroker:
            def __init__(self):
                self.requests = []

            def submit(self, request, label, cancel=None):
                self.requests.append((request, label))
                return Answer(False, note="测试里拒绝")

        broker = FakeBroker()
        manager = TaskManager()
        provider = ScriptedProvider(
            [
                tool_turn("w1", "write_file", {"path": "a.txt", "content": "x"}),
                text_turn("结论"),
            ]
        )
        tool = the_tool(provider, tasks=manager, background_after=0.0, broker=broker)
        ctx = ToolContext(cwd=tmp_path, allow_background=True)

        call_the_tool(tool, ctx, kind="general-purpose", run_in_background=True)
        assert manager.wait(manager.list()[0].id, timeout=2)

        assert [req.tool_name for req, _ in broker.requests] == ["write_file"]
        assert broker.requests[0][1] == "[general-purpose] "

    def test_without_a_broker_the_background_half_still_denies(self, tmp_path):
        manager = TaskManager()
        provider = ScriptedProvider(
            [
                tool_turn("w1", "write_file", {"path": "a.txt", "content": "x"}),
                text_turn("结论"),
            ]
        )
        tool = the_tool(provider, tasks=manager, background_after=0.0)
        ctx = ToolContext(cwd=tmp_path, allow_background=True)

        call_the_tool(tool, ctx, kind="general-purpose", run_in_background=True)
        assert manager.wait(manager.list()[0].id, timeout=2)

        # 第二发请求里才带着被拒绝的 tool_result（第一发只有任务本身）
        assert "后台任务" in str(provider.calls[1]["messages"])
