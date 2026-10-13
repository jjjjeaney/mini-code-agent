"""runagent 主循环的单元测试：剧本化的假 provider，全部本地执行，不发网络请求。"""

from __future__ import annotations

import pytest

from mini_code_agent.core.agent import runagent
from mini_code_agent.permissions import Grants, PermissionRequest
from mini_code_agent.prompting import make_ask_user
from mini_code_agent.prompts import Answer
from mini_code_agent.providers.base import ModelResponse, ToolCall
from mini_code_agent.tools import Tool, ToolContext, ToolRegistry, default_tools
from mini_code_agent.tools.shell import SHELL_SCHEMA, bash

ADD_SCHEMA = {
    "type": "object",
    "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
    "required": ["a", "b"],
}


# 工具函数现在收两样：模型给的参数字典，和一个 ToolContext。
def add(kwargs: dict, ctx: ToolContext) -> int:
    return kwargs["a"] + kwargs["b"]


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
    # 最后那条回复也进 messages——不然 result.messages 结尾是用户那侧，拿它接着聊就丢了
    # 模型说过的话。
    assert [m["role"] for m in result.messages] == ["user", "assistant"]
    assert len(provider.calls) == 1


def test_trace_renders_the_call_its_result_and_the_final_answer(registry):
    provider = ScriptedProvider([tool_turn("toolu_1", "add", {"a": 1, "b": 2}), text_turn("3")])

    result = runagent("1+2 等于几", provider, registry)

    assert result.trace == ['→ add({"a": 1, "b": 2})', "← add: 3", "3"]


def test_no_argument_call_and_structured_result_are_rendered():
    tool = Tool(name="ls", description="列目录", fn=lambda kwargs, ctx: ["a.py", "b.py"])
    registry = ToolRegistry([tool])
    provider = ScriptedProvider([tool_turn("toolu_1", "ls", {}), text_turn("两个文件")])

    result = runagent("列一下", provider, registry)

    assert result.trace == ['→ ls()', '← ls: ["a.py", "b.py"]', "两个文件"]


def test_tool_call_is_executed_and_fed_back(registry):
    provider = ScriptedProvider([tool_turn("toolu_1", "add", {"a": 1, "b": 2}), text_turn("3")])

    result = runagent("1+2 等于几", provider, registry)

    assert result.final == "3"
    assert [m["role"] for m in result.messages] == ["user", "assistant", "user", "assistant"]


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


def test_without_a_registry_the_default_tools_are_used():
    provider = ScriptedProvider([text_turn("你好")])

    runagent("你好", provider)

    assert provider.calls[0]["tools"] == default_tools().schemas()


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


def test_running_out_of_steps_asks_the_model_to_wrap_up(registry):
    """用满步数后不再打一句固定话，而是再问一次模型、把它的回复当 final。"""
    script = [tool_turn(f"toolu_{i}", "add", {"a": i, "b": i}) for i in range(3)]
    provider = ScriptedProvider([*script, text_turn("就这样吧")])

    result = runagent("没完没了", provider, registry, max_steps=3)

    assert result.final == "就这样吧"
    assert result.trace[-1] == "就这样吧"
    assert len(provider.calls) == 4  # 三次工具 + 收尾那次
    # 收尾那次不该带工具：模型手里没工具可调，就只能出文本。
    assert provider.calls[-1]["tools"] is None


class RecordingPrompter:
    """记下问了什么、按剧本回答。四个方法都得有——harness 按工具类别挑一个调。"""

    def __init__(
        self, allowed: bool = True, remember: bool = False, note: str = ""
    ) -> None:
        self.allowed = allowed
        self.remember = remember
        self.note = note
        self.tools: list[PermissionRequest] = []

    def confirm_command(self, command: str) -> Answer:
        # harness 现在不直接调这一条：bash 也走 confirm_tool，那行命令由 prompter 自己
        # 认出来（见 prompting.py）。协议要求实现它，所以留着。
        return self._answer()

    def confirm_tool(self, request: PermissionRequest) -> Answer:
        self.tools.append(request)
        return self._answer()

    def confirm_edit(self, request: PermissionRequest) -> Answer:
        self.tools.append(request)
        return self._answer()

    def _answer(self) -> Answer:
        return Answer(self.allowed, remember=self.remember, note=self.note)

    def single_choice(self, question: str, options: list[str]) -> str:
        return options[0]


def gated_registry(name: str, log: list[str]) -> ToolRegistry:
    """只装一个工具的注册表，真跑过就在 log 里记一笔。"""

    def fn(kwargs: dict, ctx: ToolContext) -> str:
        log.append(name)
        return "跑过了"

    return ToolRegistry([Tool(name=name, description="", fn=fn, params={})])


def blocked_blocks(result) -> list[dict]:
    """回给模型的那条 tool_result。"""
    return result.messages[2]["content"]


def test_a_read_only_tool_is_not_asked():
    log: list[str] = []
    prompter = RecordingPrompter()
    provider = ScriptedProvider([tool_turn("t1", "read_file", {}), text_turn("好了")])

    runagent("读一下", provider, gated_registry("read_file", log), prompter=prompter)

    assert log == ["read_file"]
    assert prompter.tools == []


def test_an_edit_tool_runs_without_asking_in_accept_edits_mode():
    log: list[str] = []
    prompter = RecordingPrompter()
    provider = ScriptedProvider([tool_turn("t1", "edit_file", {}), text_turn("改好了")])

    runagent(
        "改一下",
        provider,
        gated_registry("edit_file", log),
        permission_mode="accept edits",
        prompter=prompter,
    )

    assert log == ["edit_file"]
    assert prompter.tools == []


def test_an_edit_tool_is_denied_in_plan_mode():
    log: list[str] = []
    prompter = RecordingPrompter()
    provider = ScriptedProvider([tool_turn("t1", "edit_file", {}), text_turn("那我不改了")])

    result = runagent(
        "改一下",
        provider,
        gated_registry("edit_file", log),
        permission_mode="plan",
        prompter=prompter,
    )

    assert log == []  # 工具根本没跑
    assert prompter.tools == []  # 也没问——策略直接拒了
    assert result.trace[1].startswith("✗ edit_file: ")
    assert "plan" in result.trace[1]
    assert blocked_blocks(result)[0]["is_error"] is True
    assert result.final == "那我不改了"  # 模型收到了拒绝，接着把话说完


def test_default_mode_asks_before_editing_and_runs_when_allowed():
    log: list[str] = []
    prompter = RecordingPrompter(allowed=True)
    provider = ScriptedProvider(
        [tool_turn("t1", "edit_file", {"path": "a.txt"}), text_turn("改好了")]
    )

    runagent("改一下", provider, gated_registry("edit_file", log), prompter=prompter)

    assert [r.tool_name for r in prompter.tools] == ["edit_file"]
    assert log == ["edit_file"]


def test_a_no_from_the_user_stops_the_tool():
    log: list[str] = []
    prompter = RecordingPrompter(allowed=False)
    provider = ScriptedProvider(
        [tool_turn("t1", "edit_file", {"path": "a.txt"}), text_turn("好，不动它")]
    )

    result = runagent("改一下", provider, gated_registry("edit_file", log), prompter=prompter)

    assert log == []
    assert blocked_blocks(result)[0]["is_error"] is True
    assert "用户拒绝" in result.trace[1]


def test_bash_always_asks_even_in_accept_edits_mode():
    """档位管的是改文件那类动作，管不到跑命令。"""
    log: list[str] = []
    prompter = RecordingPrompter(allowed=True)
    provider = ScriptedProvider(
        [tool_turn("t1", "bash", {"command": "pytest -q"}), text_turn("跑完了")]
    )

    runagent(
        "跑个测试",
        provider,
        gated_registry("bash", log),
        allow_shell=True,
        permission_mode="accept edits",
        prompter=prompter,
    )

    assert [r.tool_name for r in prompter.tools] == ["bash"]
    assert log == ["bash"]


def test_bash_is_not_even_asked_when_shell_is_off():
    """没开 shell 的话它本来就跑不了。先弹一句询问、答完才发现跑不了，那是白问。"""
    log: list[str] = []
    prompter = RecordingPrompter(allowed=True)
    provider = ScriptedProvider([tool_turn("t1", "bash", {"command": "ls"}), text_turn("好")])

    runagent("跑个测试", provider, gated_registry("bash", log), prompter=prompter)

    assert prompter.tools == []  # 没问
    assert log == ["bash"]  # 工具照跑，"没开命令执行"那句话由它自己说


def test_bypass_permissions_asks_nothing():
    """唯一一档不把用户放进回路。那个 prompter 会答"不准"，但它不该轮得到被问。"""
    log: list[str] = []
    prompter = RecordingPrompter(allowed=False)
    provider = ScriptedProvider([tool_turn("t1", "edit_file", {}), text_turn("好")])

    runagent(
        "改一下",
        provider,
        gated_registry("edit_file", log),
        permission_mode="bypass permissions",
        prompter=prompter,
    )

    assert prompter.tools == []
    assert log == ["edit_file"]


def test_bypass_permissions_lets_commands_through_too():
    log: list[str] = []
    prompter = RecordingPrompter(allowed=False)
    provider = ScriptedProvider(
        [tool_turn("t1", "bash", {"command": "pytest -q"}), text_turn("好")]
    )

    runagent(
        "跑个测试",
        provider,
        gated_registry("bash", log),
        allow_shell=True,
        permission_mode="bypass permissions",
        prompter=prompter,
    )

    assert prompter.tools == []
    assert log == ["bash"]


def test_without_a_prompter_the_ask_path_denies():
    """非交互的跑法不该挂在提示符上：没人可问，就当拒绝。"""
    log: list[str] = []
    provider = ScriptedProvider([tool_turn("t1", "edit_file", {}), text_turn("好")])

    result = runagent("改一下", provider, gated_registry("edit_file", log))

    assert log == []
    assert blocked_blocks(result)[0]["is_error"] is True


def test_the_model_can_ask_the_user_and_gets_the_answer_back():
    """ask_user 就是个普通工具：问出来的答案按 tool_result 回去，模型接着往下走。"""
    prompter = RecordingPrompter()
    registry = ToolRegistry([make_ask_user(prompter)])
    provider = ScriptedProvider(
        [
            tool_turn("t1", "ask_user", {"question": "用哪个？", "options": ["选项A", "选项B"]}),
            text_turn("好"),
        ]
    )

    result = runagent("帮我拿个主意", provider, registry, prompter=prompter)

    assert result.trace[1] == "← ask_user: 选项A"  # RecordingPrompter 挑第一个
    assert result.final == "好"


class TestGrants:
    """本会话批过的调用不再问——粒度是 (工具, 路径)。"""

    def two_edits(self, first: str, second: str) -> ScriptedProvider:
        return ScriptedProvider(
            [
                tool_turn("t1", "edit_file", {"path": first}),
                tool_turn("t2", "edit_file", {"path": second}),
                text_turn("好"),
            ]
        )

    def test_a_remembered_file_is_not_asked_again(self):
        log: list[str] = []
        prompter = RecordingPrompter(allowed=True, remember=True)

        runagent(
            "改两次",
            self.two_edits("a.txt", "a.txt"),
            gated_registry("edit_file", log),
            prompter=prompter,
            max_steps=3,
        )

        assert len(prompter.tools) == 1  # 只问了一次
        assert log == ["edit_file", "edit_file"]  # 两次都真跑了

    def test_the_grant_covers_that_file_only(self):
        log: list[str] = []
        prompter = RecordingPrompter(allowed=True, remember=True)

        runagent(
            "改两个文件",
            self.two_edits("a.txt", "b.txt"),
            gated_registry("edit_file", log),
            prompter=prompter,
            max_steps=3,
        )

        assert [r.tool_args["path"] for r in prompter.tools] == ["a.txt", "b.txt"]

    def test_the_path_is_normalized_before_remembering(self):
        """a.txt / ./a.txt / src/../a.txt 是同一个文件，不该为写法差异再问一次。"""
        log: list[str] = []
        prompter = RecordingPrompter(allowed=True, remember=True)

        runagent(
            "改同一个文件",
            self.two_edits("a.txt", "./a.txt"),
            gated_registry("edit_file", log),
            prompter=prompter,
            max_steps=3,
        )

        assert len(prompter.tools) == 1

    def test_a_grant_cannot_override_plan_mode(self):
        """plan 是 deny，Grants 只能把 ask 变成 allow——翻不了盘。"""
        log: list[str] = []
        prompter = RecordingPrompter(allowed=True, remember=True)
        provider = ScriptedProvider(
            [tool_turn("t1", "edit_file", {"path": "a.txt"}), text_turn("好")]
        )

        result = runagent(
            "改一下",
            provider,
            gated_registry("edit_file", log),
            permission_mode="plan",
            prompter=prompter,
        )

        assert log == []
        assert prompter.tools == []  # 连问都没问
        assert blocked_blocks(result)[0]["is_error"] is True

    def test_grants_do_not_leak_between_runs(self):
        """不传 grants 就是空的：两次 runagent 各问各的（记忆得由调用方带着走）。"""
        log: list[str] = []
        prompter = RecordingPrompter(allowed=True, remember=True)

        for _ in range(2):
            runagent(
                "改一下",
                self.two_edits("a.txt", "a.txt"),
                gated_registry("edit_file", log),
                prompter=prompter,
                max_steps=2,
            )

        assert len(prompter.tools) == 2

    def test_a_shared_grants_object_carries_across_runs(self):
        """传同一份 Grants 就不问了——CLI 就是建一次、每轮传同一份。"""
        log: list[str] = []
        grants = Grants()
        prompter = RecordingPrompter(allowed=True, remember=True)

        for _ in range(2):
            runagent(
                "改一下",
                self.two_edits("a.txt", "a.txt"),
                gated_registry("edit_file", log),
                prompter=prompter,
                grants=grants,
                max_steps=2,
            )

        assert len(prompter.tools) == 1

    def test_a_command_is_never_remembered(self):
        """bash 那条没有"别再问"这一档：按整条命令记没意义，按程序名记又太粗。"""
        grants = Grants()
        command = PermissionRequest(tool_name="bash", tool_args={"command": "ls"})

        grants.remember(command)

        assert grants.covers(command) is False

    def test_a_refusal_does_not_grant_anything(self):
        """记住的前提是放行：一个"拒绝但记住"的答案不该把文件记进去。"""
        log: list[str] = []
        grants = Grants()
        prompter = RecordingPrompter(allowed=False, remember=True)
        provider = ScriptedProvider(
            [tool_turn("t1", "edit_file", {"path": "a.txt"}), text_turn("好")]
        )

        runagent(
            "改一下",
            provider,
            gated_registry("edit_file", log),
            prompter=prompter,
            grants=grants,
        )

        assert grants.entries == set()
        assert log == []

    def test_the_users_reason_is_carried_back_to_the_model(self):
        """用户写下的理由比我们替他猜的准，原样回给模型。"""
        log: list[str] = []
        prompter = RecordingPrompter(allowed=False, note="改 bar.py 而不是这个")
        provider = ScriptedProvider(
            [tool_turn("t1", "edit_file", {"path": "a.txt"}), text_turn("好")]
        )

        result = runagent("改一下", provider, gated_registry("edit_file", log), prompter=prompter)

        assert "改 bar.py 而不是这个" in result.trace[1]
        assert log == []


class TestMultiTurn:
    """把上一次的 result.messages 传回来，就接着聊。"""

    def test_the_second_turn_carries_the_first(self, registry):
        first = ScriptedProvider([text_turn("记住了")])
        result = runagent("我叫什么来着？", first, registry)

        second = ScriptedProvider([text_turn("那我记着")])
        runagent("那再说一遍", second, registry, messages=result.messages)

        sent = second.calls[0]["messages"]
        assert [m["role"] for m in sent] == ["user", "assistant", "user"]
        assert sent[0]["content"] == "我叫什么来着？"
        assert sent[1]["content"][0]["text"] == "记住了"
        assert sent[2]["content"] == "那再说一遍"

    def test_the_history_handed_in_is_not_touched(self, registry):
        """传进去的那份不该被就地改——调用方还指着它呢。"""
        result = runagent("第一句", ScriptedProvider([text_turn("好")]), registry)
        before = list(result.messages)

        runagent("第二句", ScriptedProvider([text_turn("嗯")]), registry, messages=result.messages)

        assert result.messages == before

    def test_without_history_it_starts_fresh(self, registry):
        provider = ScriptedProvider([text_turn("好")])

        runagent("就这一句", provider, registry)

        assert [m["role"] for m in provider.calls[0]["messages"]] == ["user"]


def folded_blocks(messages: list[dict]) -> list[dict]:
    """这份 messages 里被折掉的工具结果。"""
    return [
        block
        for message in messages
        if isinstance(message.get("content"), list)
        for block in message["content"]
        if block.get("type") == "tool_result" and "折叠" in str(block.get("content"))
    ]


class TestTrimming:
    """旧的、长的工具结果发出去之前折掉——但留下的那份还是完整的。"""

    def a_dump_registry(self) -> ToolRegistry:
        def dump(kwargs: dict, ctx: ToolContext) -> str:
            return "x" * 3000

        return ToolRegistry([Tool(name="dump", description="", fn=dump, params={})])

    def nine_dumps(self) -> ScriptedProvider:
        return ScriptedProvider(
            [*(tool_turn(f"t{i}", "dump", {}) for i in range(9)), text_turn("好了")]
        )

    def test_what_is_sent_is_folded_but_what_is_kept_is_whole(self):
        provider = self.nine_dumps()

        result = runagent("读九次", provider, self.a_dump_registry(), max_steps=9)

        # 发出去的最后那次里，最旧的那条折了（一共九条，最近八条留着）
        assert len(folded_blocks(provider.calls[-1]["messages"])) == 1
        # 而 returned 的这份（也就是落盘的那份）一条都没折
        assert folded_blocks(result.messages) == []
        assert result.messages[2]["content"][0]["content"] == "x" * 3000

    def test_nothing_is_folded_while_the_history_is_short(self):
        provider = self.nine_dumps()

        runagent("读九次", provider, self.a_dump_registry(), max_steps=9)

        # 第八次请求时手上只有八条结果，正好是"最近八条"，一条都不该折
        assert folded_blocks(provider.calls[7]["messages"]) == []


def test_a_dangerous_command_is_refused_even_after_the_user_says_yes():
    """权限闸门管"要不要问"、工具自己的闸门管"能不能做"——点了头也能被拒。

    顺序就是这么定的：先问用户（同意或拒绝都是他的决定），然后工具再判。好处是用户能在
    trace 里看见模型试过什么；代价是他可能白点一次头。
    """
    prompter = RecordingPrompter(allowed=True)
    provider = ScriptedProvider(
        [tool_turn("t1", "bash", {"command": "rm -rf /"}), text_turn("好")]
    )
    registry = ToolRegistry([Tool(name="bash", description="", fn=bash, params=SHELL_SCHEMA)])

    result = runagent(
        "删掉根目录", provider, registry, allow_shell=True, prompter=prompter, max_steps=1
    )

    assert [r.tool_name for r in prompter.tools] == ["bash"]  # 先问过用户
    assert result.trace[1].startswith("✗ bash: ")  # 但工具照样拒
    assert "黑名单" in result.trace[1]


def test_the_usage_line_is_printed_for_every_request(capsys):
    """打这一行是为了看缓存有没有省到钱：后端是自动前缀缓存，命中多少只看 cache_read。

    一轮里**每一发**都打，因为要看的是它们之间的变化——正常情况下后一发比前一发的开头长，
    命中该只增不减；中间要是被什么改动了开头（折叠旧工具结果改的正好是消息列表前排），命中
    会从那个位置往后掉成 0。
    """
    provider = ScriptedProvider(
        [
            ModelResponse(
                text="好",
                assistant_content=[{"type": "text", "text": "好"}],
                stop_reason="end_turn",
                input_tokens=100,
                cache_read_input_tokens=900,
            )
        ]
    )

    runagent("问一句", provider, ToolRegistry([]))

    out = capsys.readouterr().out
    assert "缓存 900" in out
    assert "原价 100" in out


def test_no_usage_line_when_the_backend_reports_nothing(capsys):
    """两个数都是 0 就不吭声——那是假 provider，不是"命中 0"，别拿噪音骗自己。"""
    provider = ScriptedProvider([text_turn("好")])

    runagent("问一句", provider, ToolRegistry([]))

    assert "输入 token" not in capsys.readouterr().out
