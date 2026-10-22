"""Tool / ToolResult / ToolRegistry 的单元测试，全部本地执行，不发网络请求。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mini_code_agent.memory import Memory, write
from mini_code_agent.providers.base import ToolCall
from mini_code_agent.tools import (
    Tool,
    ToolContext,
    ToolRegistry,
    ToolResult,
    default_tools,
    spill,
    web,
)
from mini_code_agent.tools.safety import MAX_TOOL_CHARS
from mini_code_agent.tools.spill import TOOL_RESULTS_DIR

ADD_SCHEMA = {
    "type": "object",
    "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
    "required": ["a", "b"],
}


# 工具函数现在收两样：模型给的参数字典，和一个 ToolContext。
def add(kwargs: dict, ctx: ToolContext) -> int:
    return kwargs["a"] + kwargs["b"]


def boom(kwargs: dict, ctx: ToolContext) -> str:
    raise RuntimeError("工具内部炸了")


def echo(kwargs: dict, ctx: ToolContext) -> dict:
    return kwargs.get("payload") or {}


def make_add() -> Tool:
    return Tool(name="add", description="把两个整数相加", fn=add, params=ADD_SCHEMA)


@pytest.fixture
def ctx(tmp_path: Path) -> ToolContext:
    return ToolContext(cwd=tmp_path)


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
        tool = Tool(name="now", description="返回当前时间", fn=lambda kwargs, ctx: "now")
        assert tool.to_anthropic_schema()["input_schema"] == {
            "type": "object",
            "properties": {},
            "required": [],
        }

    def test_missing_description_is_omitted_instead_of_sent_as_null(self):
        tool = Tool(name="now", fn=lambda kwargs, ctx: "now")
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

    def test_run_calls_the_underlying_function(self, registry, ctx):
        assert registry.run(ctx, "add", {"a": 2, "b": 3}) == 5

    def test_run_without_arguments(self, registry, ctx):
        assert registry.run(ctx, "echo") == {}

    def test_run_unknown_tool_raises_key_error(self, registry, ctx):
        with pytest.raises(KeyError, match="nope"):
            registry.run(ctx, "nope")

    def test_run_rejects_a_missing_required_argument(self, registry, ctx):
        """缺必填参数要在派发前就拦住。

        放过去的话工具会拿着 None 悄悄跑完，模型收到一个看起来正常的结果，还以为
        自己成功了——它本来能收到一句"你少传了什么"然后自己改对。
        """
        with pytest.raises(ValueError, match="缺少必填参数"):
            registry.run(ctx, "add", {"a": 1})

    def test_schemas_lists_every_tool(self, registry, ctx):
        assert [schema["name"] for schema in registry.schemas(ctx)] == ["add", "boom", "echo"]

    def test_dispatch_wraps_the_return_value_in_a_tool_result(self, registry, ctx):
        call = ToolCall(id="toolu_1", name="add", arguments={"a": 1, "b": 2})

        result = registry.dispatch(ctx, call)

        assert result == ToolResult(id="toolu_1", name="add", result=3)
        assert result.to_anthropic_block()["content"] == "3"

    def test_dispatch_turns_tool_errors_into_results(self, registry, ctx):
        result = registry.dispatch(ctx, ToolCall(id="toolu_3", name="boom", arguments={}))

        assert result.is_error is True
        assert "RuntimeError" in result.result

    def test_dispatch_unknown_tool_is_a_result_not_an_exception(self, registry, ctx):
        result = registry.dispatch(ctx, ToolCall(id="toolu_4", name="nope", arguments={}))

        assert result.is_error is True
        assert "KeyError" in result.result

    def test_dispatch_hands_a_missing_argument_back_to_the_model(self, registry, ctx):
        """缺参数和工具自己抛错走的是同一条路：变成 is_error 的结果交回模型。"""
        result = registry.dispatch(ctx, ToolCall(id="toolu_5", name="add", arguments={"a": 1}))

        assert result.is_error is True
        assert "缺少必填参数" in result.result


class TestDefaultTools:
    def test_registers_the_builtin_tools(self):
        assert [tool.name for tool in default_tools().list()] == [
            "list_files",
            "project_tree",
            "read_file",
            "read_memory",
            "remember",
            "read_skill",
            "read_skill_file",
            "write_file",
            "edit_file",
            "replace_lines",
            "bash",
            "glob",
            "grep",
            "web_search",
            "web_fetch",
            # 取落盘结果的那个**排在最后**：工具表是请求前缀的一部分（自动前缀缓存按前缀
            # 命中），往末尾追加不会让它前面那些的缓存作废，插在中间会。
            "read_tool_result",
        ]

    def test_each_call_returns_a_fresh_registry(self):
        assert default_tools() is not default_tools()

    def test_the_read_only_ones_are_exactly_these(self):
        """子 agent 里 explore / plan 两个类型只有这批工具（见 core/subagent.py）。

        断言的是**整份名单**，不是"这几个在里面"：加一个新工具、忘了想它算不算只读，
        这条就会红，逼人做那个决定。理由见 known-issues F1——靠名字列名单会漂，
        而这里连"名单"都没有，全靠每处 `Tool(...)` 自己标。
        """
        read_only = {tool.name for tool in default_tools().list() if tool.read_only}

        assert read_only == {
            "list_files",
            "project_tree",
            "read_file",
            "read_memory",
            "read_skill",
            "read_skill_file",
            "glob",
            "grep",
            "read_tool_result",
            "web_search",
            "web_fetch",
        }

    def test_the_ones_that_change_something_are_not_read_only(self):
        """剩下这五个都会改变点什么：写盘、跑命令、改长期记忆。"""
        changes_something = {
            tool.name for tool in default_tools().list() if not tool.read_only
        }

        assert changes_something == {
            "remember",
            "write_file",
            "edit_file",
            "replace_lines",
            "bash",
        }

    def test_the_foldable_ones_are_exactly_these(self):
        """缓存冷了的时候能折的只有这几个（见 folding.py），**其余一律留着**。

        判据是"又大、又能重新拿到"：文件能重读、搜索能重跑、命令和网页太长时本来就落过盘
        （头一段里写着取回的 id）。断言整份名单而不是"这几个在里面"——加一个新工具、忘了想
        它能不能折，这条就会红、逼人做那个决定。

        **默认是"不折"**，所以漏标的那一侧是安全的：少省一点，不是丢东西。
        """
        foldable = {tool.name for tool in default_tools().list() if tool.foldable}

        assert foldable == {
            "list_files",
            "read_file",
            "glob",
            "grep",
            "web_search",
            "web_fetch",
            "bash",
        }

    def test_what_carries_conclusions_is_not_foldable(self):
        """这几样折了就真丢了：项目概览是"我现在在哪儿"的参照，记忆和技能是模型**特意去取**
        回来的，`read_tool_result` 取的也是它主动要的原文。

        子 agent 那个工具（`agent`）不在这张表里——它在 CLI 层注入，压根没标 `foldable`。
        它另有测试守着（见 tests/test_subagent.py）。
        """
        foldable = {tool.name for tool in default_tools().list() if tool.foldable}

        assert not foldable & {
            "project_tree",
            "read_memory",
            "read_skill",
            "read_skill_file",
            "read_tool_result",
        }

    def test_a_later_registered_tool_is_not_foldable(self):
        """`foldable_names()` 是**现算**的，所以后注册的工具没标就不在内。

        CLI 注入的 `agent` / `ask_user` 走的就是这条路——不必在折叠那边另写一份"这些不许
        折"的名单，也就不存在两处名单漂移的问题。
        """
        registry = default_tools()
        registry.register(Tool(name="agent", fn=lambda kwargs, ctx: ""))

        assert "agent" not in registry.foldable_names()
        assert "read_file" in registry.foldable_names()

    def test_the_fold_persist_ones_are_exactly_these(self):
        """折叠时"重跑拿不回来"、所以必须先落一份的，只有这三个。

        判据是**重跑一次未必等价**：命令有副作用、网页会变、搜索花钱还会漂。`read_file` /
        `grep` / `glob` / `list_files` 不在里面——重跑免费，而且拿到的还是当前内容。
        """
        persist = {tool.name for tool in default_tools().list() if tool.fold_persist}

        assert persist == {"bash", "web_search", "web_fetch"}

    def test_fold_persist_implies_foldable(self):
        """不折的工具根本走不到那条路上，标了就是配错了——直接报出来，别静默失效。"""
        with pytest.raises(ValueError):
            Tool(name="x", fn=lambda kwargs, ctx: "", fold_persist=True)

    def test_a_later_registered_tool_is_not_fold_persisted(self):
        registry = default_tools()
        registry.register(Tool(name="agent", fn=lambda kwargs, ctx: "", spill=True))

        assert "agent" not in registry.fold_persist_names()
        assert "bash" in registry.fold_persist_names()


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
    def test_formats_title_url_and_snippet(self, fake_tavily, ctx):
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

        text = web.web_search({"query": "Leo Messi"}, ctx)

        assert "[1] Lionel Messi - Wikipedia" in text
        assert "https://en.wikipedia.org/wiki/Lionel_Messi" in text
        assert "Argentine footballer." in text

    def test_drops_the_noise_tavily_returns(self, fake_tavily, ctx):
        fake_tavily.response = {
            "request_id": "abc",
            "response_time": 1.23,
            "images": ["nope.jpg"],
            "results": [{"title": "t", "url": "u", "content": "c"}],
        }

        text = web.web_search({"query": "x"}, ctx)

        assert "abc" not in text
        assert "nope.jpg" not in text

    def test_passes_query_and_max_results_to_tavily(self, fake_tavily, ctx):
        web.web_search({"query": "python 3.13", "max_results": 3}, ctx)

        assert fake_tavily.calls == [{"query": "python 3.13", "max_results": 3}]

    def test_clamps_max_results_to_the_cap(self, fake_tavily, ctx):
        web.web_search({"query": "x", "max_results": 999}, ctx)

        assert fake_tavily.calls[0]["max_results"] == web.MAX_RESULTS_LIMIT

    def test_says_so_when_nothing_matched(self, fake_tavily, ctx):
        assert "没有搜到" in web.web_search({"query": "绝不可能存在的词"}, ctx)

    def test_missing_api_key_gives_an_actionable_error(self, monkeypatch):
        monkeypatch.delenv("TAVILY_API_KEY", raising=False)

        with pytest.raises(RuntimeError, match="TAVILY_API_KEY"):
            web._client()

    def test_client_transport_error_reaches_the_model_as_a_result(self, fake_tavily, ctx):
        fake_tavily.error = RuntimeError("rate limited")

        result = default_tools().dispatch(
            ctx, ToolCall(id="t1", name="web_search", arguments={"query": "x"})
        )

        assert result.is_error is True
        assert "rate limited" in result.result


class TestToolFiltering:
    """"此刻用不上的不发"——广告一个做不到的能力，代价是模型试一次、吃一条报错、花掉一步。"""

    def names(self, ctx) -> list[str]:
        return [schema["name"] for schema in default_tools().schemas(ctx)]

    def test_bash_is_not_advertised_without_shell(self, tmp_path):
        """没开 --shell 时它调了也只会拿到一句"没开命令执行"，而那一下是要花掉一步的。"""
        assert "bash" not in self.names(ToolContext(tmp_path))
        assert "bash" in self.names(ToolContext(tmp_path, allow_shell=True))

    def test_web_tools_are_not_advertised_without_a_key(self, tmp_path, monkeypatch):
        monkeypatch.delenv("TAVILY_API_KEY", raising=False)

        names = self.names(ToolContext(tmp_path))

        assert "web_search" not in names
        assert "web_fetch" not in names

    def test_web_tools_show_up_once_the_key_is_there(self, tmp_path, monkeypatch):
        monkeypatch.setenv("TAVILY_API_KEY", "k")

        assert "web_search" in self.names(ToolContext(tmp_path))

    def test_skill_tools_are_not_advertised_without_any_skill(self, tmp_path):
        names = self.names(ToolContext(tmp_path))

        assert "read_skill" not in names
        assert "read_skill_file" not in names

    def test_skill_tools_show_up_once_there_is_one(self, tmp_path):
        directory = tmp_path / ".agent" / "skill" / "release"
        directory.mkdir(parents=True)
        (directory / "SKILL.md").write_text(
            "---\ndescription: 发版前走一遍\n---\n\n正文\n", encoding="utf-8"
        )

        names = self.names(ToolContext(tmp_path))

        assert "read_skill" in names
        assert "read_skill_file" in names

    def test_the_memory_tool_is_not_advertised_without_any_memory(self, tmp_path):
        assert "read_memory" not in self.names(ToolContext(tmp_path))

    def test_the_memory_tool_shows_up_once_there_is_one(self, tmp_path):
        write(tmp_path, Memory(name="a-note", description="一条", type="user", body="正文"))

        assert "read_memory" in self.names(ToolContext(tmp_path))

    def test_writing_a_memory_is_always_advertised(self, tmp_path):
        """记一条永远可以——没有可读的，也能写第一条。"""
        assert "remember" in self.names(ToolContext(tmp_path))

    def test_the_always_available_ones_are_untouched(self, tmp_path):
        names = self.names(ToolContext(tmp_path))

        for name in ("read_file", "write_file", "edit_file", "grep", "glob"):
            assert name in names

    def test_a_tool_without_available_is_always_sent(self, ctx):
        """默认是"一直能用"——绝大多数工具都走这一条。"""
        registry = ToolRegistry([Tool(name="x", fn=lambda kwargs, ctx: "ok")])

        assert [schema["name"] for schema in registry.schemas(ctx)] == ["x"]

    def test_a_tool_can_hide_itself(self, ctx):
        registry = ToolRegistry(
            [Tool(name="x", fn=lambda kwargs, ctx: "ok", available=lambda _: False)]
        )

        assert registry.schemas(ctx) == []


class TestSpilling:
    """标了 spill 的工具：结果太长就整篇落盘，上下文里只留预览。

    为什么只有标了的才落盘：截掉了就真没了的那几个工具（命令有副作用、网页会变、
    取一次要花钱），落盘才有意义。read_file 那一类不落——它的内容本来就是盘上那个文件，
    `start_line` / `end_line` 能读回来，而读副本还会把 read_state 那条线掐断。
    """

    LONG = MAX_TOOL_CHARS + 1

    def registry(self, fn, *, spill: bool) -> ToolRegistry:
        return ToolRegistry([Tool(name="chatty", fn=fn, spill=spill)])

    def call(self, registry: ToolRegistry, ctx: ToolContext) -> ToolResult:
        return registry.dispatch(ctx, ToolCall(id="t1", name="chatty", arguments={}))

    def spill_files(self, ctx: ToolContext) -> list[Path]:
        return sorted((ctx.cwd / TOOL_RESULTS_DIR).glob("*.txt"))

    def test_a_long_result_lands_on_disk_whole(self, ctx):
        text = "行\n" * self.LONG

        result = self.call(self.registry(lambda kwargs, c: text, spill=True), ctx)

        saved = self.spill_files(ctx)
        assert len(saved) == 1
        assert saved[0].read_text(encoding="utf-8") == text  # 一个字都没少
        # 文件名就是那次工具调用的 id——同一个 id 反复落盘是同一个文件
        assert saved[0].stem == "t1"
        assert 'id 传 "t1"' in result.result

    def test_the_context_only_gets_a_preview(self, ctx):
        result = self.call(
            self.registry(lambda kwargs, c: "x" * self.LONG, spill=True), ctx
        )

        assert result.result.startswith("[输出太长")
        assert "read_tool_result" in result.result
        assert len(result.result) < MAX_TOOL_CHARS

    def test_the_message_says_how_many_lines_there_are(self, ctx):
        """真正要用的常在末尾，而模型只看得到开头——不告诉它多少行，它没法直接翻到最后。"""
        text = "行\n" * self.LONG
        lines = text.count("\n") + 1

        result = self.call(self.registry(lambda kwargs, c: text, spill=True), ctx)

        assert f"{lines:,} 行" in result.result

    def test_a_short_result_is_not_spilled(self, ctx):
        result = self.call(self.registry(lambda kwargs, c: "短", spill=True), ctx)

        assert result.result == "短"
        assert self.spill_files(ctx) == []

    def test_a_tool_without_the_flag_is_left_alone(self, ctx):
        """没标 spill 的照旧原样返回——截断是那些工具**自己**的事（各有各的说明）。"""
        result = self.call(
            self.registry(lambda kwargs, c: "x" * self.LONG, spill=False), ctx
        )

        assert len(result.result) == self.LONG
        assert self.spill_files(ctx) == []

    def test_a_non_string_result_is_left_alone(self, ctx):
        """只有文本才谈得上落盘；结构化结果由 to_anthropic_block 那边自己序列化。"""
        result = self.call(self.registry(lambda kwargs, c: {"a": 1}, spill=True), ctx)

        assert result.result == {"a": 1}

    def test_a_failed_spill_falls_back_to_truncating(self, ctx, monkeypatch):
        """盘写不进去就退回普通限额——不能让一次落盘失败把几 MB 原样放进上下文。"""
        monkeypatch.setattr(spill, "save", lambda *args, **kwargs: None)

        result = self.call(
            self.registry(lambda kwargs, c: "x" * self.LONG, spill=True), ctx
        )

        assert len(result.result) < MAX_TOOL_CHARS + 100
        assert "已截断" in result.result

    def test_what_was_spilled_can_be_read_back(self, ctx):
        """落下去的和取回来的必须是同一份——不然这个机制就是在骗模型。"""
        text = "".join(f"第 {index} 行\n" for index in range(1, self.LONG + 1))

        spilled = self.call(self.registry(lambda kwargs, c: text, spill=True), ctx)
        name = spilled.result.split('id 传 "')[1].split('"')[0]

        back = default_tools().dispatch(
            ctx, ToolCall(id="t2", name="read_tool_result", arguments={"id": name})
        )

        assert back.is_error is False
        assert "1 | 第 1 行" in back.result
        assert "100 | 第 100 行" in back.result
    def test_it_spills_into_the_session_directory_when_one_is_given(self, tmp_path):
        """CLI 传的是**这次会话的目录**——删会话时能跟着一起端掉（见 tools/spill.py）。"""
        session_dir = tmp_path / ".agent" / "tool-results" / "20260101-000000-0001abcd"
        ctx = ToolContext(cwd=tmp_path, spill_dir=session_dir)

        result = self.call(self.registry(lambda kwargs, c: "x" * self.LONG, spill=True), ctx)

        saved = list(session_dir.glob("*.txt"))
        assert len(saved) == 1
        assert saved[0].stem == "t1"
        assert saved[0].read_text(encoding="utf-8") == "x" * self.LONG
        # 给模型看的那条路径要指到会话目录里去，不能只说个文件名
        assert ".agent/tool-results/20260101-000000-0001abcd/t1.txt" in result.result

    def test_an_ill_shaped_id_falls_back_to_the_old_name(self, ctx):
        """`tool_use_id` 是**服务端响应**里的字符串，不能无条件拼进路径。

        形状不对就退回 `<工具名>-<摘要>`：照常落盘、照常取得回来，只是名字回到旧格式。
        卡的是写文件——读那边另有围栏，但写比读更糟，那边至少还有兜底。
        """
        registry = self.registry(lambda kwargs, c: "x" * self.LONG, spill=True)

        result = registry.dispatch(
            ctx, ToolCall(id="../../逃出去", name="chatty", arguments={})
        )

        saved = self.spill_files(ctx)
        assert len(saved) == 1
        assert saved[0].parent == ctx.cwd / TOOL_RESULTS_DIR
        assert saved[0].stem.startswith("chatty-")
        assert saved[0].stem in result.result

    def test_the_ill_shaped_id_fallback_is_deterministic(self, ctx):
        """回退名必须稳定：折叠每次都从没改过的会话文件重算，随机名会让
        "已经存在就不写"永不命中，每折一次多一个文件。
        """
        first = spill.save(ctx, "chatty", "../../x", "一份")
        second = spill.save(ctx, "chatty", "../../x", "另一份")

        assert first == second
        assert first is not None and first.stem.startswith("chatty-")

    def test_saving_the_same_id_twice_does_not_overwrite(self, ctx):
        """同一个 id 第二次落盘**什么都不做**——折叠会反复重算，不能每算一次就重写一遍。"""
        first = spill.save(ctx, "chatty", "t1", "第一份")
        assert first is not None
        first.write_text("被改过", encoding="utf-8")

        second = spill.save(ctx, "chatty", "t1", "第二份")

        assert second == first
        assert first.read_text(encoding="utf-8") == "被改过"

    def test_a_failed_save_into_returns_none(self, tmp_path, monkeypatch):
        def boom(*args, **kwargs):
            raise OSError("盘满了")

        monkeypatch.setattr(spill, "_write_atomic", boom)

        assert spill.save_into(tmp_path, "chatty", "t1", "正文") is None

    def test_display_path_is_relative_inside_and_absolute_outside(self, tmp_path):
        inside = tmp_path / ".agent" / "tool-results" / "s1" / "t1.txt"
        outside = tmp_path.parent / "elsewhere" / "t1.txt"

        assert spill.display_path(tmp_path, inside) == ".agent/tool-results/s1/t1.txt"
        assert spill.display_path(tmp_path, outside) == outside.as_posix()
