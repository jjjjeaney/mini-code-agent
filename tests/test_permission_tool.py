"""set_permission_mode 工具：收紧不提问，放宽必须用户当次同意。全部本地执行。"""

from __future__ import annotations

from pathlib import Path

from mini_code_agent.permission_tool import make_set_permission_mode
from mini_code_agent.permissions import PERMISSION_MODES
from mini_code_agent.prompts import NobodyToAskError
from mini_code_agent.tools.base import ToolContext


class ChoicePrompter:
    """只实现 single_choice：这个工具只会问这一种。"""

    def __init__(self, answer: str = "拒绝", *, error: Exception | None = None) -> None:
        self.answer = answer
        self.error = error
        self.questions: list[tuple[str, tuple[str, ...]]] = []

    def single_choice(self, question: str, options) -> str:
        self.questions.append((question, tuple(options)))
        if self.error is not None:
            raise self.error
        return self.answer


class FakeState:
    def __init__(self, mode: str = "default") -> None:
        self.permission_mode = mode


def a_ctx(tmp_path: Path, mode: str = "default") -> ToolContext:
    return ToolContext(cwd=tmp_path, permission_mode=mode)


def switch(tool, ctx, mode: str) -> str:
    return tool.fn({"mode": mode}, ctx)


class TestMetadata:
    def test_the_schema_offers_all_four_modes(self):
        tool = make_set_permission_mode(ChoicePrompter())

        assert tool.name == "set_permission_mode"
        assert tool.params["properties"]["mode"]["enum"] == list(PERMISSION_MODES)

    def test_it_is_a_barrier_not_a_read_only_tool(self):
        assert make_set_permission_mode(ChoicePrompter()).read_only is False


class TestStricterSwitches:
    def test_switching_to_plan_needs_no_approval(self, tmp_path):
        prompter = ChoicePrompter()
        state = FakeState("default")
        ctx = a_ctx(tmp_path, "default")

        result = switch(make_set_permission_mode(prompter, state=state), ctx, "plan")

        assert prompter.questions == []
        assert ctx.permission_mode == "plan"
        assert state.permission_mode == "plan"
        assert "立即" in result
        assert "下一轮" in result
        assert "拒绝" in result

    def test_switching_from_bypass_to_default_needs_no_approval(self, tmp_path):
        prompter = ChoicePrompter()
        ctx = a_ctx(tmp_path, "bypass permissions")

        switch(make_set_permission_mode(prompter), ctx, "default")

        assert prompter.questions == []
        assert ctx.permission_mode == "default"

    def test_switching_to_the_same_mode_says_so(self, tmp_path):
        prompter = ChoicePrompter()
        ctx = a_ctx(tmp_path, "default")

        result = switch(make_set_permission_mode(prompter), ctx, "default")

        assert prompter.questions == []
        assert "已经是" in result
        assert ctx.permission_mode == "default"


class TestLoosening:
    def test_it_asks_and_applies_on_consent(self, tmp_path):
        prompter = ChoicePrompter("同意")
        state = FakeState("plan")
        ctx = a_ctx(tmp_path, "plan")

        result = switch(make_set_permission_mode(prompter, state=state), ctx, "default")

        assert len(prompter.questions) == 1
        assert prompter.questions[0][1] == ("同意", "拒绝")
        assert ctx.permission_mode == "default"
        assert state.permission_mode == "default"
        assert "用户同意" in result

    def test_a_refusal_keeps_the_old_mode(self, tmp_path):
        prompter = ChoicePrompter("拒绝")
        state = FakeState("plan")
        ctx = a_ctx(tmp_path, "plan")

        result = switch(make_set_permission_mode(prompter, state=state), ctx, "default")

        assert ctx.permission_mode == "plan"
        assert state.permission_mode == "plan"
        assert "没有同意" in result

    def test_free_text_is_a_refusal_with_a_note(self, tmp_path):
        prompter = ChoicePrompter("先别切")
        ctx = a_ctx(tmp_path, "plan")

        result = switch(make_set_permission_mode(prompter), ctx, "accept edits")

        assert ctx.permission_mode == "plan"
        assert "用户说：先别切" in result

    def test_nobody_to_ask_keeps_the_old_mode(self, tmp_path):
        prompter = ChoicePrompter(error=NobodyToAskError("没人可问"))
        ctx = a_ctx(tmp_path, "plan")

        result = switch(make_set_permission_mode(prompter), ctx, "default")

        assert ctx.permission_mode == "plan"
        assert "没法问你" in result

    def test_the_question_names_both_modes(self, tmp_path):
        prompter = ChoicePrompter("拒绝")

        switch(make_set_permission_mode(prompter), a_ctx(tmp_path, "plan"), "default")

        question = prompter.questions[0][0]
        assert "plan" in question
        assert "default" in question
        assert "放宽" in question


class TestBypass:
    def test_the_question_spells_out_the_risks(self, tmp_path):
        prompter = ChoicePrompter("拒绝")

        switch(
            make_set_permission_mode(prompter),
            a_ctx(tmp_path, "default"),
            "bypass permissions",
        )

        question = prompter.questions[0][0]
        assert "bypass permissions" in question
        assert ".env" in question
        assert "不会再问" in question
        assert "确定同意吗" in question

    def test_bypass_is_not_applied_without_consent(self, tmp_path):
        prompter = ChoicePrompter("拒绝")
        ctx = a_ctx(tmp_path, "default")

        switch(make_set_permission_mode(prompter), ctx, "bypass permissions")

        assert ctx.permission_mode == "default"

    def test_bypass_is_applied_after_consent(self, tmp_path):
        prompter = ChoicePrompter("同意")
        ctx = a_ctx(tmp_path, "default")

        switch(make_set_permission_mode(prompter), ctx, "bypass permissions")

        assert ctx.permission_mode == "bypass permissions"


class TestBadInput:
    def test_an_unknown_mode_is_refused_without_changing_anything(self, tmp_path):
        prompter = ChoicePrompter()
        state = FakeState("plan")
        ctx = a_ctx(tmp_path, "plan")

        result = switch(make_set_permission_mode(prompter, state=state), ctx, "yolo")

        assert "不是权限模式" in result
        assert prompter.questions == []
        assert ctx.permission_mode == "plan"
        assert state.permission_mode == "plan"

    def test_without_a_state_only_the_context_changes(self, tmp_path):
        ctx = a_ctx(tmp_path, "default")

        switch(make_set_permission_mode(ChoicePrompter()), ctx, "plan")

        assert ctx.permission_mode == "plan"
