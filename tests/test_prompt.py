"""Prompter 协议的单元测试：缺省实现怎么答、能不能换一个进来，全部本地执行。"""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from mini_code_agent.tools.base import ToolContext
from mini_code_agent.tools.permissions import PermissionRequest
from mini_code_agent.tools.prompt import NobodyToAskError, NullPrompter


def request(name: str = "write_file") -> PermissionRequest:
    return PermissionRequest(tool_name=name, tool_args={"path": "a.txt"})


class TestNullPrompter:
    def test_it_answers_no_to_the_three_guards(self):
        prompter = NullPrompter()

        assert prompter.confirm_command("pytest -q") is False
        assert prompter.confirm_tool(request()) is False
        assert prompter.confirm_edit(request("edit_file")) is False

    def test_a_question_raises_instead_of_picking_an_answer(self):
        """没有缺省答案就不能替用户挑一个——那是替他做了决定。"""
        with pytest.raises(NobodyToAskError, match="没人可问"):
            NullPrompter().single_choice("用哪个实现？", ["a", "b"])

    def test_the_error_still_carries_the_question_and_the_options(self):
        """模型得看得见它到底想问什么，不然收到一句"没人可问"也不知道该问什么。"""
        with pytest.raises(NobodyToAskError) as info:
            NullPrompter().single_choice("用哪个实现？", ["a", "b"])

        message = str(info.value)
        assert "用哪个实现？" in message
        assert "a" in message
        assert "b" in message


class FakePrompter:
    """照 Protocol 写一个假的。确认那条路要能自动化，靠的就是这个位置换得掉。"""

    def __init__(self) -> None:
        self.asked: list[str] = []

    def confirm_command(self, command: str) -> bool:
        self.asked.append(command)
        return True

    def confirm_tool(self, request: PermissionRequest) -> bool:
        self.asked.append(request.tool_name)
        return True

    def confirm_edit(self, request: PermissionRequest) -> bool:
        self.asked.append(request.tool_name)
        return False

    def single_choice(self, question: str, options: Sequence[str]) -> str:
        self.asked.append(question)
        return options[0]


class TestToolContextWiring:
    def test_a_context_carries_a_null_prompter_by_default(self, tmp_path):
        assert isinstance(ToolContext(cwd=tmp_path).prompter, NullPrompter)

    def test_the_default_denies_instead_of_waiting_for_input(self, tmp_path):
        """缺省行为必须是"拒绝"，不是"等输入"——测试和一次性跑都指望这个。"""
        assert ToolContext(cwd=tmp_path).prompter.confirm_tool(request()) is False

    def test_a_context_takes_another_prompter(self, tmp_path):
        fake = FakePrompter()

        ctx = ToolContext(cwd=tmp_path, prompter=fake)

        assert ctx.prompter.confirm_command("pytest -q") is True
        assert fake.asked == ["pytest -q"]

    def test_two_contexts_do_not_share_one_prompter(self, tmp_path):
        first = ToolContext(cwd=tmp_path)
        second = ToolContext(cwd=tmp_path)

        assert first.prompter is not second.prompter
