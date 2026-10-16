"""Prompter 协议的单元测试：缺省实现怎么答、能不能换一个进来，全部本地执行。"""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from mini_code_agent.permissions import PermissionRequest
from mini_code_agent.prompts import Answer, NobodyToAskError, NullPrompter
from mini_code_agent.tools.base import ToolContext


def request(name: str = "write_file") -> PermissionRequest:
    return PermissionRequest(tool_name=name, tool_args={"path": "a.txt"})


class TestNullPrompter:
    def test_it_answers_no_to_the_guards(self):
        prompter = NullPrompter()

        assert prompter.confirm_tool(request()).allowed is False
        assert prompter.confirm_edit(request("edit_file")).allowed is False

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

    def confirm_tool(self, request: PermissionRequest) -> Answer:
        self.asked.append(request.tool_name)
        return Answer(True)

    def confirm_edit(self, request: PermissionRequest) -> Answer:
        self.asked.append(request.tool_name)
        return Answer(False)

    def single_choice(self, question: str, options: Sequence[str]) -> str:
        self.asked.append(question)
        return options[0]


class TestItIsNotPartOfTheToolLayer:
    def test_the_tool_context_does_not_carry_a_prompter(self, tmp_path):
        """问用户不是工具的能力：确认是 harness 在调工具**之前**发的，工具那边不需要它。

        这条钉的是一个架构决定——哪天要让工具自己也能问用户（比如 ask_user），那得是有意
        改的，不是顺手把字段加回来。
        """
        assert not hasattr(ToolContext(cwd=tmp_path), "prompter")

    def test_a_fake_stands_in_for_the_protocol(self):
        """harness 只认协议：换一个实现进来就够了，不用经过 ToolContext。"""
        fake = FakePrompter()

        assert fake.confirm_tool(request()).allowed is True
        assert fake.confirm_edit(request("edit_file")).allowed is False
        assert fake.single_choice("用哪个？", ["a", "b"]) == "a"
