"""权限模式的单元测试：解析用户敲的那一档，全部本地执行。"""

from __future__ import annotations

import pytest

from mini_code_agent.tools.base import ToolContext
from mini_code_agent.tools.permissions import (
    EDIT_TOOLS,
    PERMISSION_MODES,
    PermissionRequest,
    decide_permission,
    parse_permission_mode,
)


def request(name: str, **tool_args) -> PermissionRequest:
    """造一个权限请求。参数默认空着——判定只看工具名。"""
    return PermissionRequest(tool_name=name, tool_args=tool_args)


class TestParsePermissionMode:
    def test_the_three_canonical_values(self):
        assert parse_permission_mode("accept edit") == "accept edit"
        assert parse_permission_mode("default") == "default"
        assert parse_permission_mode("plan") == "plan"

    def test_the_space_can_be_typed_as_a_separator_or_left_out(self):
        """`accept edit` 中间那个空格在命令行上别扭，这几种写法都得认。"""
        for text in ("accept edit", "accept_edit", "accept-edit", "acceptedit"):
            assert parse_permission_mode(text) == "accept edit"

    def test_case_does_not_matter(self):
        assert parse_permission_mode("Accept Edit") == "accept edit"
        assert parse_permission_mode("DEFAULT") == "default"
        assert parse_permission_mode("Plan") == "plan"

    def test_extra_spaces_do_not_matter(self):
        assert parse_permission_mode("  accept   edit  ") == "accept edit"

    def test_the_result_is_a_plain_string(self):
        """返回的就是那三个字面量本身，可以直接比、可以塞进 dict，不用再转一层。"""
        mode = parse_permission_mode("plan")

        assert isinstance(mode, str)
        assert mode == "plan"

    def test_an_unknown_word_is_refused(self):
        with pytest.raises(ValueError, match="不是权限模式"):
            parse_permission_mode("yolo")

    def test_a_prefix_is_not_enough(self):
        """只有整个词对得上才认，`accept` 不算 `accept edit`。"""
        with pytest.raises(ValueError, match="不是权限模式"):
            parse_permission_mode("accept")

    def test_the_error_lists_what_is_valid(self):
        with pytest.raises(ValueError) as info:
            parse_permission_mode("yolo")

        message = str(info.value)
        assert "'accept edit'" in message
        assert "'default'" in message
        assert "'plan'" in message

    def test_a_non_string_is_refused(self):
        """别让它抛出 AttributeError——调用方拿到的该是一句能看懂的话。"""
        with pytest.raises(ValueError, match="得是字符串"):
            parse_permission_mode(None)


class TestTheModeList:
    def test_it_holds_exactly_the_three_modes(self):
        assert set(PERMISSION_MODES) == {"accept edit", "default", "plan"}

    def test_every_listed_mode_parses_to_itself(self):
        """这张表和对外承诺是同一份：列出来的每一个词都得真的能用。"""
        for mode in PERMISSION_MODES:
            assert parse_permission_mode(mode) == mode


class TestDecidePermission:
    """三档的规则表。

    判定是个纯函数：它只答 allow / deny / ask，"ask 之后真去问、用户说不行怎么办"是
    harness 那一层的事。
    """

    @pytest.mark.parametrize("mode", PERMISSION_MODES)
    def test_bash_always_asks(self, tmp_path, mode):
        """不论哪一档都要问。一条命令会干什么，用户事先看不出来。"""
        decision = decide_permission(request("bash", command="ls"), ToolContext(cwd=tmp_path), mode)

        assert decision.decision == "ask"

    @pytest.mark.parametrize("name", EDIT_TOOLS)
    def test_an_edit_tool_is_allowed_in_accept_edit_mode(self, tmp_path, name):
        context = ToolContext(cwd=tmp_path)

        assert decide_permission(request(name), context, "accept edit").decision == "allow"

    @pytest.mark.parametrize("name", EDIT_TOOLS)
    def test_an_edit_tool_asks_in_default_mode(self, tmp_path, name):
        context = ToolContext(cwd=tmp_path)

        assert decide_permission(request(name), context, "default").decision == "ask"

    @pytest.mark.parametrize("name", EDIT_TOOLS)
    def test_an_edit_tool_is_denied_in_plan_mode(self, tmp_path, name):
        context = ToolContext(cwd=tmp_path)

        assert decide_permission(request(name), context, "plan").decision == "deny"

    @pytest.mark.parametrize("mode", PERMISSION_MODES)
    def test_read_only_tools_are_never_gated(self, tmp_path, mode):
        """读、列目录、搜索本来就只受路径闸门管，逐个问没人受得了。"""
        context = ToolContext(cwd=tmp_path)

        for name in ("read_file", "list_files", "grep", "glob", "project_tree", "echo"):
            assert decide_permission(request(name), context, mode).decision == "allow"

    def test_an_unknown_tool_is_allowed(self, tmp_path):
        context = ToolContext(cwd=tmp_path)

        assert decide_permission(request("nope"), context, "plan").decision == "allow"

    def test_the_mode_can_be_written_the_way_the_user_typed_it(self, tmp_path):
        context = ToolContext(cwd=tmp_path)

        assert decide_permission(request("edit_file"), context, "Accept_Edit").decision == "allow"

    def test_a_bad_mode_raises_instead_of_falling_back(self, tmp_path):
        """回落到默认档的后果是：用户以为在 plan，实际在 default 上跑。"""
        context = ToolContext(cwd=tmp_path)

        with pytest.raises(ValueError, match="不是权限模式"):
            decide_permission(request("edit_file"), context, "yolo")

    def test_the_denial_says_why(self, tmp_path):
        """这句话会被原样回给模型，所以它得说清是哪种拒绝。"""
        decision = decide_permission(request("edit_file"), ToolContext(cwd=tmp_path), "plan")

        assert "plan" in decision.message
