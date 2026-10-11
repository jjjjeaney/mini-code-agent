"""权限模式的单元测试：解析用户敲的那一档，全部本地执行。"""

from __future__ import annotations

import pytest

from mini_code_agent.permissions import (
    EDIT_TOOLS,
    PERMISSION_MODES,
    PermissionRequest,
    decide_permission,
    parse_permission_mode,
)

# decide_permission 的第二个参数是 allow_shell。位置参数直接写 True / False 看不出是什么，
# 测试里用这两个名字代替。
SHELL_OFF = False
SHELL_ON = True


def request(name: str, **tool_args) -> PermissionRequest:
    """造一个权限请求。参数默认空着——判定只看工具名。"""
    return PermissionRequest(tool_name=name, tool_args=tool_args)


class TestParsePermissionMode:
    def test_the_four_canonical_values(self):
        assert parse_permission_mode("accept edits") == "accept edits"
        assert parse_permission_mode("bypass permissions") == "bypass permissions"
        assert parse_permission_mode("default") == "default"
        assert parse_permission_mode("plan") == "plan"

    def test_the_space_can_be_typed_as_a_separator_or_left_out(self):
        """两个词中间那个空格在命令行上别扭，这几种写法都得认。"""
        for text in ("accept edits", "accept_edits", "accept-edits", "acceptedits", "acceptEdits"):
            assert parse_permission_mode(text) == "accept edits"
        for text in (
            "bypass permissions",
            "bypass_permissions",
            "bypass-permissions",
            "bypasspermissions",
            "bypassPermissions",
        ):
            assert parse_permission_mode(text) == "bypass permissions"

    def test_case_does_not_matter(self):
        assert parse_permission_mode("Accept Edits") == "accept edits"
        assert parse_permission_mode("DEFAULT") == "default"
        assert parse_permission_mode("Plan") == "plan"

    def test_extra_spaces_do_not_matter(self):
        assert parse_permission_mode("  accept   edits  ") == "accept edits"

    def test_the_result_is_a_plain_string(self):
        """返回的就是那四个字面量本身，可以直接比、可以塞进 dict，不用再转一层。"""
        mode = parse_permission_mode("plan")

        assert isinstance(mode, str)
        assert mode == "plan"

    def test_an_unknown_word_is_refused(self):
        with pytest.raises(ValueError, match="不是权限模式"):
            parse_permission_mode("yolo")

    def test_the_short_forms_work(self):
        """命令行上敲一整串太啰嗦，`accept` / `bypass` 是约定好的短名。"""
        assert parse_permission_mode("accept") == "accept edits"
        assert parse_permission_mode("bypass") == "bypass permissions"
        assert parse_permission_mode("Bypass") == "bypass permissions"

    def test_a_prefix_is_not_enough(self):
        """短名是整词约定的，不是前缀：`bypas`、`accepte` 都不算。"""
        for text in ("bypas", "accepte", "bypass perm"):
            with pytest.raises(ValueError, match="不是权限模式"):
                parse_permission_mode(text)

    def test_the_error_lists_what_is_valid(self):
        with pytest.raises(ValueError) as info:
            parse_permission_mode("yolo")

        message = str(info.value)
        assert "'accept edits'" in message
        assert "'bypass permissions'" in message
        assert "'default'" in message
        assert "'plan'" in message
        assert "'bypass'" in message  # 短写法也列出来

    def test_a_non_string_is_refused(self):
        """别让它抛出 AttributeError——调用方拿到的该是一句能看懂的话。"""
        with pytest.raises(ValueError, match="得是字符串"):
            parse_permission_mode(None)


class TestTheModeList:
    def test_it_holds_exactly_the_four_modes(self):
        assert set(PERMISSION_MODES) == {
            "accept edits",
            "bypass permissions",
            "default",
            "plan",
        }

    def test_every_listed_mode_parses_to_itself(self):
        """这张表和对外承诺是同一份：列出来的每一个词都得真的能用。"""
        for mode in PERMISSION_MODES:
            assert parse_permission_mode(mode) == mode


class TestDecidePermission:
    """规则表。

    判定是个纯函数，参数全是权限自己的东西（工具名、shell 开没开、档位）——它不认识
    ToolContext。"ask 之后真去问、用户说不行怎么办"是 harness 那一层的事，不在这儿测。
    """

    @pytest.mark.parametrize("mode", ["default", "accept edits", "plan"])
    def test_bash_asks_when_shell_is_on(self, mode):
        """除了 bypass，其它档都要问。一条命令会干什么，用户事先看不出来。"""
        decision = decide_permission(request("bash", command="ls"), SHELL_ON, mode)

        assert decision.decision == "ask"

    def test_bash_is_not_asked_under_bypass(self):
        """bypass 的全部意义就是这个：什么都不问。"""
        decision = decide_permission(request("bash", command="ls"), SHELL_ON, "bypass permissions")

        assert decision.decision == "allow"

    @pytest.mark.parametrize("mode", PERMISSION_MODES)
    def test_bash_is_let_through_when_shell_is_off(self, mode):
        """没开 shell 时它本来就会报"没开命令执行"，没必要先弹一句询问再让人发现跑不了。"""
        decision = decide_permission(request("bash", command="ls"), SHELL_OFF, mode)

        assert decision.decision == "allow"

    @pytest.mark.parametrize("name", EDIT_TOOLS)
    def test_an_edit_tool_is_allowed_in_accept_edits_mode(self, name):
        assert decide_permission(request(name), SHELL_ON, "accept edits").decision == "allow"

    @pytest.mark.parametrize("name", EDIT_TOOLS)
    def test_an_edit_tool_is_allowed_under_bypass(self, name):
        assert decide_permission(request(name), SHELL_ON, "bypass permissions").decision == "allow"

    @pytest.mark.parametrize("name", EDIT_TOOLS)
    def test_an_edit_tool_asks_in_default_mode(self, name):
        assert decide_permission(request(name), SHELL_ON, "default").decision == "ask"

    @pytest.mark.parametrize("name", EDIT_TOOLS)
    def test_an_edit_tool_is_denied_in_plan_mode(self, name):
        assert decide_permission(request(name), SHELL_ON, "plan").decision == "deny"

    @pytest.mark.parametrize("mode", PERMISSION_MODES)
    def test_read_only_tools_are_never_gated(self, mode):
        """读、列目录、搜索本来就只受路径闸门管，逐个问没人受得了。"""
        for name in ("read_file", "list_files", "grep", "glob", "project_tree", "echo"):
            assert decide_permission(request(name), SHELL_ON, mode).decision == "allow"

    def test_an_unknown_tool_is_allowed(self):
        assert decide_permission(request("nope"), SHELL_ON, "plan").decision == "allow"

    def test_the_mode_can_be_written_the_way_the_user_typed_it(self):
        decision = decide_permission(request("edit_file"), SHELL_ON, "Accept_Edits")

        assert decision.decision == "allow"

    def test_a_bad_mode_raises_instead_of_falling_back(self):
        """回落到默认档的后果是：用户以为在 plan，实际在 default 上跑。"""
        with pytest.raises(ValueError, match="不是权限模式"):
            decide_permission(request("edit_file"), SHELL_ON, "yolo")

    def test_the_denial_says_why(self):
        """这句话会被原样回给模型，所以它得说清是哪种拒绝。"""
        decision = decide_permission(request("edit_file"), SHELL_ON, "plan")

        assert "plan" in decision.message
