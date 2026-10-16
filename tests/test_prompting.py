"""prompting.py 的单元测试：渲染出来的文本、用户输入怎么被解释。全部本地执行。

真正碰终端的那几行（typer.prompt / typer.confirm）打桩来验——被打桩的正是"读键盘"
那一层，所以上面那套渲染和解释仍然在真跑。
"""

from __future__ import annotations

import pytest
import typer

from mini_code_agent import prompting
from mini_code_agent.permissions import PermissionRequest
from mini_code_agent.prompting import (
    TerminalPrompter,
    interpret_choice,
    make_ask_user,
    preview_command,
    preview_edit,
    preview_tool,
)
from mini_code_agent.prompts import Answer, NobodyToAskError, NullPrompter
from mini_code_agent.providers.base import ToolCall
from mini_code_agent.tools import ToolContext, ToolRegistry


def request(tool_name: str, **tool_args) -> PermissionRequest:
    return PermissionRequest(tool_name=tool_name, tool_args=tool_args)


def aborting(*args, **kwargs):
    raise typer.Abort()


def answering(*answers):
    """按顺序回答。用完了就抛 Abort——测试卡在追问循环里时，一眼能看出是缺了答案。"""
    queue = list(answers)

    def fake(*args, **kwargs):
        if not queue:
            raise typer.Abort()
        return queue.pop(0)

    return fake


class TestPreviewCommand:
    def test_the_command_is_shown_verbatim(self):
        assert "pytest -q" in preview_command("pytest -q")

    def test_it_says_where_the_command_runs(self):
        """用户判的就是"这条命令会碰到什么"，光给一条命令是判不动的。"""
        text = preview_command("rm -rf /workspace")

        assert "容器" in text
        assert "没有网络" in text

    def test_the_models_own_description_rides_above_the_command(self):
        """模型写的那句说明单独占一行，而且摆在**命令上面**。

        分开摆是有意的：说明可能跟命令实际做的事对不上，而用户要放的是**命令**——
        两样挨着但不混在一起，对不上才看得出来。
        """
        text = preview_command("pytest -q", "跑一遍测试")

        assert "跑一遍测试" in text
        assert text.index("跑一遍测试") < text.index("pytest -q")

    def test_without_a_description_nothing_is_invented(self):
        """没给说明就不写那一行——别编一个出来。"""
        assert "模型说这条是用来" not in preview_command("ls")


class TestPreviewEdit:
    def test_edit_file_shows_both_sides_of_the_change(self):
        text = preview_edit(
            request("edit_file", path="a.txt", old_string="alpha", new_string="ALPHA")
        )

        assert "要改 a.txt" in text
        assert "- alpha" in text
        assert "+ ALPHA" in text
        assert "只换第一处" in text

    def test_edit_file_says_when_it_replaces_every_match(self):
        text = preview_edit(
            request("edit_file", path="a.txt", old_string="x", new_string="y", replace_all=True)
        )

        assert "所有匹配" in text

    def test_replace_lines_shows_the_line_range(self):
        text = preview_edit(
            request("replace_lines", path="a.txt", start_line=2, end_line=3, new_text="BETA")
        )

        assert "第 2-3 行" in text
        assert "+ BETA" in text

    def test_an_insertion_is_worded_as_an_insertion(self):
        """end = start - 1 是空区间，不能说成"替换第 3-2 行"那种没法读的话。"""
        text = preview_edit(
            request("replace_lines", path="a.txt", start_line=3, end_line=2, new_text="NEW")
        )

        assert "插入" in text
        assert "第 3 行" in text

    def test_an_empty_new_text_is_worded_as_a_deletion(self):
        text = preview_edit(
            request("replace_lines", path="a.txt", start_line=2, end_line=4, new_text="")
        )

        assert "删掉" in text
        assert "第 2-4 行" in text
        assert "+ " not in text

    def test_write_file_says_how_big_it_is(self):
        text = preview_edit(request("write_file", path="a.txt", content="one\ntwo\n"))

        assert "要写 a.txt" in text
        assert "2 行" in text

    def test_an_unknown_tool_falls_back_to_the_arguments_view(self):
        text = preview_edit(request("custom_tool", whatever=1))

        assert "要用 custom_tool" in text


class TestPreviewTool:
    def test_the_arguments_are_laid_out(self):
        text = preview_tool(request("custom_tool", a=1))

        assert "要用 custom_tool" in text
        assert '"a": 1' in text


class TestTruncation:
    def test_a_huge_payload_says_it_was_cut(self):
        """静默截断最糟：用户会以为自己看到的就是全部。"""
        text = preview_edit(request("write_file", path="a.txt", content="x" * 20000))

        assert "已截断" in text


class TestInterpretChoice:
    def test_an_index_picks_that_option(self):
        assert interpret_choice("2", ["a", "b"]) == "b"

    def test_the_option_text_itself_works_too(self):
        assert interpret_choice("b", ["a", "b"]) == "b"

    def test_case_does_not_matter(self):
        assert interpret_choice("Friday", ["friday", "monday"]) == "friday"

    def test_anything_else_is_taken_as_the_users_own_answer(self):
        """模型给的选项常常不全，用户得有一条自己写答案的路。"""
        assert interpret_choice("用第三种方案", ["a", "b"]) == "用第三种方案"

    def test_an_empty_line_asks_again(self):
        assert interpret_choice("   ", ["a", "b"]) is None

    def test_an_out_of_range_index_asks_again(self):
        """敲错的数字不能当答案收下。"""
        assert interpret_choice("9", ["a", "b"]) is None

    def test_with_no_options_everything_is_free_text(self):
        assert interpret_choice("3", []) == "3"


class TestTerminalPrompter:
    def test_a_yes_goes_through(self, monkeypatch):
        monkeypatch.setattr(prompting.typer, "prompt", answering("y"))

        prompter = TerminalPrompter()

        assert prompter.confirm_tool(request("bash", command="pytest -q")).allowed is True

    def test_a_no_goes_through(self, monkeypatch):
        monkeypatch.setattr(prompting.typer, "prompt", answering("n"))

        assert TerminalPrompter().confirm_tool(request("bash", command="ls")).allowed is False

    def test_bash_is_shown_as_a_command_line(self, monkeypatch, capsys):
        """bash 从 confirm_tool 进来，渲染的却是那行命令——分派是 prompter 自己做的。"""
        monkeypatch.setattr(prompting.typer, "prompt", answering("y"))

        TerminalPrompter().confirm_tool(request("bash", command="pytest -q"))

        assert "pytest -q" in capsys.readouterr().err

    def test_bash_shows_the_description_the_model_wrote(self, monkeypatch, capsys):
        """模型给的那句说明一路传到预览里——它是用户判断的依据里的一部分。

        harness 全程只递**整个** request（`_gate` 把模型给的参数原样装进去），所以
        "description 藏在 tool_args 哪个键上"这件事只有 prompter 知道。
        """
        monkeypatch.setattr(prompting.typer, "prompt", answering("y"))

        TerminalPrompter().confirm_tool(
            request("bash", command="pytest -q", description="跑一遍测试")
        )

        assert "跑一遍测试" in capsys.readouterr().err

    def test_the_preview_goes_to_stderr_not_stdout(self, monkeypatch, capsys):
        """stdout 留给最终答案：`mca "..." > out.txt` 不该收到提示。"""
        monkeypatch.setattr(prompting.typer, "prompt", answering("y"))

        TerminalPrompter().confirm_tool(request("bash", command="pytest -q"))

        captured = capsys.readouterr()
        assert "pytest -q" in captured.err
        assert captured.out == ""

    def test_an_interrupt_is_a_no_not_an_exit(self, monkeypatch):
        """Ctrl+C / 输入流断了：拒绝这一次，别把整个会话掀了。"""
        monkeypatch.setattr(prompting.typer, "prompt", aborting)

        answer = TerminalPrompter().confirm_edit(request("edit_file", path="a.txt"))

        assert answer.allowed is False

    def test_editing_offers_a_remember_option(self, monkeypatch):
        """改文件那一档可以答"本会话这个文件别再问了"。"""
        monkeypatch.setattr(prompting.typer, "prompt", answering("a"))

        answer = TerminalPrompter().confirm_edit(request("edit_file", path="a.txt"))

        assert answer.allowed is True
        assert answer.remember is True

    def test_a_command_does_not_offer_it(self, monkeypatch, capsys):
        monkeypatch.setattr(prompting.typer, "prompt", answering("a"))

        answer = TerminalPrompter().confirm_tool(request("bash", command="pytest -q"))

        assert answer.remember is False
        assert "本会话" not in capsys.readouterr().err  # 提示里也不该出现这一档

    def test_a_written_reason_comes_back_as_a_note(self, monkeypatch):
        """不是那三个键的输入，当作用户把理由写下来了——那算拒绝，理由带回去。"""
        monkeypatch.setattr(prompting.typer, "prompt", answering("改 bar.py 而不是这个"))

        answer = TerminalPrompter().confirm_edit(request("edit_file", path="a.txt"))

        assert answer.allowed is False
        assert answer.note == "改 bar.py 而不是这个"

    def test_an_empty_answer_is_asked_again(self, monkeypatch):
        """直接回车不算回答：放行这种事不该有一个顺手敲过去就过的默认值。"""
        monkeypatch.setattr(prompting.typer, "prompt", answering("", "y"))

        prompter = TerminalPrompter()

        assert prompter.confirm_tool(request("bash", command="pytest -q")).allowed is True

    def test_single_choice_returns_the_picked_option(self, monkeypatch):
        monkeypatch.setattr(prompting.typer, "prompt", answering("2"))

        assert TerminalPrompter().single_choice("用哪个？", ["a", "b"]) == "b"

    def test_single_choice_re_asks_on_a_useless_answer(self, monkeypatch):
        monkeypatch.setattr(prompting.typer, "prompt", answering("", "9", "1"))

        assert TerminalPrompter().single_choice("用哪个？", ["a", "b"]) == "a"

    def test_an_interrupt_while_asking_raises(self, monkeypatch):
        """提问没有缺省答案：替他挑一个就是替他做了决定。"""
        monkeypatch.setattr(prompting.typer, "prompt", aborting)

        with pytest.raises(NobodyToAskError, match="没人回答"):
            TerminalPrompter().single_choice("用哪个？", ["a", "b"])


class FixedPrompter:
    """按剧本回答的假 prompter，顺便记下问过什么。"""

    def __init__(self, answer: str) -> None:
        self.answer = answer
        self.asked: list[tuple[str, list[str]]] = []

    def confirm_tool(self, request: PermissionRequest) -> Answer:
        return Answer(True)

    def confirm_edit(self, request: PermissionRequest) -> Answer:
        return Answer(True)

    def single_choice(self, question: str, options: list[str]) -> str:
        self.asked.append((question, list(options)))
        return self.answer


class TestAskUserTool:
    """把"问用户"包成工具之后，模型那边看到的是什么。"""

    def test_the_model_sees_it_under_the_name_ask_user(self):
        tool = make_ask_user(FixedPrompter("a"))

        assert tool.name == "ask_user"
        assert tool.params["required"] == ["question", "options"]
        assert tool.description  # 得有一句告诉模型什么时候该用它

    def test_it_returns_what_the_user_picked(self, tmp_path):
        tool = make_ask_user(FixedPrompter("选项B"))

        answer = tool.fn(
            {"question": "用哪个？", "options": ["选项A", "选项B"]}, ToolContext(cwd=tmp_path)
        )

        assert answer == "选项B"

    def test_it_passes_the_question_and_the_options_through(self, tmp_path):
        prompter = FixedPrompter("选项A")
        tool = make_ask_user(prompter)

        tool.fn({"question": "用哪个？", "options": ["选项A", "选项B"]}, ToolContext(cwd=tmp_path))

        assert prompter.asked == [("用哪个？", ["选项A", "选项B"])]

    def test_options_must_be_a_list_of_strings(self, tmp_path):
        """模型偶尔会把数组写成字符串——照单全收的话 "选项A" 会被当成三个选项。"""
        tool = make_ask_user(FixedPrompter("x"))

        with pytest.raises(ValueError, match="字符串数组"):
            tool.fn({"question": "用哪个？", "options": "选项A"}, ToolContext(cwd=tmp_path))

    def test_nobody_to_ask_comes_back_as_an_error_result(self, tmp_path):
        """注册表会把异常翻成模型看得见的结果——"没人可问"这条路不用另写代码。"""
        registry = ToolRegistry([make_ask_user(NullPrompter())])
        call = ToolCall(
            id="t1", name="ask_user", arguments={"question": "用哪个？", "options": ["a"]}
        )

        result = registry.dispatch(ToolContext(cwd=tmp_path), call)

        assert result.is_error is True
        assert "没人可问" in result.result
