"""把问题摆到终端上的那个实现：提示写 stderr，答案从键盘读。

这是 prompts.py 里那个 Prompter 协议的一份实现，也是这个项目里唯一 import typer 的地方
——协议那边只有标准库，终端这一层才碰键盘。

几件有意这么定的事：

- **提示走 stderr**。问答不是正文，混进 stdout 会弄脏终端输出；而且问题本身对管道那头没用
- **Ctrl+C 或输入流断了 = 拒绝这一次**，不是结束会话。一次拒绝是模型能接着协商的信息；
  真想退出该在提示符那一层表达（cli.py 的外层循环已经那么做了）
- **渲染和读键盘分开**：preview_* 和 interpret_choice 是纯函数，给定输入就有确定的输出，
  单独就能测；真正碰终端的只剩几行 typer 调用
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

import typer

from mini_code_agent.permissions import PermissionRequest
from mini_code_agent.prompts import Answer, NobodyToAskError, Prompter
from mini_code_agent.tools.base import Tool, ToolContext
from mini_code_agent.tools.safety import truncate_text

INDENT = "    "

ASK_USER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "question": {"type": "string", "description": "要问用户的问题，一句话说清楚"},
        "options": {
            "type": "array",
            "items": {"type": "string"},
            "description": "供他挑选的若干项；他也可以自己写一个，所以选项不全是致命的",
        },
    },
    "required": ["question", "options"],
}


def _block(text: str) -> str:
    """整段缩进。用户扫的是"哪些内容跟我有关"，缩进让正文从提示里分出来。"""
    return "\n".join(INDENT + line for line in text.splitlines())


def _mark(text: str, sign: str) -> str:
    """每行加个 +/-，读起来就是一处改动。空文本给一行空的，免得只剩一个光秃秃的符号。"""
    lines = text.splitlines() or [""]
    return "\n".join(f"{sign} {line}" for line in lines)


def preview_command(command: str, description: str = "") -> str:
    """一条命令的预览。

    沙箱那几行要写全：用户判的就是"这条命令会碰到什么"，光给一条命令是判不动的。

    `description` 是模型自己写的"这条命令要做什么"。它跟命令正文**分开摆、不拼进一行**，
    这是有意的：说明可能跟命令实际做的事对不上，而用户要放的其实是**命令**——两样分开，
    对不上才看得出来。也别指望它准，它只是给用户一个起点，命令原文还在下面摆着。
    """
    head = "要跑这条命令："
    if description:
        head += f"\n\n{INDENT}模型说这条是用来：{description}"
    return (
        f"{head}\n\n{_block(command)}\n\n"
        f"{INDENT}（跑在一次性容器里：只有项目目录挂进去、可写；根文件系统只读；没有网络；"
        f"跑完就删）"
    )


def preview_tool(request: PermissionRequest) -> str:
    """认不出类别的工具：把工具名和参数原样摆出来。"""
    args = json.dumps(request.tool_args, ensure_ascii=False, indent=2, default=str)
    return f"要用 {request.tool_name}：\n\n{_block(truncate_text(args))}"


def preview_edit(request: PermissionRequest) -> str:
    """改文件类的预览：尽量说清"改哪儿"，认不出来的退回参数视图。

    这里**不看磁盘**——所以 write_file 是新建还是覆盖、replace_lines 换掉的那几行原来
    长什么样，它都不知道。要看真 diff 得把询问挪到改文件那套骨架里（写临时文件之后、
    顶替之前），那是另一个取舍，先不背。
    """
    if request.tool_name == "edit_file":
        return _preview_replace(request)
    if request.tool_name == "replace_lines":
        return _preview_lines(request)
    if request.tool_name == "write_file":
        return _preview_write(request)
    return preview_tool(request)


def _preview_replace(request: PermissionRequest) -> str:
    args = request.tool_args
    head = f"要改 {args.get('path')}"
    head += "（所有匹配到的都换）" if args.get("replace_all") else "（只换第一处）"
    old = _mark(str(args.get("old_string", "")), "-")
    new = _mark(str(args.get("new_string", "")), "+")
    diff = f"{old}\n{new}"
    return f"{head}：\n\n{_block(truncate_text(diff))}"


def _preview_lines(request: PermissionRequest) -> str:
    args = request.tool_args
    path = args.get("path")
    start, end = args.get("start_line"), args.get("end_line")
    new_text = str(args.get("new_text", ""))
    # end = start - 1 是空区间：只插入，不替换。
    inserting = isinstance(start, int) and isinstance(end, int) and end < start

    if inserting:
        head = f"要在 {path} 的第 {start} 行前面插入"
    elif new_text:
        head = f"要把 {path} 的第 {start}-{end} 行换成"
    else:
        head = f"要删掉 {path} 的第 {start}-{end} 行"

    if not new_text:
        return head
    return f"{head}：\n\n{_block(truncate_text(_mark(new_text, '+')))}"


def _preview_write(request: PermissionRequest) -> str:
    args = request.tool_args
    content = str(args.get("content", ""))
    return (
        f"要写 {args.get('path')}（整篇 {len(content.splitlines())} 行）：\n\n"
        f"{_block(truncate_text(_mark(content, '+')))}"
    )


def interpret_choice(raw: str, options: Sequence[str]) -> str | None:
    """把用户敲的一行解释成答案。

    认三种：选项的序号、选项原文（大小写不计）、以及用户自己写的一句。返回 None 表示
    "这一下什么都没说"，该再问一次——空行是，序号越界也是（多半是敲错了，把那个数字
    当成答案收下更糟）。
    """
    text = raw.strip()
    if not text:
        return None
    if options and text.isdigit():
        index = int(text)
        if 1 <= index <= len(options):
            return options[index - 1]
        return None
    for option in options:
        if text.casefold() == option.casefold():
            return option
    return text


class TerminalPrompter:
    """把三个问题摆到终端上的那一版。

    没有构造参数、不存状态：它只做两件事——把该看的给人看，把答案读回来。
    """

    def confirm_tool(self, request: PermissionRequest) -> Answer:
        # bash 走这一条进来，但要看的是那行命令本身、不是它前面那串参数——分派留在
        # prompter 里，harness 就不必知道"命令藏在 args 的哪个键上"。
        #
        # 那条说明也在这儿取：harness 一路传下来的是**整个** request（`_gate` 把模型给的
        # 参数原样装进去），所以它不需要知道 "description" 这个键的存在。
        if request.tool_name == "bash":
            return self._ask(
                preview_command(
                    str(request.tool_args.get("command", "")),
                    str(request.tool_args.get("description", "")),
                )
            )
        return self._ask(preview_tool(request))

    def confirm_edit(self, request: PermissionRequest) -> Answer:
        return self._ask(preview_edit(request), remember_option=True)

    def single_choice(self, question: str, options: Sequence[str]) -> str:
        typer.echo(f"? {question}", err=True)
        for index, option in enumerate(options, start=1):
            typer.echo(f"{INDENT}{index}) {option}", err=True)
        if options:
            typer.echo(f"{INDENT}（也可以直接输入别的）", err=True)

        while True:
            try:
                raw = typer.prompt("选哪个", default="", show_default=False, err=True)
            except typer.Abort:
                raise NobodyToAskError(f"没人回答，这个选择做不了：{question}") from None
            answer = interpret_choice(str(raw), options)
            if answer is not None:
                return answer
            typer.echo("没听懂，再输一次。", err=True)

    def _ask(self, preview: str, remember_option: bool = False) -> Answer:
        """把预览摆出来，读一个回答。

        改文件那一档多一个 `a`（本会话这个文件别再问了）。**直接回车不算回答**：放行这种
        事不该有一个"顺手敲过去就过"的默认值。不是那三个键的其它输入，当成用户把理由写了
        下来——那就算拒绝，那句话会被带回给模型。
        """
        typer.echo(preview, err=True)
        keys = "[y] 是  [n] 否" + ("  [a] 本会话这个文件别再问" if remember_option else "")
        hint = f"放行吗？{keys}（也可以直接写下理由，那算拒绝）"

        while True:
            try:
                raw = typer.prompt(hint, default="", show_default=False, err=True)
            except typer.Abort:
                # Ctrl+C 或者输入流断了：这一次不批准。别把它升级成"结束整个会话"——
                # 用户想退出时该在提示符那一层按 Ctrl+C。
                typer.echo("（没有回答，按拒绝处理）", err=True)
                return Answer(False)

            text = str(raw).strip()
            lowered = text.lower()
            if lowered in ("y", "yes"):
                return Answer(True)
            if lowered in ("n", "no"):
                return Answer(False)
            if remember_option and lowered in ("a", "always"):
                return Answer(True, remember=True)
            if text:
                return Answer(False, note=text)
            typer.echo(f"没听懂，回 y / n{(' / a' if remember_option else '')}。", err=True)


def make_ask_user(prompter: Prompter) -> Tool:
    """把"问用户"包成模型能调的工具。

    谁手上有 prompter、谁在装配的时候把它注册进来——这个工具**不**内置在 default_tools()
    里：没人可问的地方它就不该存在。好处是 tools 层一行都不用动，也不必给 ToolContext
    加字段。

    内层的 ask_user 闭包抓着外面的 prompter。抓的是**那个对象**：以后要换实现或换状态，
    改它内部，别整个替换——换成另一个对象，这个闭包还指着老的。
    """

    def ask_user(kwargs: dict[str, Any], ctx: ToolContext) -> str:
        options = kwargs["options"]
        if not isinstance(options, list) or not all(isinstance(item, str) for item in options):
            raise ValueError(f"options 得是字符串数组，收到的是 {options!r}")
        return prompter.single_choice(kwargs["question"], options)

    return Tool(
        name="ask_user",
        description=(
            "把选择摆给用户，让他在几个选项里挑一个。需要用户拍板的事——偏好、取舍、"
            "没有唯一正确答案的设计选择——用它，不要自己替他决定。"
        ),
        fn=ask_user,
        params=ASK_USER_SCHEMA,
    )
