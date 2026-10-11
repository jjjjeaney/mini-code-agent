"""把问题摆到终端上的那个实现：提示写 stderr，答案从键盘读。

这是 tools/prompt.py 里那个 Prompter 协议的一份实现，也是这个项目里唯一 import typer
的地方——协议住在 tools 层，而那边不许 import cli。

几件有意这么定的事：

- **提示走 stderr**。一次性模式下 stdout 是最终答案（`mca "..." > out.txt`），提示混进去
  就毁了那个约定；而且问题本身对管道那头没用
- **Ctrl+C 或输入流断了 = 拒绝这一次**，不是结束会话。一次拒绝是模型能接着协商的信息；
  真想退出该在提示符那一层表达（cli.py 的外层循环已经那么做了）
- **渲染和读键盘分开**：preview_* 和 interpret_choice 是纯函数，给定输入就有确定的输出，
  单独就能测；真正碰终端的只剩几行 typer 调用
"""

from __future__ import annotations

import json
from collections.abc import Sequence

import typer

from mini_code_agent.tools.permissions import PermissionRequest
from mini_code_agent.tools.prompt import NobodyToAskError
from mini_code_agent.tools.safety import truncate_text

INDENT = "    "


def _block(text: str) -> str:
    """整段缩进。用户扫的是"哪些内容跟我有关"，缩进让正文从提示里分出来。"""
    return "\n".join(INDENT + line for line in text.splitlines())


def _mark(text: str, sign: str) -> str:
    """每行加个 +/-，读起来就是一处改动。空文本给一行空的，免得只剩一个光秃秃的符号。"""
    lines = text.splitlines() or [""]
    return "\n".join(f"{sign} {line}" for line in lines)


def preview_command(command: str) -> str:
    """一条命令的预览。

    沙箱那几行要写全：用户判的就是"这条命令会碰到什么"，光给一条命令是判不动的。
    """
    return (
        f"要跑这条命令：\n\n{_block(command)}\n\n"
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
    """把四个问题摆到终端上的那一版。

    没有构造参数、不存状态：它只做两件事——把该看的给人看，把答案读回来。
    """

    def confirm_command(self, command: str) -> bool:
        return self._yes_no(preview_command(command))

    def confirm_tool(self, request: PermissionRequest) -> bool:
        return self._yes_no(preview_tool(request))

    def confirm_edit(self, request: PermissionRequest) -> bool:
        return self._yes_no(preview_edit(request))

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

    def _yes_no(self, preview: str) -> bool:
        typer.echo(preview, err=True)
        try:
            return typer.confirm("放行吗？", default=False, err=True)
        except typer.Abort:
            # Ctrl+C 或者输入流断了：这一次不批准。别把它升级成"结束整个会话"——
            # 用户想退出时该在提示符那一层按 Ctrl+C。
            typer.echo("（没有回答，按拒绝处理）", err=True)
            return False
