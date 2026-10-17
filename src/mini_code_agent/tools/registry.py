"""工具注册表：登记工具，并把模型发起的调用派发出去。"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from mini_code_agent.memory import index_text as memory_index
from mini_code_agent.providers.base import ToolCall
from mini_code_agent.skills import index_text as skill_index
from mini_code_agent.tools import spill
from mini_code_agent.tools.base import Tool, ToolContext, ToolResult
from mini_code_agent.tools.builtin import ECHO_SCHEMA, echo
from mini_code_agent.tools.files import (
    EDIT_FILE_SCHEMA,
    GLOB_SCHEMA,
    GREP_SCHEMA,
    LIST_FILES_SCHEMA,
    PROJECT_TREE_SCHEMA,
    READ_FILE_SCHEMA,
    READ_TOOL_RESULT_SCHEMA,
    REPLACE_LINES_SCHEMA,
    WRITE_FILE_SCHEMA,
    edit_file,
    glob,
    grep,
    list_files,
    project_tree,
    read_file,
    read_tool_result,
    replace_lines,
    write_file,
)
from mini_code_agent.tools.memory import (
    READ_MEMORY_SCHEMA,
    REMEMBER_SCHEMA,
    read_memory,
    remember,
)
from mini_code_agent.tools.safety import MAX_TOOL_CHARS, truncate_text
from mini_code_agent.tools.shell import SHELL_SCHEMA, bash
from mini_code_agent.tools.skill import (
    READ_SKILL_FILE_SCHEMA,
    READ_SKILL_SCHEMA,
    read_skill,
    read_skill_file,
)
from mini_code_agent.tools.web import (
    WEB_FETCH_SCHEMA,
    WEB_SEARCH_SCHEMA,
    is_configured,
    web_fetch,
    web_search,
)


class ToolRegistry:
    def __init__(self, tools: Iterable[Tool] | None = None) -> None:
        self.tools: dict[str, Tool] = {}
        for tool in tools or ():
            self.register(tool)

    def register(self, tool: Tool) -> Tool:
        """注册一个工具。重名直接报错，不让后注册的悄悄顶掉先注册的。"""
        if tool.name in self.tools:
            raise ValueError(f"工具 {tool.name!r} 已经注册过了")
        self.tools[tool.name] = tool
        return tool

    def list(self) -> list[Tool]:
        """按注册顺序列出所有工具。"""
        return list(self.tools.values())

    def foldable_names(self) -> frozenset[str]:
        """这张表里**允许被折**的工具名（语义见 `Tool.foldable`，机制见 folding.py）。

        由注册表**现算**，不写死一份名单。好处是 CLI 另外注入的 `agent` / `ask_user` 自动
        落在外面——那两个没标那个字段，而**子 agent 交回来的结论正是最不该折的一类**（它
        背后是一整趟子 agent，而且它交回来的就是"方向"本身）。
        """
        return frozenset(tool.name for tool in self.tools.values() if tool.foldable)

    def run(
        self, ctx: ToolContext, name: str, arguments: dict[str, Any] | None = None
    ) -> Any:
        """按名字调用一个工具。工具自身的异常、以及缺了必填参数，都照常往上抛。"""
        tool = self.tools.get(name)
        if tool is None:
            raise KeyError(f"没有注册名为 {name!r} 的工具")

        # 按工具自己的 schema 校验必填参数——校验的就是发给模型的那份契约。
        # 少了参数在这里就拦住，别让工具拿着 None 悄悄跑完：那样模型会以为成功了，
        # 而它本来能收到一句"你少传了 text"然后自己改对。
        args = arguments or {}
        missing = [key for key in tool.params.get("required", []) if key not in args]
        if missing:
            raise ValueError(f"调用 {tool.name} 缺少必填参数：{'、'.join(missing)}")

        return tool.fn(args, ctx)

    def schemas(self, ctx: ToolContext) -> list[dict[str, Any]]:
        """传给 provider.complete(tools=...) 的列表。

        **此刻用不上的不发**（见 `Tool.available`）——广告一个做不到的能力，模型会试一次、
        吃一条报错、还花掉一步。

        筛的只有"存不存在"这一类：没开 `--shell`、没配 key、一份技能/记忆都没有。
        **权限不在筛的范围里**：那是闸门的事，而且藏起来会连"为什么不行"那句话一起藏掉。
        """
        return [
            tool.to_anthropic_schema() for tool in self.tools.values() if tool.available(ctx)
        ]

    def dispatch(self, ctx: ToolContext, call: ToolCall) -> ToolResult:
        """执行一次模型发起的调用，返回可以直接回填进对话的 ToolResult。

        和 run 的区别有两个：这里不往外抛异常（工具失败要作为结果交回模型让它自己改，
        否则一次工具报错就会终结整轮对话），以及**超长的结果在这里落盘**。
        """
        try:
            result = self.run(ctx, call.name, call.arguments)
        except Exception as exc:
            return ToolResult(
                id=call.id,
                name=call.name,
                result=f"{type(exc).__name__}: {exc}",
                is_error=True,
            )
        tool = self.tools.get(call.name)
        # 没注册的工具走不到这儿——上面 run 已经抛过 KeyError 了。
        if tool is not None and tool.spill:
            result = _spill_if_too_long(ctx, call.name, call.id, result)
        return ToolResult(id=call.id, name=call.name, result=result)


def _spill_if_too_long(
    ctx: ToolContext, name: str, tool_use_id: str, result: Any
) -> Any:
    """太长的结果整篇存到盘上，返回"预览 + 存哪儿"那一段话。

    只有**标了 spill 的工具**会走到这儿（调用方判的）。为什么搁在这一层而不是工具里：
    工具拿不到"多长算长"这把统一的尺，而落盘这种通用机制也不该在十几个工具里各写一遍。

    `tool_use_id` 一路传下去当**文件名**（见 spill.save）：它是一次调用的身份，比随机串更适合
    当键——同一个 id 反复落盘是同一个文件，不会越攒越多。

    落盘失败（盘满、权限不对）就**退回普通限额截断**——不能让一次盘写失败把几 MB 的文本
    原样放进上下文，那正是这个机制要防的事。
    """
    if not isinstance(result, str) or len(result) <= MAX_TOOL_CHARS:
        return result
    saved = spill.save(ctx, name, tool_use_id, result)
    if saved is None:
        return truncate_text(result, MAX_TOOL_CHARS)
    # 给模型看的路径尽量相对项目根（它自己就在那儿活动）；万一不在（spill_dir 被指到别处），
    # 退回绝对路径，总比印一条错的强。
    inside = saved.is_relative_to(ctx.cwd)
    where = saved.relative_to(ctx.cwd).as_posix() if inside else saved.as_posix()
    return spill.preview_message(result, name=saved.stem, where=where)


def default_tools() -> ToolRegistry:
    """装配一份可以直接用的工具集。

    每次调用都返回新实例：调用方想往自己的那份里再注册工具时，
    不会污染到别人手里那份。

    只读的那些标了 `read_only=True`（语义见 tools/base.py 那个字段）——子 agent 里
    explore / plan 两个类型的工具池就是按它筛出来的（见 core/subagent.py）。标的时候
    想的是"这个动作会不会改变什么"，所以 `echo` 不标：它往终端打印，那是输出。
    """
    return ToolRegistry(
        [
            Tool(
                name="echo",
                description="把一段文本打印到终端。需要让用户看到某段内容时用它。",
                fn=echo,
                params=ECHO_SCHEMA,
            ),
            Tool(
                name="list_files",
                description=(
                    "列出某个目录下面一层的文件和子目录，不往下钻。想知道某个目录里有什么时用它。"
                ),
                fn=list_files,
                params=LIST_FILES_SCHEMA,
                read_only=True,
                foldable=True,
            ),
            Tool(
                name="project_tree",
                description=(
                    "把整个项目画成一棵树，用于一眼看清整体结构。"
                    "想知道项目怎么组织、有哪些模块时用它。"
                ),
                fn=project_tree,
                params=PROJECT_TREE_SCHEMA,
                read_only=True,
            ),
            Tool(
                name="read_file",
                description=(
                    "读一个文本文件的内容。文件很长时可以只读一段："
                    "start_line / end_line 指定行号范围，从 1 数起。"
                    "返回的每行前面带着它在文件里的行号（形如 `12 | 正文`），"
                    "那是给你按位置引用用的，**不是文件内容的一部分**。"
                ),
                fn=read_file,
                params=READ_FILE_SCHEMA,
                read_only=True,
                foldable=True,
            ),
            Tool(
                name="read_memory",
                description=(
                    "按名字取一条**长期记忆**的正文。清单在系统提示里；跟手头这件事沾边的，"
                    "下手之前先取出来——清单上那一句话不够，别照着它猜。"
                ),
                fn=read_memory,
                params=READ_MEMORY_SCHEMA,
                read_only=True,
                # 一条记忆都没有就别发它——没清单的时候它只会让模型白试一次、花掉一步。
                available=lambda ctx: bool(memory_index(ctx.cwd)),
            ),
            Tool(
                name="remember",
                description=(
                    "把一条**长期记忆**记下来，以后每次会话都带着它。只记跨会话还成立的事："
                    "用户的偏好和习惯、这个项目的约定和边界、外部资源的指针。这一次任务的内容"
                    "别记——那些在会话记录里。正文里**点出依据**（哪份文件、哪条命令、什么时候"
                    "的事），以后才核实得了它还算不算数。写之前会先问用户。"
                ),
                fn=remember,
                params=REMEMBER_SCHEMA,
            ),
            Tool(
                name="read_skill",
                description=(
                    "取一份 skill 的正文——也就是那份**照着做**的步骤。清单在系统提示里，"
                    "每份只有名字和一句描述；**动手之前先扫一遍**，对得上就把它读出来按它做。"
                    "正文末尾会列出这份 skill 还带了哪些附件。"
                ),
                fn=read_skill,
                params=READ_SKILL_SCHEMA,
                read_only=True,
                # 同上：一份技能都没有就别发"取技能"的两个工具。
                available=lambda ctx: bool(skill_index(ctx.cwd)),
            ),
            Tool(
                name="read_skill_file",
                description=(
                    "取一份 skill 目录里的附件（脚本、模板、参考文档）。"
                    "有哪些附件 read_skill 会告诉你。只能读那一个 skill 自己目录里的东西。"
                ),
                fn=read_skill_file,
                params=READ_SKILL_FILE_SCHEMA,
                read_only=True,
                available=lambda ctx: bool(skill_index(ctx.cwd)),
            ),
            Tool(
                name="write_file",
                description=(
                    "写一个文件：不存在就新建，存在就用给的内容整个替换。"
                    "传的是**整篇**内容，不是片段；改局部用 edit_file。"
                    "覆盖已存在的文件前必须先读过它。"
                ),
                fn=write_file,
                params=WRITE_FILE_SCHEMA,
            ),
            Tool(
                name="edit_file",
                description=(
                    "把文件里的一段文本换成另一段，局部改动用它。"
                    "old_string 必须和原文一模一样，所以改之前得先读过这个文件。"
                    "从 read_file 的输出里抄的话，**先把每行前面那段 `行号 | ` 削掉**——"
                    "它不是原文，带着它一定匹配不上。"
                    "整篇重写用 write_file。"
                ),
                fn=edit_file,
                params=EDIT_FILE_SCHEMA,
            ),
            Tool(
                name="replace_lines",
                description=(
                    "按行号替换一段：把第 start_line 到 end_line 行换成 new_text。"
                    "行号就用 read_file 输出里印的那些，不用自己数。"
                    "改的内容在文件里不唯一、或者要按位置增删几行时用它；"
                    "按内容找位置用 edit_file。"
                ),
                fn=replace_lines,
                params=REPLACE_LINES_SCHEMA,
            ),
            Tool(
                name="bash",
                description=(
                    "在一次性沙箱容器里跑一条 shell 命令，返回退出码和输出。"
                    "只有项目目录挂进去、可写；没有网络。"
                    "跑测试、装依赖之外的命令、用命令行工具时用它。"
                    "需要在启动 agent 时打开 --shell 才可用。"
                ),
                fn=bash,
                params=SHELL_SCHEMA,
                # 没开 --shell 就别发：调了也只会拿到一句"没开命令执行"，而模型试这一下
                # 是要花掉一步的（max_steps 里的一步）。
                available=lambda ctx: ctx.allow_shell,
                # 命令的输出取不回来——它有副作用，重跑一遍未必是同一件事。
                spill=True,
                # 但它也是历史里最肿的一块（测试日志、构建输出），而且落了盘就取得回来，
                # 头一段里写着 id。
                foldable=True,
            ),
            Tool(
                name="glob",
                description=(
                    "按文件名或路径模式递归查找文件和目录。"
                    "想知道某类文件都在哪儿时用它，比逐个目录 list_files 快；"
                    "返回的路径可以直接拿去读。"
                ),
                fn=glob,
                params=GLOB_SCHEMA,
                read_only=True,
                foldable=True,
            ),
            Tool(
                name="grep",
                description=(
                    "按内容搜索，返回每条匹配的文件、行号和那一行。"
                    "找某个函数、变量、字符串在哪儿定义或使用时用它，比逐个文件读快。"
                ),
                fn=grep,
                params=GREP_SCHEMA,
                read_only=True,
                foldable=True,
            ),
            Tool(
                name="web_search",
                description=(
                    "联网搜索。遇到你不确定、可能已过时、或需要查证的事实时用它，"
                    "比如近期新闻、最新版本号、人物近况。返回若干条结果的标题、链接和摘要；"
                    "要读其中某一条的正文，把它的链接交给 web_fetch。"
                    "适用于查询最新信息、官方文档、"
                    "第三方库用法和不确定的事实。"
                ),
                fn=web_search,
                params=WEB_SEARCH_SCHEMA,
                read_only=True,
                foldable=True,
                # 没配 key 就别发这两个——发出去是必错的。
                available=lambda _: is_configured(),
            ),
            Tool(
                name="web_fetch",
                description=(
                    "打开一个具体的网页，把正文取回来。**先用 web_search 找到链接**，再用它细读"
                    "那一页——搜索给的摘要常常不够，要照着做的步骤、要抄的代码都在正文里。"
                    "一次一个链接。"
                ),
                fn=web_fetch,
                params=WEB_FETCH_SCHEMA,
                read_only=True,
                foldable=True,
                available=lambda _: is_configured(),
                # 正文是花钱取回来的，网页下一刻也可能变——截掉的部分再要就得重新取一次。
                spill=True,
            ),
            Tool(
                name="read_tool_result",
                description=(
                    "取一条**落盘的工具结果**的正文。某个工具的输出太长时，它的结果里会写明"
                    "完整内容存在哪儿、id 是什么，用那个 id 来取；很长的那种可以配 "
                    "start_line / end_line 分段读。"
                ),
                fn=read_tool_result,
                params=READ_TOOL_RESULT_SCHEMA,
                read_only=True,
                # **故意不加 available**：别的 `.agent` 工具是"有东西才发"（没记忆就没
                # read_memory），这一个不行——落盘发生在**这一轮之内**，而工具表一轮只算
                # 一次（在 runagent 开头），按"盘上有没有落盘文件"来判的话，模型在这一轮里
                # 刚拿到路径、却调不出这个工具。代价是它一直占着一格 schema。
            ),
        ]
    )

