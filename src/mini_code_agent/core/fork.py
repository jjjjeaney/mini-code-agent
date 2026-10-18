"""fork 子 agent：继承父 agent 的上下文，复用缓存前缀。

和 `agent`（core/subagent.py）的分工是**上下文**，不是权限：

- `agent` 从零开始，只拿到一段任务描述。它读完的东西留在它自己的上下文里，主 agent 只收回
  一段结论——省的是主 agent 的上下文。
- `fork` 从**父 agent 现在的位置**开始：同一份 system prompt、同一张工具表、除最后一条
  assistant 消息之外的全部历史。相对父 agent 上一步发出去的请求（那正好是 `messages[:-1]`），
  fork 的请求是**纯追加**，自动前缀缓存整段命中。适合"必须带着现在这段对话才做得对"的事。

**为什么丢掉最后一条 assistant 消息**：fork 是在那条消息里的 `tool_use` 被派发时跑的，那条
消息包含了派发中的这个调用本身，还可能并排放着别的 `tool_use`。保留它就得给每个 `tool_use`
补一条 `tool_result`，而 fork 的结果此刻还不存在——请求直接不合法。整条去掉是唯一合法、且
不破坏前缀的做法（那条是响应，本来就不在上一发请求里）。代价是 fork 看不到同一批里的兄弟
调用，工具说明里因此劝模型别把它和别的工具并排。

**为什么 fork 工具在 fork 里也继续挂着**：工具表是请求前缀的一部分，把它从嵌套的表里删掉，
前缀就变了、缓存全废——那正是这个功能存在的理由。所以递归不靠"不发"，靠运行时的深度判断：
`ctx.fork.depth > 0` 时直接回一句普通结果。

**read_state 给的是副本**：fork 继承父 agent 读到过什么（父读过的文件它可以直接改），但它
自己的读写只落在副本上——跑完之后它新读的文件不替父 agent 挣到"可以改"的资格，它改过的
文件则会让父 agent 下次改之前被要求重读。
"""

from __future__ import annotations

from typing import Any

from mini_code_agent.console import console
from mini_code_agent.core.agent import runagent
from mini_code_agent.core.subagent import LabeledPrompter, tally_calls
from mini_code_agent.permissions import Grants
from mini_code_agent.prompts import Prompter
from mini_code_agent.providers.base import Provider
from mini_code_agent.tools import Tool, ToolContext

FORK_LABEL = "[fork] "

# 一次 fork 最多几步。和子 agent 一样：它是来"顺着现在这条线做一件事"的，不是来重开一摊。
FORK_MAX_STEPS = 8

FORK_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "description": {
            "type": "string",
            "description": (
                "三五个词说清这件事是什么。它只进终端那一行头，给用户看的——"
                "**别把任务写在这儿**，那在 prompt 里"
            ),
        },
        "prompt": {
            "type": "string",
            "description": (
                "交给 fork 的任务。它看得到当前对话的全文，所以背景不用重讲；"
                "但看不到你最后这条工具调用（以及同一批里的其它调用和它们的结果）"
            ),
        },
    },
    "required": ["description", "prompt"],
}


def _describe() -> str:
    return (
        "派一个 **fork 子任务**：它继承你现在这份对话——同一份系统提示、同一张工具表、"
        "除你最后这条工具调用之外的全部消息——所以背景不用重讲，缓存也直接命中。"
        "它的最终回复作为这次工具调用的结果回来，中间步骤你看不到。\n\n"
        "**什么时候用 fork，什么时候用 `agent`：**\n"
        "- **上下文依赖**：这件事要靠前面说过的背景、读过的文件、已经定下的方向才做得对 → fork。\n"
        "- **任务独立性**：想从**现在这个状态**出发，独立试另一种做法 → fork。\n"
        "- **上下文成本**：这件事需要你已经付过费的那段上下文，"
        "重讲一遍既费 token 又会失真 → fork。\n"
        "- **任务背景简单**：一段自足的任务描述就能说清 → `agent`，"
        "它从零开始，读完的东西不占你的上下文。\n\n"
        "**它看不到你最后这条工具调用**（包括同一批里的其它调用和它们的结果）；"
        "**同一批工具调用里别和别的工具并排**；**不能嵌套 fork**；它是同步的，跑完才回来。"
        "它改过的文件，你之后改之前要先重读。"
    )


def _task_message(task: str) -> str:
    """fork 收到的那条任务消息。**只影响新追加的尾巴**，不会动摇缓存前缀。"""
    return (
        "【fork 子任务】主 agent 把下面这件事交给你，在现有对话的基础上直接做完。\n"
        "不要再调用 fork；需要用户拍板的事、以及你改了哪些文件，都写进最终回复。\n"
        "（你看不到主 agent 最后那条工具调用，也看不到同一批里其它调用的结果。）\n\n"
        f"{task}"
    )


def make_fork_tool(
    *,
    provider: Provider,
    prompter: Prompter,
    grants: Grants,
) -> Tool:
    """把 fork 包成模型能调用的工具。

    闭包抓着 provider / prompter / grants：fork 要真发请求、要能问用户、要共用本会话批过的
    那几条。工具层因此一行都不用动，也不必给 ToolContext 加这些东西。
    """

    def run_fork(kwargs: dict[str, Any], ctx: ToolContext) -> str:
        fork = ctx.fork
        if fork is None:
            return "fork 不可用：这次运行没有打开它（CLI 上加 --fork）。"
        if fork.depth > 0:
            return (
                "不能再次 fork：你本身就是被 fork 出来的。"
                "这件事请你自己在当前上下文里做完。"
            )
        if fork.system is None:
            return "fork 不可用：这次运行没有 system prompt，继承不了父 agent 的前缀。"

        task = kwargs["prompt"]
        if not isinstance(task, str) or not task.strip():
            raise ValueError(
                "prompt 得是非空的字符串。fork 虽然看得到对话，但这一条仍要写清你要它做什么。"
            )

        console.print(f"▶ {FORK_LABEL}{kwargs['description']}", style="cyan", markup=False)

        # 最后一条是本次要回填结果的那条 assistant 消息（fork 这个 tool_use 就在里面）。
        # 它是上一发的**响应**，本来就不在任何请求前缀里，去掉它不损失缓存。
        base = list(fork.messages)
        if base and base[-1].get("role") == "assistant":
            base = base[:-1]

        # 副本：进来时继承父 agent 读到过什么，出去之后一条都不回流。
        read_state = ctx.read_state.copy()

        result = runagent(
            _task_message(task),
            provider,
            fork.registry,
            system=fork.system,
            max_steps=FORK_MAX_STEPS,
            cwd=ctx.cwd,
            allow_shell=ctx.allow_shell,
            permission_mode=ctx.permission_mode,
            prompter=LabeledPrompter(prompter, "fork"),
            grants=grants,
            messages=base,
            spill_dir=ctx.spill_dir,
            label=FORK_LABEL,
            read_state=read_state,
            # 工具继续挂着（删掉会改工具表前缀），能不能真的再 fork 由 depth 挡。
            allow_fork=True,
            fork_depth=fork.depth + 1,
            # 父 agent 算好的那份 schema，原样发下去。
            tool_schemas=fork.tools,
        )

        # 只数它自己这几条：直接数 result.messages 会把父 agent 整段历史算进去。
        note = tally_calls(result.messages[len(base) :])
        console.print(f"◀ {FORK_LABEL}{note}", style="dim", markup=False)

        if result.error:
            body = f"【fork 没跑成】{result.error}"
        elif result.final is None:
            body = (
                "【fork 没给出结论】它可能把步数用满了，也可能最后一发是空的"
                "（这两种在 AgentResult 里分不开）。可以再派一次、把范围缩窄一点，或者自己来。"
            )
        else:
            body = result.final

        return f"[fork 子 agent：{note}]\n\n{body}"

    return Tool(
        name="fork",
        description=_describe(),
        fn=run_fork,
        params=FORK_SCHEMA,
        # 结论没有天然上限；太长的落盘，上下文里留预览 + id。
        spill=True,
        # 结论就是"方向"本身，折了就真没了（和 agent 一样）。
        foldable=False,
        # `--fork` 关着的时候 ToolContext.fork 是 None，工具也就不发出去。
        available=lambda ctx: ctx.fork is not None,
    )
