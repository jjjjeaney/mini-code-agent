"""子 agent：把"查清楚某件事"外包出去，只把结论拿回来。

**它解决的是上下文增长，不是"多开几个模型"。** 主 agent 为了回答一个问题读 20 个文件，
那 20 个文件就永远留在它的上下文里，后面每一发都要重发（缓存命中是 1/50 价，不是零），
而且挤占真正要做的事。子 agent 读完之后只把一段结论交回来，那 20 个文件跟着它的上下文
一起丢掉。

**四种类型，同一个工具**（见 `AGENT_TYPES`）：`general-purpose` 能动手，`explore` 和
`plan` 只能读，`fork` 继承父 agent 现在这份上下文和缓存前缀。前三种是"从零开始、只拿一段
任务描述"（`_pool` 现筛一份工具池）；`fork` 用的是父 agent 那一套 system prompt、工具表和历史
（见 `ForkContext`），所以它不经过 `_pool`，而是单独走一条分支。

**为什么在 `core/` 而不在 `tools/`**：它要调 `runagent`（递归跑一个子 agent），而 `tools/`
里的东西不能 import `core/`——`core/agent.py` 要 import `tools/`，反过来就成环了。这跟
`prompting.py` 是个 CLI 层模块、`ask_user` 在那儿注入是同一个道理：**harness 才有的能力，
就在 harness 这一层装进注册表**。

几条定下来的（都是先想清楚才写的，不是顺手的）：

- **不能递归**：子 agent 的池子里没有 `agent` 工具。这一条是**结构性**的——`agent` 和
  `ask_user` 都是在 CLI 层 `registry.register()` 注入的，不来自 `default_tools()`，而池子是
  从 `default_tools()` 现筛一份**新的**（`_pool`），两者根本不相干。代价是这个保证靠
  "新增工具一律进 `default_tools()`"这条隐含约定撑着，所以测试里有一条守它。
- **`ask_user` 不在池子里**：子 agent 要问用户就让主 agent 去问——它在总结里写清"需要用户
  确认 X"即可。
- **`remember` 不在池子里**：长期记忆是跨会话的东西，该由知道整段对话的主 agent 来记。
  子 agent 发现了值得记的事，在总结里说一句就行。
- **权限照问，和主 agent 一样**（改文件、跑命令都先问用户）。子 agent 不是绕开闸门的一条路。
  `Grants` 也共用一份：它本来就是按 (工具, 文件路径) 记的会话级批准，用户批的是"这个文件、
  这次会话"，不是"谁去动它"。代价是主 agent 批过的文件子 agent 不再问——所以权限提示必须
  带出**来源标记**（`LabeledPrompter`），不然用户分不清"别再问"批给了谁。
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from mini_code_agent.console import console
from mini_code_agent.core.agent import runagent
from mini_code_agent.interaction import BrokerPrompter, PermissionBroker
from mini_code_agent.prompts import Answer, Prompter
from mini_code_agent.system_prompt import build_subagent_prompt
from mini_code_agent.tools import Tool, ToolRegistry, default_tools

if TYPE_CHECKING:
    # 这几个都只出现在标注里，运行时不碰。
    from mini_code_agent.core.tasks import TaskManager
    from mini_code_agent.permissions import Grants, PermissionRequest
    from mini_code_agent.providers.base import Provider
    from mini_code_agent.tools import ToolContext

# 那一行过程最多列几类工具。多了就只说个总数——它进的是**主 agent 的上下文**，
# 而那一行只是"这个结论查得多深"的信号，不是流水账。
_MAX_TALLY_KINDS = 5

# 一次 `agent` 调用跑满这么久就自动转后台。**这是体验上的数，不是技术边界**：
# 用户愿意为一次同步调用等多久。改它不影响正确性。
BACKGROUND_AFTER_SECONDS = 120.0

# 同时跑着几个后台任务就不再接新的了。**只在模型主动要后台时硬拦**：自动交接的那一条本来
# 就已经在跑了，拦下来只会把它卡在前台。
BACKGROUND_LIMIT = 2

# fork 那一支：终端上的前缀，以及一次 fork 最多几步。和别的类型一样——它是来"顺着现在这条线
# 做一件事"的，不是来重开一摊。
FORK_LABEL = "[fork] "
FORK_MAX_STEPS = 8


@dataclass(frozen=True)
class AgentType:
    """一种子 agent。

    `summary` 进 `agent` 工具的 description，模型看着它挑——所以写的是"什么时候用它"，
    不是"它是什么"。

    `read_only` 为真时池子里只留 `tools/base.py` 里标了 `read_only` 的那些（外加联网）。
    **它和"不能改文件"是一回事**：只读工具本来就碰不了盘。

    `fork` 为真时这一型**不走 `_pool`**：它用的是父 agent 的那套 system prompt、工具表和
    历史（见 `ForkContext`），是 `run_subagent` 里单独的一条分支。`read_only` / `max_steps`
    对它只用于展示，真正的步数上限是 `FORK_MAX_STEPS`。

    `max_steps` 默认 8，和 `runagent` 自己的默认一样。几个类型现在都是这个数——先跑一跑看
    实际用几步，再决定要不要分开定；现在拍的数没有依据。
    """

    name: str
    summary: str
    read_only: bool
    max_steps: int = 8
    fork: bool = False


AGENT_TYPES: dict[str, AgentType] = {
    "general-purpose": AgentType(
        name="general-purpose",
        summary=(
            "通用任务，能读能搜能改文件能跑命令。需要它**动手**（改代码、跑测试）时用它。"
        ),
        read_only=False,
    ),
    "explore": AgentType(
        name="explore",
        summary=(
            "在代码库里**找事实**并报告位置：某个东西在哪儿定义、怎么实现的、哪儿用到了。"
            "只能读。"
        ),
        read_only=True,
    ),
    "plan": AgentType(
        name="plan",
        summary="读代码、设计一个**能照着做**的实施方案。只能读。",
        read_only=True,
    ),
    "fork": AgentType(
        name="fork",
        summary=(
            "**继承你现在这份对话和缓存前缀**，从当前状态出发独立做一件事。"
            "要靠前面聊过的背景、读过的文件才做得对时用它；需要启动时开 --fork。"
        ),
        read_only=False,
        max_steps=FORK_MAX_STEPS,
        fork=True,
    ),
}

AGENT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "type": {
            "type": "string",
            "enum": list(AGENT_TYPES),
            "description": "派哪种子 agent，几种的区别见这个工具的说明",
        },
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
                "交给子 agent 的完整任务描述。**它看不到你和用户的对话**，所以背景、约束、"
                "你要它回答什么，都得写进去；别写「见上面」这种指代"
            ),
        },
        "run_in_background": {
            "type": "boolean",
            "description": (
                "true = 立刻返回任务 id，先去做别的，结果会在后面的请求前自动送到你手上。"
                "只有**现在不需要这个结论、而且你还有别的活要干**时才用它。"
                "派了就别重派同一个任务。"
            ),
        },
    },
    "required": ["type", "description", "prompt"],
}


def _pool(agent_type: AgentType) -> ToolRegistry:
    """按类型给一份工具池。**每次现建一份新的。**

    "现建一份新的"这件事本身就是那条递归护栏：CLI 手里那份注册表另外装过 `agent` 和
    `ask_user`，而这里是从 `default_tools()` 重新筛的，跟它不相干——所以那两个**自动**
    进不来，不需要写进下面的排除名单。

    排除名单里因此只剩 `remember` 一个。理由见模块说明：长期记忆该由主 agent 来记。

    `fork` 不走这儿：它的工具表是父 agent 那一份（缓存前缀必须逐字节一致）。真调到这里是
    调用方的 bug，直接报出来，别悄悄给一份错的池子。
    """
    if agent_type.fork:
        raise ValueError("fork 用的是父 agent 的那张工具表，不经过 _pool")
    tools = default_tools().list()
    if agent_type.read_only:
        return ToolRegistry([tool for tool in tools if tool.read_only])
    return ToolRegistry([tool for tool in tools if tool.name != "remember"])


def tally_calls(messages: list[dict[str, Any]]) -> str:
    """把这段历史里的工具调用数成一句话（`fork` 也用它，所以是公开名）。"""
    """这一趟调了多少次工具、都是哪些，例如 `14 次工具调用（read_file×12、grep×2）`。

    从 messages 里的 `tool_use` 块数出来，**不给 `AgentResult` 加字段**——那份历史本来就
    在手边，加一个字段只是把同一个事实存两遍，还会多一处要维护的东西。

    按次数从多到少排，最多列 `_MAX_TALLY_KINDS` 类；再多就只报个类数。一次没调过
    （子 agent 直接答了）也说一声，别返回空串——那一行是要拼进给主 agent 的结果里的。
    """
    counts: Counter[str] = Counter()
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if isinstance(block, dict) and block.get("type") == "tool_use":
                counts[str(block.get("name"))] += 1

    if not counts:
        return "没调过工具"
    shown = counts.most_common(_MAX_TALLY_KINDS)
    breakdown = "、".join(f"{name}×{count}" for name, count in shown)
    if len(counts) > len(shown):
        breakdown += f" 等 {len(counts)} 类"
    return f"{sum(counts.values())} 次工具调用（{breakdown}）"


class LabeledPrompter:
    """给子 agent 的 prompter 包一层：每次请示前先说清是谁在问（`fork` 也用它）。"""
    """转发给真的 prompter，只在问之前打一行说明是哪个子 agent 在问。

    **为什么非包这一层不可**：`runagent` 的 `label` 只进得了它自己的 `console.print`，而
    权限确认走的是 prompter 自己那条路（`TerminalPrompter` 直接往 stderr 写）。不包的话，
    子 agent 要改文件时用户看到的提示和主 agent 一模一样——配上"Grants 两份共用"那条，
    他根本分不清"本会话别再问"批给了谁、批的是什么。

    实现的是 `Prompter` 协议**全部三个方法**：协议要完整。`single_choice` 实际上到不了
    （子 agent 的池子里没有 `ask_user`），但有它这条协议才是闭合的。
    """

    def __init__(self, inner: Prompter, name: str) -> None:
        self._inner = inner
        self._name = name

    def _announce(self) -> None:
        console.print(f"（[{self._name}] 子 agent 请求：）", style="dim", markup=False)

    def confirm_tool(self, request: PermissionRequest) -> Answer:
        self._announce()
        return self._inner.confirm_tool(request)

    def confirm_edit(self, request: PermissionRequest) -> Answer:
        self._announce()
        return self._inner.confirm_edit(request)

    def single_choice(self, question: str, options: Sequence[str]) -> str:
        self._announce()
        return self._inner.single_choice(question, options)


def _describe() -> str:
    """`agent` 工具的说明。类型那段从 `AGENT_TYPES` 现拼——表改了说明跟着改，不会漂。

    最后那两句是必须的：一句说**什么时候该用它**（不然模型不知道该省上下文），
    一句说 plan 档下的坑（`type` 的枚举是静态的，收不窄，只能在文字里拦）。
    """
    kinds = "\n".join(f"- `{t.name}`：{t.summary}" for t in AGENT_TYPES.values())
    return (
        "派一个**子 agent**去做一件事，把它最后那段结论拿回来。\n\n"
        "**它的上下文是独立而且一次性的**：它读过的那些文件不会进你的上下文，只有结论回来。"
        "所以**要翻很多文件才查得清的事**就该用它——**用户要计划、方案、“该怎么改”的时候"
        "尤其该用**：先派 `plan` 把现状和取舍查清楚，再照它的结论讲。省下的是上下文，"
        "不是钱（它每一步都是另外的请求）。\n\n"
        "**小事别派**：一次 `read_file`、一条 `grep` 就够的，自己来更快。\n\n"
        f"`type` 可选：\n{kinds}\n\n"
        "`fork` 和别的三型不一样：它继承你现在的系统提示、工具表和历史（**只少你最后那条"
        "工具调用**），所以背景不用重讲、缓存直接命中；代价是它看不到你最后这条调用，也看不到"
        "同一批里其它调用的结果——**别把它和别的工具并排派**。它继承你当前的权限档位"
        "（plan 档下同样改不了文件），而且**不能嵌套 fork**；"
        "它改过的文件，你之后改之前要先重读。\n\n"
        "**plan 档下只用 `explore` 或 `plan`。** 那一档里 `general-purpose` 改文件会被直接拒，"
        "派了大概率白烧步数。\n\n"
        f"**跑满 {int(BACKGROUND_AFTER_SECONDS)} 秒会自动转后台**，不用你特意说。"
        "`run_in_background=true` 则是一开始就转：立刻拿到任务 id，先去做别的。"
        "后台任务的结果会作为一条通知自动出现在你后面的请求里；太长的话会落盘，"
        "用 `read_tool_result` 按任务 id 取。**转后台之后需要当场批准的新调用仍然会问你**——"
        "请求会在你回到提示符时排上来；只有 `ask_user` 那道选择题不行，需要用户拍板的事"
        "让它写进结论里。"
    )


def _fork_task_message(task: str) -> str:
    """fork 收到的那条任务消息。**只影响新追加的尾巴**，不会动摇缓存前缀。"""
    return (
        "【fork 子任务】主 agent 把下面这件事交给你，在现有对话的基础上直接做完。\n"
        "不要再派 fork；需要用户拍板的事、以及你改了哪些文件，都写进最终回复。\n"
        "（你看不到主 agent 最后那条工具调用，也看不到同一批里其它调用的结果。）\n\n"
        f"{task}"
    )


def make_agent_tool(
    *,
    provider: Provider,
    prompter: Prompter,
    grants: Grants,
    model: str | None = None,
    tasks: TaskManager | None = None,
    background_after: float = BACKGROUND_AFTER_SECONDS,
    broker: PermissionBroker | None = None,
) -> Tool:
    """造出 `agent` 工具——主 agent 拿它派子 agent。

    **照抄 `prompting.make_ask_user(prompter)` 那个范式**：闭包捕获那几样 `ToolContext` 里
    没有的东西（provider / prompter / grants / model），在 CLI 层 `registry.register()`
    装进注册表。工具层一行都不用改，`ToolContext` 也不必认识"provider"这个概念。

    闭包抓的是**对象**，不是值——`prompter` 和 `grants` 都要求是全程活着的那一份（前者记着
    用户"本会话别再问"的答案，后者是本会话批过什么）。`permission_mode` 不在这儿抓，它走
    `ctx`：那个会被 `/permission_mode` 中途改掉，抓在闭包里会过期。
    """

    def run_subagent(kwargs: dict[str, Any], ctx: ToolContext) -> str:
        name = str(kwargs["type"])
        agent_type = AGENT_TYPES.get(name)
        if agent_type is None:
            # 认不出来是**模型给脏了**，不是工具坏了。所以当普通结果回给它自己改——
            # 抛异常的话会被 dispatch 包成 "ValueError: …" 那副样子，读起来像我们出了错。
            options = "、".join(AGENT_TYPES)
            return f"没有名为 {name!r} 的子 agent 类型。可选：{options}"

        task = kwargs["prompt"]
        if not isinstance(task, str) or not task.strip():
            raise ValueError(
                "prompt 得是非空的字符串。子 agent 看不到你和用户的对话，它的全部信息就是这一段。"
            )

        label = f"[{name}] "
        wants_background = bool(kwargs.get("run_in_background", False))
        # 一次性模式没有下一轮可以送结果，所以那边 `allow_background` 是 False——模型就算
        # 要了后台也只同步跑完，并在结果里说明一句。
        can_background = ctx.allow_background and tasks is not None
        if wants_background and can_background and tasks.running() >= BACKGROUND_LIMIT:
            return (
                f"已经有 {BACKGROUND_LIMIT} 个后台任务在跑，先别派新的。"
                "去做点别的、等它们的完成通知；这件事要是急，就用同步的方式跑完。"
            )
        # 主动要后台 = “跑够 0 秒就交接”；没主动要 = 前台跑满 `background_after` 再交接。
        use_background = wants_background and can_background
        after = 0.0 if use_background else (background_after if can_background else None)

        # 两条分支只在这里分开，后面的后台交接、结果措辞、长度闸全部共用。
        # `base` 是 fork 继承过来的那段父历史，计数时要把它切掉。
        base: list[dict[str, Any]] | None = None
        if agent_type.fork:
            fork = ctx.fork
            if fork is None:
                return (
                    "fork 不可用：这次运行没有打开它"
                    "（启动时加 --fork，或在交互里 /fork on）。"
                )
            if fork.depth > 0:
                return (
                    "不能再次 fork：你本身就是被 fork 出来的。"
                    "这件事请你自己在当前上下文里做完。"
                )
            if fork.system is None:
                return "fork 不可用：这次运行没有 system prompt，继承不了父 agent 的前缀。"

            # 最后一条是本次要回填结果的那条 assistant 消息（agent 这个 tool_use 就在里面）。
            # 它是上一发的**响应**，本来就不在任何请求前缀里，去掉它不损失缓存。
            base = list(fork.messages)
            if base and base[-1].get("role") == "assistant":
                base = base[:-1]

            label = FORK_LABEL
            system = fork.system
            registry_for_subagent = fork.registry
            max_steps = FORK_MAX_STEPS
            # 副本：进来时继承父 agent 读到过什么，出去之后一条都不回流。
            read_state = ctx.read_state.copy()
            sub_task = _fork_task_message(task)
            tool_schemas = fork.tools
            fork_depth = fork.depth + 1
            child_name = "fork"
            allow_fork = True
        else:
            system = build_subagent_prompt(ctx.cwd, name, ctx.permission_mode, model)
            registry_for_subagent = _pool(agent_type)
            max_steps = agent_type.max_steps
            read_state = None
            sub_task = task
            tool_schemas = None
            fork_depth = 0
            child_name = name
            allow_fork = False

        start_background = None
        if can_background:

            def _start(work):
                return tasks.start(
                    work,
                    label=label,
                    agent_type=name,
                    description=str(kwargs["description"]),
                    cwd=ctx.cwd,
                    spill_dir=ctx.spill_dir,
                    tally=tally_calls,
                )

            start_background = _start

        # 后台那一半的权限问题：有 broker 就入队、等读键盘线程在持有终端时来问；没有
        # （一次性模式、测试没注入）就退回一律拒绝的 BackgroundPrompter。
        background_prompter = None
        if broker is not None:

            def _background_prompter(cancel):
                return BrokerPrompter(broker, cancel, label)

            background_prompter = _background_prompter

        # 头尾这两行由这里打：runagent 自己那些行会带同一个 label，但它不知道"派的是谁、
        # 派去干什么"。
        console.print(f"▶ {label}{kwargs['description']}", style="cyan", markup=False)
        result = runagent(
            sub_task,
            provider,
            registry_for_subagent,
            system=system,
            max_steps=max_steps,
            cwd=ctx.cwd,
            allow_shell=ctx.allow_shell,
            permission_mode=ctx.permission_mode,
            prompter=LabeledPrompter(prompter, child_name),
            grants=grants,
            spill_dir=ctx.spill_dir,
            label=label,
            messages=base,
            read_state=read_state,
            allow_fork=allow_fork,
            fork_depth=fork_depth,
            tool_schemas=tool_schemas,
            background_after=after,
            handoff=start_background,
            background_prompter=background_prompter,
        )
        if result.handoff is not None:
            console.print(f"◀ {label}转到后台：{result.handoff}", style="dim", markup=False)
            if use_background:
                head = f"【后台任务 {result.handoff} 已派发】"
            else:
                head = (
                    f"【已经跑了 {int(background_after)} 秒，"
                    f"转到后台继续：{result.handoff}】"
                )
            return (
                f"{head}（{name}：{kwargs['description']}）\n"
                "它会在后台继续跑；结果会自动出现在你后面的请求里。"
                "**不要重派同一个任务。**"
            )

        # fork 的 `result.messages` 带着父 agent 整段历史，计数从 `len(base)` 往后切；
        # 其它类型那份从头到尾都是它自己的。
        note = tally_calls(result.messages if base is None else result.messages[len(base) :])
        console.print(f"◀ {label}{note}", style="dim", markup=False)

        # **三种收场分开说，而且都不抛异常。** 子 agent 没能得出结论，跟"命令跑了但退出码
        # 非 0"是一类事（见 shell.py 定过的那个规矩）：它是**信息**，交给主 agent 自己判断
        # 下一步，不该变成一个 is_error 把这次调用说成"工具坏了"。
        if result.error:
            body = f"【子 agent 没跑成】{result.error}"
        elif result.final is None:
            body = (
                "【子 agent 没给出结论】它可能把步数用满了，也可能最后一发是空的"
                "（这两种在 AgentResult 里分不开）。可以再派一次、把范围缩窄一点，或者自己来。"
            )
        else:
            body = result.final

        # 那一行过程在**最前面**，不在末尾：下面还要过一道长度闸（这个工具标了 spill），
        # 而截断和落盘预览都只留开头——放末尾的话它会被连同正文一起切掉。project_tree
        # 踩过同一个坑（见 files.py 里那句"说明放前面"）。
        prefix = ""
        if wants_background and not can_background:
            prefix = "（这次运行没有后台模式，已经同步跑完。）\n\n"
        return f"{prefix}[{name} 子 agent：{note}]\n\n{body}"

    return Tool(
        name="agent",
        description=_describe(),
        fn=run_subagent,
        params=AGENT_SCHEMA,
        # 子 agent 的总结没有天然上限。不封顶的话，一段很长的总结会原样进主 agent 的上下文，
        # 把"只把结论拿回来"这个初衷反过来打脸。标上之后太长的落盘，上下文里留预览 + id
        # （要全文用 read_tool_result 取）——那一行过程在最前面，所以预览里一定看得见。
        spill=True,
    )


# 旧名字留着：`tests/test_subagent.py` 直接按私有名导入。
_LabeledPrompter = LabeledPrompter
_tally = tally_calls
