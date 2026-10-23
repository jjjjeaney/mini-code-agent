"""agent 主循环。"""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from mini_code_agent.console import console
from mini_code_agent.interaction import ask_permission
from mini_code_agent.permissions import Grants, PermissionRequest, decide_permission
from mini_code_agent.prompts import Answer, BackgroundPrompter, NullPrompter, Prompter
from mini_code_agent.providers.base import (
    ModelResponse,
    Provider,
    ProviderError,
    ToolCall,
    TransientProviderError,
)
from mini_code_agent.results import DEFAULT_RESULTS
from mini_code_agent.sandbox import DEFAULT_SANDBOX
from mini_code_agent.tools import (
    ForkContext,
    ReadFileState,
    ToolContext,
    ToolRegistry,
    ToolResult,
    default_tools,
)
from mini_code_agent.usage import UsageMeter, UsageTotals

# 瞬时故障（连不上、超时、限流、对面 5xx）**补一次**之前等多久。
#
# SDK 自己那 2 次重试的退避大概是 0.5 + 1 秒，所以这个数要比它明显长，否则补的那次等于白补；
# 但也不能长到用户以为卡死了。只补一次——补几次写死在 ask() 里，要改就在那儿加一层。
RETRY_PAUSE_SECONDS = 3.0

# 一批只读工具里最多同时跑几个。这些工具是 I/O 密集的（读盘、联网），线程够用；上限主要是
# 别让一次批量 web 请求把对面的限流打穿，也别把磁盘打满。
PARALLEL_LIMIT = 4


@dataclass
class ContextAnchor:
    """「这一发请求有多大」的一个快照。压缩阈值靠它（见 summarizing.context_size）。

    `tokens` 是那一发的输入总量——三个输入字段相加（它们不重叠，见 `_usage_note`）。
    `covered` 是**那一发请求里实际装了多少条 messages**。

    **两个必须来自同一发**，而 `covered` 尤其不能拿 `AgentResult.messages` 的长度去顶：那份
    比发出去的长——响应是先拿到的，assistant 那条是之后才 append 进去的。
    """

    tokens: int
    covered: int


@dataclass
class AgentResult:
    final: str | None
    trace: list[str]
    messages: list[dict[str, Any]]

    # 这一轮没跑成的原因（人话，来自 ProviderError）。非空时 final 一定是 None。
    #
    # **它和"用满步数"不一样**：那个是模型自己没在给定步数内收尾，这个是根本没能把请求发出去。
    # 调用方要分开说——前者提一句就行，后者要告诉用户"再发一次"还是"去查配置"。
    error: str | None = None

    # 这一轮里**带工具的那最后一发**有多大。一发都没发出去过（假 provider 不报用量）就是 None
    # ——那时调用方只能按字符粗估，别把它当成"很小"。
    #
    # **为什么是"带工具的最后一发"而不是"最后发"**：用满步数时最后那一发是不带 tools 的收尾
    # 请求（见下面循环尾巴），它少了整段工具 schema，而那一大段占了前缀的一多半。拿它当锚点会
    # 系统性小看半个前缀。
    anchor: ContextAnchor | None = None

    # 这一轮在跑到两分钟时被交接给后台了，值是任务 id。非空时 final 一定是 None——
    # 结论还没出来，调用方要用这个 id 去任务表里找。
    handoff: str | None = None

    # `final` 这段文字是不是已经实时打印过了。只表示"屏幕上有了"，不表示"这一轮流式过"：
    # 工具调用之前的叙述文字不算（final 还是 None），后台交接/出错/用满步数也一律 False。
    # CLI 靠它决定轮末还要不要 `console.print(result.final)`。
    final_streamed: bool = False

    # 最终回答那一发的用量行。只有 `defer_final_usage=True` 时才会填——调用方要在**打印完
    # 答案之后**再把它打出来，这样流式和非流式都是一个顺序：答案在前、用量在后。工具调用
    # 中间步骤的用量仍然由 `send()` 立刻打印（那时还没有答案，紧跟着的是工具行）。
    final_usage: str | None = None

    # 这一轮**所有请求**（含同步子 agent）的用量总和，已经格式化好。只有最终回答存在、
    # 而且请求数大于 1 时才填——只有一发时 `final_usage` 本身就是总和，再打一遍是噪音。
    # CLI 在 `final_usage` 之后打印它。
    turn_usage: str | None = None


# 取消时写进 AgentResult.error 的那句。任务表拿它区分"用户取消"和"真出错"。
CANCELLED_ERROR = "已取消"

# 往哪儿打一行。第一个参数是文本、第二个是 style——和 runagent 里原来那些
# `console.print(..., style=..., markup=False)` 一一对应。
Emit = Callable[[str, str], None]

# 流式正文往哪儿写：只收一段文字，不加换行、不上色。和 Emit 分开，因为它的调用频率是
# 每个 delta 一次（一次回复可能上百次），不能走 console.print 那种整行渲染。
OnText = Callable[[str], None]


class WaitHint(Protocol):
    """流式请求开始/结束时各叫一次。"等第一个 token"长什么样由调用方决定。

    放在 agent 层是因为只有它知道"请求刚发出去、还没有任何输出"这个窗口；CLI 那边实现成
    一行会被擦掉的提示，测试里可以是一个只记调用的假对象。
    """

    def begin(self) -> None: ...

    def end(self) -> None: ...


# 后台那一半的工作函数：收（往哪儿打、怎么取消），返回它自己那段的 AgentResult。
BackWork = Callable[[Emit, threading.Event], "AgentResult"]

# runagent 交接时调用它：把一个工作函数交给后台，拿回任务 id。
Handoff = Callable[[BackWork], str]


class _Cancelled(Exception):
    """协作式取消。只在内部用——它在 drive 里被接住，转成 CANCELLED_ERROR。"""


def _console_emit(text: str, style: str) -> None:
    """默认的输出方式：直接打到终端。和改动之前逐字节一致。"""
    console.print(text, style=style, markup=False)


def _usage_note(response: ModelResponse | UsageTotals) -> str:
    """把这一发的用量写成一行。

    输入那三个数**不重叠**（见 ModelResponse 的注释）：input_tokens 是没走缓存的，cache_read
    是命中的，cache_creation 是往缓存里新写的——**三个加起来**才是这次请求的输入。所以总输入
    得算出来给它看：只写"输入 12,000 · 缓存 8,000"，很容易被读成后者是前者的子集，那就把账
    看错了（而这一行的全部意义就是把账看对）。

    缓存那两截**只在真发生时提**。这个后端是自动前缀缓存，常态下 cache_creation 是 0，每次都
    印一个 0 只是噪音。
    """
    parts = []
    if response.cache_read_input_tokens:
        parts.append(f"缓存命中 {response.cache_read_input_tokens:,}")
    if response.cache_creation_input_tokens:
        parts.append(f"新写缓存 {response.cache_creation_input_tokens:,}")

    size = (
        response.input_tokens
        + response.cache_read_input_tokens
        + response.cache_creation_input_tokens
    )
    if not parts:
        return f"输入 {size:,} · 输出 {response.output_tokens:,}"
    breakdown = " + ".join([f"原价 {response.input_tokens:,}", *parts])
    return f"输入 {size:,}（{breakdown}）· 输出 {response.output_tokens:,}"


def _as_text(value: Any) -> str:
    """压成一行文本。非字符串走 JSON，这样一来换行会被转义，不会把流水账冲散。"""
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)


def _render_call(call: ToolCall) -> str:
    """渲染一次工具调用，例如 `→ list_files({"path": "."})`。"""
    return f"→ {call.name}({_as_text(call.arguments) if call.arguments else ''})"


def _render_result(result: ToolResult) -> str:
    """渲染一次工具结果，失败时用 ✗ 开头。"""
    mark = "✗" if result.is_error else "←"
    return f"{mark} {result.name}: {_as_text(result.result)}"


def _is_parallel(call: ToolCall, registry: ToolRegistry) -> bool:
    """这条调用能不能和相邻的只读调用一起跑。

    判据是 `Tool.read_only`：它保证"不写盘、不跑命令、不向用户输出、不改长期记忆"，而且
    `decide_permission` 对这类工具一律 allow——所以放到 worker 线程里跑 `_gate` 也不会弹问题。
    查不到的调用名按串行处理（`dispatch` 本来就会回一条错误结果）。
    """
    tool = registry.tools.get(call.name)
    return tool is not None and tool.read_only


def _segments(calls: list[ToolCall], registry: ToolRegistry) -> list[list[ToolCall]]:
    """按原顺序切成若干段：连续只读的一段，每个不可并行的调用单独一段。

    段与段串行，段内并行。**不能把读和写混在一段里**：模型把它们排成这个顺序，串行语义下
    后面的读要看到前面的写；跨过写去并行会改变结果。
    """
    segments: list[list[ToolCall]] = []
    run: list[ToolCall] = []
    for call in calls:
        if _is_parallel(call, registry):
            run.append(call)
            continue
        if run:
            segments.append(run)
            run = []
        segments.append([call])
    if run:
        segments.append(run)
    return segments


def _run_parallel(
    calls: list[ToolCall],
    run_one: Callable[[ToolCall], ToolResult],
    limit: int,
) -> list[ToolResult]:
    """并发跑一段只读调用，按原下标返回结果。

    每条一个 daemon 线程、用一个信号量限并发。**不用 ThreadPoolExecutor**：它的线程不是
    daemon，Ctrl+C 之后解释器退出时会 join，慢的 web_fetch 会把进程挂住。某一条抛异常只影响
    它自己（包成 `is_error` 的结果），同段其它调用照常跑完。
    """
    results: list[ToolResult | None] = [None] * len(calls)
    semaphore = threading.Semaphore(limit)

    def worker(index: int, call: ToolCall) -> None:
        with semaphore:
            try:
                results[index] = run_one(call)
            except Exception as exc:  # noqa: BLE001 - 一条坏了不能带走同段其它调用
                results[index] = ToolResult(
                    id=call.id,
                    name=call.name,
                    result=f"{type(exc).__name__}: {exc}",
                    is_error=True,
                )

    threads = [
        threading.Thread(target=worker, args=(index, call), daemon=True)
        for index, call in enumerate(calls)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    return [
        result
        if result is not None
        else ToolResult(
            id=call.id,
            name=call.name,
            result="并行执行没有返回结果",
            is_error=True,
        )
        for call, result in zip(calls, results, strict=True)
    ]


def _ask(prompter: Prompter, request: PermissionRequest, grants: Grants) -> Answer:
    """真的去问一次用户。本会话批过的就不问了，直接放行。

    问**哪一种**、什么时候记 Grants 的细节在 `interaction.ask_permission` 里——读键盘线程服务
    后台请求时用的是同一份，两处不能各写一套。
    """
    return ask_permission(prompter, request, grants)


def _gate(
    call: ToolCall,
    ctx: ToolContext,
    prompter: Prompter,
    grants: Grants,
) -> ToolResult | None:
    """权限闸门：放行返回 None，不放行则返回要回给模型的那条结果。

    档位和 `allow_shell` **每次现读 `ctx`**，不从 `runagent` 的参数里抓一份：模型可以在
    回合中途调用 `set_permission_mode` 改它，同一轮后面的调用必须按新档位判。

    只有判定说"ask"的时候才轮到这里问用户（先查本会话批过没有）。**Grants 只能把 ask
    变成 allow，翻不了 deny**——在 default 下批过的东西，切到 plan 照样不给改。

    拒绝走**结果**不走异常：模型收到"用户不同意"才能换个做法，抛异常是把一次正常的人机
    协商变成崩溃。也不让它原样重试——那只会再被拒一次。
    """
    request = PermissionRequest(
        tool_name=call.name, tool_args=call.arguments, sandbox=ctx.sandbox
    )
    decision = decide_permission(request, ctx.allow_shell, ctx.permission_mode)

    blocked: str | None = None
    if decision.decision == "deny":
        blocked = f"这次调用没有放行：{decision.message}"
    elif decision.decision == "ask":
        answer = _ask(prompter, request, grants)
        if not answer.allowed:
            blocked = f"用户拒绝了这次调用（{decision.message}）"
            if answer.note:
                # 用户写下来的理由比我们替他猜的准，原样带回去。
                blocked += f"。用户说：{answer.note}"

    if blocked is None:
        return None
    return ToolResult(
        id=call.id,
        name=call.name,
        result=f"{blocked}。换个不需要它的做法，别原样重试。",
        is_error=True,
    )


def runagent(
    prompt: str,
    provider: Provider,
    registry: ToolRegistry | None = None,
    system: str | None = None,
    refresh_system: Callable[[str], str] | None = None,
    max_steps: int = 20,
    cwd: Path | None = None,
    allow_shell: bool = False,
    permission_mode: str = "default",
    prompter: Prompter | None = None,
    grants: Grants | None = None,
    messages: list[dict[str, Any]] | None = None,
    spill_dir: Path | None = None,
    label: str = "",
    read_state: ReadFileState | None = None,
    allow_fork: bool = False,
    fork_depth: int = 0,
    tool_schemas: list[dict[str, Any]] | None = None,
    background_after: float | None = None,
    handoff: Handoff | None = None,
    emit: Emit | None = None,
    notices: Callable[[], str | None] | None = None,
    allow_background: bool = False,
    background_prompter: Callable[[threading.Event], Prompter] | None = None,
    on_text: OnText | None = None,
    wait_hint: WaitHint | None = None,
    defer_final_usage: bool = False,
    usage: UsageMeter | None = None,
    sandbox: str = DEFAULT_SANDBOX,
    results: str = DEFAULT_RESULTS,
) -> AgentResult:
    """跑一轮对话，模型要求用工具时自动执行并把结果交回去。

    max_steps 限制模型调用次数，用满还没等到纯文本回复时 final 为 None。
    trace 是一串渲染好的文本，按发生顺序记下每次工具调用、它的结果、以及最后那条回复，
    专门用来打印给人看。
    不传 registry 就用 default_tools() 装好的那套内置工具。

    permission_mode 是权限档位（四档见 permissions.py），决定什么要问用户。它只是**起始值**：
    模型可以在回合中途调用 `set_permission_mode` 改 `ctx.permission_mode`，同一轮后面的调用
    立即按新档位判定。

    sandbox 决定 `bash` 跑在一次性容器里还是宿主机上（见 tools/shell.py）。它只是起始环境：
    子 agent 通过 `ctx.sandbox` 继承；权限判定不看它，但权限预览会用它说明命令在哪儿跑。

    `results` 决定工具结果打不打到终端（full / hidden，见 results.py）。**只影响屏幕**：
    trace 永远记完整结果，模型拿到的、落盘的、会话里记的一概不变。hidden 下出错的结果
    仍然完整打印。子 agent 通过 `ctx.results` 继承。

    `refresh_system` 可选：收当前档位、返回一份新 system。模型切档之后，**下一次请求前**用它
    重拼，让提示词里的档位说明立刻跟上；档位没变就不调用。它会重读 AGENTS.md、记忆和 git，
    所以只该在切档时触发。不给就一直用传进来的 `system`——子 agent 和 fork 的提示词是快照，
    必须原样发（见 known-issues H8）。
    prompter 是"问用户"的方式，不传就用一律拒绝的那个——非交互的跑法不该挂在提示符上。
    grants 记的是本会话已经批过的调用，**要在多轮之间共用一份**（所以建在 CLI 那一层，
    不在这儿新建）——不传就是空的，等于每轮都重新问。
    messages 是之前几轮的消息：把上一次的 result.messages 传回来就接着聊，不传就是新会话。
    传进来那份**不会**被就地改——返回的是新的一份。
    spill_dir 是"太长而落盘的工具结果往哪儿放"，CLI 那边给的是这一次会话的目录
    （`.agent/tool-results/<会话 id>/`），这样删会话时能一起端掉；不传就落在共用的目录里。
    label 是打在每一行前面的前缀（`[explore] ` 这种），**给子 agent 用的**——终端上要一眼看
    得出哪几行是哪个子 agent 打的、钱花在谁身上。它**只进 console.print，不进 trace**：
    trace 是这一轮的流水记录（要落盘、要交回调用方），不是给人实时看的东西。默认空串，
    主 agent 那条路传的就是它，输出一个字节都不变。

    read_state 是"读过哪些文件"的那份状态。**它和 messages 不一样：传进来就共用同一个对象。**
    fork 子 agent 用的是它的一份**副本**（见 core/subagent.py 的 fork 分支）——继承父 agent
    读到过什么，但跑完
    之后一条记录都不回流。

    `allow_fork` 决定这一轮 `agent` 的 `fork` 类型能不能用（它决定 `ctx.fork` 建不建）；
    能不能再往下 fork 由 `fork_depth` 挡——深了会在**运行时**回一句普通结果，而不是把
    `agent` 从工具表里删掉（删掉会改前缀，缓存就废了）。

    tool_schemas 只有 fork 传：把父 agent 已经算好的那份 schema **原样**发下去，多一个少一个
    都会让缓存前缀对不上。不传就照旧现算。

    **一批工具调用里，相邻的只读调用会并行执行**（判据是 `Tool.read_only`，见 `_segments`）；
    不可并行的调用是段与段之间的屏障，被它隔开的读不会跨过去并行。结果顺序始终是模型给出的
    顺序，`tool_result` 消息的拼法不变。

    **后台模式**（`run_in_background` / 两分钟自动转）：

    - `background_after` 是"跑够多少秒就交接"，`handoff` 是交接时把剩下的活交给谁。
      两个都给了，`drive` 会在**两发请求之间**检查一次——正在跑的工具/命令不会被半路甩掉。
      交接点上 messages 是完整的，不会留下悬空的 tool_use。
    - `emit` 是输出路由（默认直接打终端）。后台那一半拿到的是一个缓冲写入口。
    - `notices` 在每一发之前问一次"有没有后台任务完成了"；有就纯追加一条 user 消息。
    - `allow_background` 交给 `ToolContext`，`agent` 工具据此决定认不认 `run_in_background`。
    - `background_prompter` 决定后台那一半用哪份 prompter（参数是它自己的 cancel event）。
      交互会话传 `BrokerPrompter`——请求排队、由读键盘线程在持有终端时问；不传就退回一律
      拒绝的 `BackgroundPrompter`（没有 broker 的调用方、测试）。

    **流式输出**：传了 `on_text` 就流式——模型正文边生成边交给它，但循环拿到的
    `ModelResponse` 仍然来自完整消息，历史回填、工具调用、用量、stop_reason 一样不少。
    `wait_hint` 可选：请求发出到第一个正文 chunk 之间调 `begin()`，请求结束（成功、失败、
    正文已开始）后调 `end()`，给"等待提示"一个准确的窗口。推理过程不显示。
    只有面向终端的前台回合传 `on_text`；后台续跑、子 agent 和压缩摘要不传，走非流式的
    `complete()`——它们和前台共用终端，实时打字会互相插字。后台那一半交接时会强制清空。
    传了 `on_text` 的 provider 必须实现 `stream`。

    `defer_final_usage=True` 时，最终回答那一发的用量行不在这里打，而是放进
    `AgentResult.final_usage`，由调用方在打印完答案之后输出——这样流式和非流式都是
    "答案在前、用量在后"。工具中间步骤的用量不受影响，仍然紧跟着那一发打印。

    `usage` 是这一轮任务的记账器；不传就新建一个。子 agent 通过 `ToolContext.usage` 拿到
    同一个对象，所以 `AgentResult.turn_usage`（"本次任务共 N 发"）能算上它们。
    """
    registry = registry if registry is not None else default_tools()
    # 新拼一份，别就地改调用方手里那份。**得在 ctx 之前**：fork 那份现场要用它（见 ForkContext）。
    messages = [*(messages or []), {"role": "user", "content": prompt}]
    # fork 要的那份父 agent 现场。`--fork` 开着、或者我们本身就在一个 fork 里——后一种必须把
    # 工具继续挂在表上（否则工具表前缀变了、缓存全废），所以这里照样建，由 depth 挡递归。
    fork_ctx = (
        ForkContext(messages=messages, system=system, registry=registry, depth=fork_depth)
        if (allow_fork or fork_depth > 0)
        else None
    )
    # ctx 得在挑工具**之前**建：哪些工具此刻能用，是照着它判的（没开 shell 就没有 bash）。
    # 一轮之内这张表是固定的——每步重算的话，缓存前缀会被反复冲掉（工具那半占前缀一多半）。
    # 权限档位也带进来：子 agent 要继承主 agent 当前那一档（见 tools/base.py 那个字段）。
    # `_gate` 每次现读它；`set_permission_mode` 能在回合中途直接改，所以同一轮后面的调用
    # 立即按新档位判。
    ctx = ToolContext(
        cwd or Path.cwd(),
        read_state=read_state if read_state is not None else ReadFileState(),
        allow_shell=allow_shell,
        spill_dir=spill_dir,
        permission_mode=permission_mode,
        fork=fork_ctx,
        allow_background=allow_background,
        usage=usage,
        sandbox=sandbox,
        results=results,
    )
    tools = tool_schemas if tool_schemas is not None else registry.schemas(ctx)
    if fork_ctx is not None:
        # fork 拿它原样发出去；见上面 tool_schemas 那段。
        fork_ctx.tools = tools
    # prompter 不放进 ctx：问用户不是工具的能力，确认是这一层在调工具之前发的。
    prompter = prompter if prompter is not None else NullPrompter()
    grants = grants if grants is not None else Grants()
    usage = usage if usage is not None else UsageMeter()
    output: Emit = emit if emit is not None else _console_emit
    # 两分钟从**这一轮开始**算，不是从交接那一刻。排队、压缩都不在里头——runagent 一进
    # 来就是干活的起点。
    started = time.monotonic()

    def drive(
        msgs: list[dict[str, Any]],
        steps_left: int,
        *,
        emit: Emit,
        prompter: Prompter,
        cancel: threading.Event | None,
        allow_handoff: bool,
        notices: Callable[[], str | None] | None,
        on_text: OnText | None,
        wait_hint: WaitHint | None,
        defer_final_usage: bool,
        usage: UsageMeter,
    ) -> AgentResult:
        """真正跑循环。前台和后台那一半走的是同一个函数，区别只在参数。

        `allow_handoff` 为 False 时不再二次交接——后台的不能再把自己转手出去。
        `notices` 只有主 agent 传：子 agent 不该吞掉主 agent 的后台通知。
        `on_text` / `wait_hint` 只在前台传；后台那一半固定不传（见 continuation）。
        `usage` 前后台共用同一个对象——后台续跑的请求也算进这一轮。
        """
        trace: list[str] = []
        final: str | None = None
        final_streamed = False
        anchor: ContextAnchor | None = None
        # 最近一发请求有没有把**最终回答的正文**实时交出去。每发重置。
        last_text_shown = False
        # 最终回答那一发的用量行；`defer_final_usage` 为真时交给调用方在答案之后打印。
        pending_usage: str | None = None
        # 这一轮当前用的 system，以及它是按哪一档拼的。切档后下一次请求前重拼；没有
        # `refresh_system` 时永远是传进来的那一份（子 agent / fork 的快照）。
        current_system = system
        system_mode = permission_mode

        def send(sent: list[dict[str, Any]], **kwargs: Any) -> ModelResponse:
            """发一发，顺手把这一发的用量打出来。

            为什么要看这个：打的后端是**自动**前缀缓存——请求的**开头**和上一次一样，那一部分就
            按便宜价算，请求里不用传任何东西。所以这一行是"缓存到底省到钱没有"的唯一观测手段。

            而且要看的是**同一轮里多次请求之间**的变化：正常情况下后一发就是前一发再往后接一截，
            命中应该只增不减；哪一步要是掉下来了，就说明有人动了前面——那是要查的（这一行就是为
            看这件事打的）。

            **发出去的就是 `msgs` 本身**，中间没有任何裁剪。这不是随手写的：这个端点的缓存
            以**整个请求**为单位匹配，只追加就几乎全命中，中间改一处就几乎全不中（实测 18,560 →
            384，见 known-issues 的 D5）。所以"只追加"是省钱的**前提**，而这一行代码正是那个前提
            的落点——别在这儿加变换。
            """
            nonlocal anchor, last_text_shown, pending_usage, current_system, system_mode

            if cancel is not None and cancel.is_set():
                raise _Cancelled

            # 模型可能在上一步切了档：发出去之前把 system 换成新档位拼的那份。只比较档位，
            # 没变就不重拼——重拼会重读 AGENTS.md / 记忆 / git，而且前缀必然重新计费。
            if refresh_system is not None and ctx.permission_mode != system_mode:
                current_system = refresh_system(ctx.permission_mode)
                system_mode = ctx.permission_mode
            if current_system is not None:
                kwargs["system"] = current_system

            last_text_shown = False
            pending_usage = None
            if on_text is None:
                response = provider.complete(sent, **kwargs)
            else:
                started = False

                def start() -> None:
                    nonlocal started
                    if not started:
                        # 和轮末 `console.print(f"\n{final}")` 那个空行对齐。
                        on_text("\n")
                        started = True

                def handle_text(chunk: str) -> None:
                    nonlocal last_text_shown
                    start()
                    last_text_shown = True
                    on_text(chunk)

                try:
                    if wait_hint is not None:
                        wait_hint.begin()
                    response = provider.stream(sent, **kwargs, on_text=handle_text)
                finally:
                    # 异常路径也要收尾：否则后面的用量行、错误消息会粘在正文末尾。
                    if started:
                        on_text("\n")
                    if wait_hint is not None:
                        wait_hint.end()
            usage.record(response)
            used = (
                response.input_tokens,
                response.output_tokens,
                response.cache_read_input_tokens,
            )
            # 全是 0 就不吭声——那是假 provider（测试里那些），不是"这一发真的没花钱"。
            if not any(used):
                return response
            if kwargs.get("tools"):
                # 只认**带工具**的那几发（理由见 ContextAnchor）：用满步数收尾那一发不带 tools。
                anchor = ContextAnchor(
                    tokens=(
                        response.input_tokens
                        + response.cache_read_input_tokens
                        + response.cache_creation_input_tokens
                    ),
                    covered=len(sent),
                )
            usage_line = f"{label}（这一发 token：{_usage_note(response)}）"
            if defer_final_usage and not response.tool_calls:
                # 最终回答：先让调用方把答案打出来，再把这一行交出去。
                pending_usage = usage_line
            else:
                emit(usage_line, "dim")
            return response

        def ask(sent: list[dict[str, Any]], **kwargs: Any) -> ModelResponse:
            """发一发；**瞬时故障再补一次**，然后才认输。

            为什么要补：SDK 自己已经重试过 2 次了（退避大约 0.5 + 1 秒），它救不回来的说明麻烦比
            那更久。所以这一层唯一的增量就是"等得更久"——3 秒比它那一串长，又短到人还愿意等。

            几条定下来的：

            - **补在发请求这一层，不是整轮重跑**：工具可能已经跑过了（文件改过了、命令跑过了），
              整轮重来会把副作用再做一遍；"再问一句"是幂等的。
            - **只补一次**：补多了就是拿钱和时间去赌一个大概率还是不行的结果。
            - **必须说出来**：不然提示符看着像卡死了；而且那是又一次全量计费，用户得有机会按
              Ctrl+C 打断。
            - **不重试的那几类一个字都不多等**：401 再试一百次还是 401，只会把"去查 .env"这句话
              推迟几秒。
            """
            try:
                return send(sent, **kwargs)
            except TransientProviderError as exc:
                emit(f"{label}（{exc}，等 {RETRY_PAUSE_SECONDS:g} 秒自动再试一次…）", "dim")
                if cancel is not None:
                    # 后台任务被取消了就别陪着睡完这三秒。
                    if cancel.wait(RETRY_PAUSE_SECONDS):
                        raise _Cancelled from None
                else:
                    time.sleep(RETRY_PAUSE_SECONDS)
                # 补的这次再失败就往外放：外层接住，把这一轮收掉（会话不倒）。
                return send(sent, **kwargs)

        try:
            for step in range(steps_left):
                if cancel is not None and cancel.is_set():
                    raise _Cancelled
                if notices is not None:
                    notice = notices()
                    if notice:
                        # **纯追加**：不碰前面的任何一条，前缀缓存照旧。两个请求之间插一条
                        # user 消息是协议允许的（连续 user 会被当成同一轮）。
                        msgs.append(
                            {
                                "role": "user",
                                "content": (
                                    f"{notice}\n\n"
                                    "（这是后台任务的完成通知；如果合适，在回答里顺带告诉用户。）"
                                ),
                            }
                        )
                if (
                    allow_handoff
                    and handoff is not None
                    and background_after is not None
                    and time.monotonic() - started >= background_after
                ):
                    # 交接点必须落在两发请求之间：此时 msgs 以一条完整的 tool_result 或用户任务
                    # 结尾，后台接着往下跑不会留下悬空的 tool_use。
                    rest = steps_left - step
                    snapshot = list(msgs)
                    trace_snapshot = list(trace)

                    def continuation(
                        emit_fn: Emit,
                        cancel_event: threading.Event,
                        _snapshot: list[dict[str, Any]] = snapshot,
                        _rest: int = rest,
                    ) -> AgentResult:
                        # 有 broker 的交互会话给的是 BrokerPrompter（请求会被排队、由读键盘
                        # 线程去问）；没有就退回一律拒绝的 BackgroundPrompter。
                        background = (
                            background_prompter(cancel_event)
                            if background_prompter is not None
                            else BackgroundPrompter()
                        )
                        return drive(
                            _snapshot,
                            _rest,
                            emit=emit_fn,
                            prompter=background,
                            cancel=cancel_event,
                            allow_handoff=False,
                            notices=None,
                            # 后台和前台共用终端，实时打字会互相插字；而且此时用户可能正在
                            # 提示符上输入。不传 on_text，走非流式的 complete()。
                            on_text=None,
                            wait_hint=None,
                            # 后台输出进的是缓冲区，没有"调用方打印答案"这一步——用量行
                            # 必须现在就写进去，否则会丢。
                            defer_final_usage=False,
                            # 后台续跑的请求也算进这一轮（子 agent 的 meter 是同一个）。
                            usage=usage,
                        )

                    task_id = handoff(continuation)
                    return AgentResult(
                        final=None,
                        trace=trace_snapshot,
                        messages=list(msgs),
                        anchor=anchor,
                        handoff=task_id,
                    )

                response = ask(msgs, tools=tools)

                if not response.tool_calls:
                    final = response.text
                    if final is not None:
                        trace.append(final)
                        final_streamed = last_text_shown
                        # 这一条也要进 messages：不然 result.messages 结尾是用户那侧，拿它接着聊就
                        # 丢了模型最后说的话。
                        msgs.append({"role": "assistant", "content": response.assistant_content})
                    break

                # 请求刚回来就发现被取消了：这条 assistant 消息**先别回填**。回填了就得给每个
                # tool_use 配一条 tool_result，否则历史不合法——而这条响应已经没人要了。
                if cancel is not None and cancel.is_set():
                    raise _Cancelled

                # assistant 这轮的内容要原样回填：tool_use 块藏在里面，模型靠它认领结果。
                msgs.append({"role": "assistant", "content": response.assistant_content})

                def run_one(call: ToolCall) -> ToolResult:
                    # 闸门搁在派发之前：判了不放行，就不该碰到那个工具。
                    result = _gate(call, ctx, prompter, grants)
                    if result is None:
                        result = registry.dispatch(ctx, call)
                    return result

                tool_results: list[ToolResult] = []
                for segment in _segments(response.tool_calls, registry):
                    # 先把这一段的调用行按顺序打出来。并行段里这些行会全部出现在结果之前——
                    # 这就是"它们是一起跑的"在终端上的样子。
                    for call in segment:
                        trace.append(_render_call(call))
                        emit(f"{label}{_render_call(call)}", "cyan")

                    if len(segment) == 1:
                        segment_results = [run_one(segment[0])]
                    else:
                        segment_results = _run_parallel(
                            segment, run_one, PARALLEL_LIMIT
                        )

                    # 结果按原顺序拼：tool_result 虽然按 id 配对，但 trace、终端和测试都依赖
                    # 稳定顺序。
                    for result in segment_results:
                        # trace 永远记完整结果——它是这一轮的流水（出错时靠它数调用次数），
                        # 不是给人实时看的。要不要上屏只看 results，**出错的结果两种情况
                        # 都打**：藏掉失败原因，用户会以为工具跑成了。
                        trace.append(_render_result(result))
                        if results == "full" or result.is_error:
                            emit(
                                f"{label}{_render_result(result)}",
                                "red" if result.is_error else "green",
                            )
                        tool_results.append(result)

                # 同一条回复里可能有多个 tool_use，对应的 tool_result 必须合成一条 user 消息发回去。
                msgs.append(
                    {
                        "role": "user",
                        "content": [r.to_anthropic_block() for r in tool_results],
                    }
                )
            if final is None:
                instruction = "调用工具次数达到上限，不要再调用工具了，根据已有结果直接输出结果"
                msgs.append({"role": "user", "content": instruction})
                response = ask(msgs)
                final = response.text
                if final is not None:
                    trace.append(final)
                    final_streamed = last_text_shown
                    msgs.append({"role": "assistant", "content": response.assistant_content})
        except _Cancelled:
            return AgentResult(
                final=None,
                trace=trace,
                messages=msgs,
                error=CANCELLED_ERROR,
                anchor=anchor,
            )
        except ProviderError as exc:
            # 请求根本发不出去（或者发出去被拒了），这一轮到此为止。
            #
            # **但已经发生过的事要带回去**：工具可能已经跑过了——文件改过了、命令跑过了——那些
            # 消息必须跟着返回、跟着落盘。丢了的话下一轮模型会以为没发生过，把副作用再做一遍。
            #
            # 在这儿接住而不是往上抛，就是为了这个：异常路径拿不到 locals 里的 messages。
            return AgentResult(
                final=None, trace=trace, messages=msgs, error=str(exc), anchor=anchor
            )
        totals = usage.snapshot()
        turn_usage = (
            f"{label}（本次任务共 {totals.requests} 发：{_usage_note(totals)}）"
            if final is not None and totals.requests > 1
            else None
        )
        return AgentResult(
            final=final,
            trace=trace,
            messages=msgs,
            anchor=anchor,
            final_streamed=final_streamed,
            final_usage=pending_usage,
            turn_usage=turn_usage,
        )

    return drive(
        messages,
        max_steps,
        emit=output,
        prompter=prompter,
        cancel=None,
        allow_handoff=True,
        notices=notices,
        on_text=on_text,
        wait_hint=wait_hint,
        defer_final_usage=defer_final_usage,
        usage=usage,
    )
