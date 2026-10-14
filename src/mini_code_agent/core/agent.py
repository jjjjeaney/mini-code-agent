"""agent 主循环。"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mini_code_agent.console import console
from mini_code_agent.permissions import EDIT_TOOLS, Grants, PermissionRequest, decide_permission
from mini_code_agent.prompts import Answer, NullPrompter, Prompter
from mini_code_agent.providers.base import (
    ModelResponse,
    Provider,
    ProviderError,
    ToolCall,
    TransientProviderError,
)
from mini_code_agent.tools import ToolContext, ToolRegistry, ToolResult, default_tools
from mini_code_agent.trimming import fold_old_tool_results

# 瞬时故障（连不上、超时、限流、对面 5xx）**补一次**之前等多久。
#
# SDK 自己那 2 次重试的退避大概是 0.5 + 1 秒，所以这个数要比它明显长，否则补的那次等于白补；
# 但也不能长到用户以为卡死了。只补一次——补几次写死在 ask() 里，要改就在那儿加一层。
RETRY_PAUSE_SECONDS = 3.0


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


def _usage_note(response: ModelResponse) -> str:
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


def _ask(prompter: Prompter, request: PermissionRequest, grants: Grants) -> Answer:
    """真的去问一次用户。本会话批过的就不问了，直接放行。

    问**哪一种**由这里定：改文件的给他看改动、别的看参数（bash 也走这条，prompter 认得
    工具名，自己会去取那行命令）。渲染成什么样是 prompter 的事，但"该问哪一种"只有 harness
    知道——它手上才有工具名。

    用户答"本会话别再问"时在这里记一笔；记成什么粒度是 Grants 的事。
    """
    if grants.covers(request):
        return Answer(True)
    if request.tool_name in EDIT_TOOLS:
        answer = prompter.confirm_edit(request)
    else:
        answer = prompter.confirm_tool(request)
    if answer.allowed and answer.remember:
        # 前提是 allowed：一个"拒绝但记住"的答案不该把文件记进去。
        grants.remember(request)
    return answer


def _gate(
    call: ToolCall,
    prompter: Prompter,
    allow_shell: bool,
    permission_mode: str,
    grants: Grants,
) -> ToolResult | None:
    """权限闸门：放行返回 None，不放行则返回要回给模型的那条结果。

    只有判定说"ask"的时候才轮到这里问用户（先查本会话批过没有）。**Grants 只能把 ask
    变成 allow，翻不了 deny**——在 default 下批过的东西，切到 plan 照样不给改。

    拒绝走**结果**不走异常：模型收到"用户不同意"才能换个做法，抛异常是把一次正常的人机
    协商变成崩溃。也不让它原样重试——那只会再被拒一次。
    """
    request = PermissionRequest(tool_name=call.name, tool_args=call.arguments)
    decision = decide_permission(request, allow_shell, permission_mode)

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
    max_steps: int = 8,
    cwd: Path | None = None,
    allow_shell: bool = False,
    permission_mode: str = "default",
    prompter: Prompter | None = None,
    grants: Grants | None = None,
    messages: list[dict[str, Any]] | None = None,
    spill_dir: Path | None = None,
) -> AgentResult:
    """跑一轮对话，模型要求用工具时自动执行并把结果交回去。

    max_steps 限制模型调用次数，用满还没等到纯文本回复时 final 为 None。
    trace 是一串渲染好的文本，按发生顺序记下每次工具调用、它的结果、以及最后那条回复，
    专门用来打印给人看。
    不传 registry 就用 default_tools() 装好的那套内置工具。

    permission_mode 是权限档位（四档见 permissions.py），决定什么要问用户；
    prompter 是"问用户"的方式，不传就用一律拒绝的那个——非交互的跑法不该挂在提示符上。
    grants 记的是本会话已经批过的调用，**要在多轮之间共用一份**（所以建在 CLI 那一层，
    不在这儿新建）——不传就是空的，等于每轮都重新问。
    messages 是之前几轮的消息：把上一次的 result.messages 传回来就接着聊，不传就是新会话。
    传进来那份**不会**被就地改——返回的是新的一份。
    spill_dir 是"太长而落盘的工具结果往哪儿放"，CLI 那边给的是这一次会话的目录
    （`.agent/tool-results/<会话 id>/`），这样删会话时能一起端掉；不传就落在共用的目录里。
    """
    registry = registry if registry is not None else default_tools()
    # ctx 得在挑工具**之前**建：哪些工具此刻能用，是照着它判的（没开 shell 就没有 bash）。
    # 一轮之内这张表是固定的——每步重算的话，缓存前缀会被反复冲掉（工具那半占前缀一多半）。
    ctx = ToolContext(cwd or Path.cwd(), allow_shell=allow_shell, spill_dir=spill_dir)
    tools = registry.schemas(ctx)
    # 新拼一份，别就地改调用方手里那份。
    messages = [*(messages or []), {"role": "user", "content": prompt}]
    trace: list[str] = []
    final: str | None = None
    # prompter 不放进 ctx：问用户不是工具的能力，确认是这一层在调工具之前发的。
    prompter = prompter if prompter is not None else NullPrompter()
    grants = grants if grants is not None else Grants()
    reported_folding = False

    def sending() -> list[dict[str, Any]]:
        """该发出去的那一份：旧的、长的工具结果折掉，顺手报一次账。

        `messages` 本身留着完整的——会话文件和 result.messages 都得是全的；折的只是这一次
        请求发出去的那份。报账只报一次（每步都报会吵），但每步都得重算：历史在长。
        """
        nonlocal reported_folding
        trimmed = fold_old_tool_results(messages)
        if trimmed.count and not reported_folding:
            console.print(
                f"（折叠了 {trimmed.count} 条旧的工具结果，约 {trimmed.chars} 字符）",
                style="dim",
                markup=False,
            )
            reported_folding = True
        return trimmed.messages

    def send(sent: list[dict[str, Any]], **kwargs: Any) -> ModelResponse:
        """发一发，顺手把这一发的用量打出来。

        为什么要看这个：打的后端是**自动**前缀缓存——请求的**开头**和上一次一样，那一部分就
        按便宜价算，请求里不用传任何东西。所以这一行是"缓存到底省到钱没有"的唯一观测手段。

        而且要看的是**同一轮里多次请求之间**的变化：正常情况下后一发比前一发的开头长，命中
        应该只增不减；要是中间被什么东西改动了开头（比如折叠旧工具结果，它改的正好是消息列表
        前排），命中就会从那个位置往后全部掉成 0。这一行就是为看这件事打的。
        """

        response = provider.complete(sent, **kwargs)
        used = (response.input_tokens, response.output_tokens, response.cache_read_input_tokens)
        # 全是 0 就不吭声——那是假 provider（测试里那些），不是"这一发真的没花钱"。
        if not any(used):
            return response
        console.print(f"（这一发 token：{_usage_note(response)}）", style="dim", markup=False)
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
            console.print(
                f"（{exc}，等 {RETRY_PAUSE_SECONDS:g} 秒自动再试一次…）",
                style="dim",
                markup=False,
            )
            time.sleep(RETRY_PAUSE_SECONDS)
            # 补的这次再失败就往外放：外层接住，把这一轮收掉（会话不倒）。
            return send(sent, **kwargs)

    try:
        for _ in range(max_steps):
            response = ask(sending(), tools=tools, system=system)

            if not response.tool_calls:
                final = response.text
                if final is not None:
                    trace.append(final)
                    # 这一条也要进 messages：不然 result.messages 结尾是用户那侧，拿它接着聊就
                    # 丢了模型最后说的话。
                    messages.append({"role": "assistant", "content": response.assistant_content})
                break

            # assistant 这轮的内容要原样回填：tool_use 块藏在里面，模型靠它认领结果。
            messages.append({"role": "assistant", "content": response.assistant_content})

            results: list[ToolResult] = []
            for call in response.tool_calls:
                # 请求和它的结果挨着记，打印出来才是一条顺得下来的流水。
                trace.append(_render_call(call))
                console.print(_render_call(call), style="cyan", markup=False)

                # 闸门搁在派发之前：判了不放行，就不该碰到那个工具。
                result = _gate(call, prompter, allow_shell, permission_mode, grants)
                if result is None:
                    result = registry.dispatch(ctx, call)

                trace.append(_render_result(result))
                console.print(
                    _render_result(result),
                    style="red" if result.is_error else "green",
                    markup=False,
                )
                results.append(result)

            # 同一条回复里可能有多个 tool_use，对应的 tool_result 必须合成一条 user 消息发回去。
            messages.append({"role": "user", "content": [r.to_anthropic_block() for r in results]})
        if final is None:
            instruction = "调用工具次数达到上限，不要再调用工具了，根据已有结果直接输出结果"
            messages.append({"role": "user", "content": instruction})
            response = ask(sending(), system=system)
            final = response.text
            if final is not None:
                trace.append(final)
                messages.append({"role": "assistant", "content": response.assistant_content})
    except ProviderError as exc:
        # 请求根本发不出去（或者发出去被拒了），这一轮到此为止。
        #
        # **但已经发生过的事要带回去**：工具可能已经跑过了——文件改过了、命令跑过了——那些
        # 消息必须跟着返回、跟着落盘。丢了的话下一轮模型会以为没发生过，把副作用再做一遍。
        #
        # 在这儿接住而不是往上抛，就是为了这个：异常路径拿不到 locals 里的 messages。
        return AgentResult(final=None, trace=trace, messages=messages, error=str(exc))
    return AgentResult(final=final, trace=trace, messages=messages)
