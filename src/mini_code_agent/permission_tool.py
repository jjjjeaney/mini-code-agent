"""把"切换权限档位"包成模型能调的工具。

为什么住在 harness 层而不是 `tools/`：它要问用户（prompter）、还要改会话状态
（`ChatState`），这两样都不该进工具层——`default_tools()` 里的工具只认识 `ToolContext`，
那条边界由 `tests/test_prompt.py` 守着。装配和把活的 `ChatState` 传进来见 `cli._chat`。
"""

from __future__ import annotations

from typing import Any, Protocol

from mini_code_agent.permissions import (
    PERMISSION_MODES,
    is_looser,
    parse_permission_mode,
)
from mini_code_agent.prompts import NobodyToAskError, Prompter
from mini_code_agent.tools.base import Tool, ToolContext


class ModeState(Protocol):
    """只需要一个可写的档位字段；`commands.base.ChatState` 天然满足。

    不直接 import ChatState：那个模块反过来在命令层，和这里没有依赖关系，没必要结上。
    """

    permission_mode: str


MODE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "mode": {
            "type": "string",
            "enum": list(PERMISSION_MODES),
            "description": (
                "目标档位。plan：只读做方案，改文件会被直接拒绝；"
                "default：改文件和跑命令之前都问用户；"
                "accept edits：改文件不问、跑命令仍然问；"
                "bypass permissions：什么都不问（只有用户明确要求时才考虑）。"
            ),
        }
    },
    "required": ["mode"],
}

# 切换之后会发生什么。和 commands/builtin.py 的 GLOSS 分开写：那边是"这一档是什么"，
# 这里是"切过去之后模型的动作会怎样被判"。
_EFFECTS: dict[str, str] = {
    "plan": "改文件会被直接拒绝，跑命令仍然会问用户",
    "default": "改文件和跑命令之前都会问用户",
    "accept edits": "改文件不再问用户，跑命令仍然会问",
    "bypass permissions": "改文件、跑命令都不再问用户",
}


def _question(current: str, target: str) -> str:
    """放宽档位时问用户的那句话。**固定由 harness 生成**，模型不能自己写。"""
    if target == "bypass permissions":
        return (
            f"模型请求把权限档位从「{current}」换成「bypass permissions」。"
            "这一档下模型改文件、跑命令都不会再问你：命令能改项目目录里的任何文件，"
            "也能把 .env 读进上下文。这等于把你从回路里整个拿掉。确定同意吗？"
        )
    return (
        f"模型请求把权限档位从「{current}」换成「{target}」"
        f"（{_EFFECTS[target]}）。这是一次放宽：之后模型的动作会更少经过你。同意吗？"
    )


def _switched(target: str, *, approved: bool) -> str:
    lead = "用户同意，" if approved else ""
    extra = (
        "现在起改文件的调用会被直接拒绝，不要再请求它们。"
        if target == "plan"
        else ""
    )
    return (
        f"{lead}权限档位已切换为 {target}（{_EFFECTS[target]}）。"
        "本轮后续工具调用立即按新档位判定；系统提示和工具表仍是本轮开始时的版本"
        f"（有下一轮才会更新）。{extra}"
    )


def make_set_permission_mode(
    prompter: Prompter, *, state: ModeState | None = None
) -> Tool:
    """造一个切档工具。`state` 是交互循环里那个**活对象**，切档要写回它。

    拿的是对象不是值：`ChatState.permission_mode` 会被 `/permission_mode` 和这个工具
    反复改写，存一份副本的话下一次读到的就是旧的。
    """

    def set_permission_mode(kwargs: dict[str, Any], ctx: ToolContext) -> str:
        try:
            target = parse_permission_mode(str(kwargs.get("mode", "")))
        except ValueError as exc:
            return f"{exc}。档位仍然是 {ctx.permission_mode}。"

        current = parse_permission_mode(ctx.permission_mode)
        if target == current:
            return f"当前已经是 {target}，档位没有变化。"

        looser = is_looser(current, target)
        if looser:
            try:
                answer = prompter.single_choice(_question(current, target), ("同意", "拒绝"))
            except NobodyToAskError as exc:
                return f"没法问你，权限档位保持 {current}：{exc}"
            text = answer.strip()
            if text != "同意":
                detail = f"用户说：{text}。" if text and text != "拒绝" else ""
                return (
                    f"用户没有同意把权限档位从 {current} 换成 {target}。"
                    f"档位仍然是 {current}。{detail}"
                )

        # 先改 ctx：同一回合后面的工具调用由 _gate 现读它。再改 state：下一轮的系统提示、
        # `/permission_mode` 和汇报回合读的是那一份。
        ctx.permission_mode = target
        if state is not None:
            state.permission_mode = target
        return _switched(target, approved=looser)

    return Tool(
        name="set_permission_mode",
        description=(
            "切换这次运行的权限档位。要方案、只需只读调查时切到 plan；"
            "用户明确要求每次改文件或跑命令都先确认时切到 default。"
            "往更宽松的方向切会先问用户，他不同意就保持原档；"
            "bypass permissions 也可以切，但确认里会写明风险——只有用户明确要求时才考虑，"
            "不要主动推荐。切换立即生效：同一轮后续调用按新档位判定；"
            "系统提示和工具表下一轮才更新，切到 plan 后本轮不要再请求改文件的工具。"
        ),
        fn=set_permission_mode,
        params=MODE_SCHEMA,
        read_only=False,
    )
