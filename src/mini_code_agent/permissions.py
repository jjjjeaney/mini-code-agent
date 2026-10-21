"""权限模式：用户指定的那一档，以及判定要用的几个类型。

四档，和 Claude Code 同名同义：

- `default`：改文件和跑命令都停下来问
- `accept edits`：改文件不问（自动放行），跑命令仍然要问
- `plan`：不给改文件——计划模式，先让模型把打算怎么做讲清楚，再切出去执行
- `bypass permissions`：什么都不问

**bypass permissions 是唯一一档把用户从回路里拿掉的**，所以启动时显式给最好；会话里唯一
能进去的路是模型调用 `set_permission_mode`、并在用户确认过写明风险的提示之后生效——
`/permission_mode` 命令仍然拒绝这一档。开了之后模型跑命令不再经过用户，而命令能改项目里
任何文件、也能把 `.env` 读进上下文（见 known-issues A1 / A2）。它放行的只是"要不要问"这一层
——工具自己的闸门（围栏、符号链接、敏感路径、读-before-写、改前备份）照样管用：权限不是
授权，批准过的事也不该能写出围栏。

bash 跟不跟档位走，看档位：`default` 和 `accept edits` 下它一律要问（一条命令会干什么，
用户事先看不出来），`bypass permissions` 下不问，`plan` 下仍然要问——那一格还没定，见
known-issues F2。

这些是**默认**答案：模式定的是"不问的时候怎么办"，具体到某一次调用还可能被别的条件改成
要问（比如本会话批过的文件就不再问，见 Grants）。这个模块管三件事：把用户敲的档位解析成
规范值、判定要用的请求/结果两个类型、以及本会话已经批过的调用；判定逻辑本身在
decide_permission 里，它不认识 Grants。
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass, field
from typing import Literal, get_args

# 四个档位。参数标注、parse_permission_mode 的返回值、报错里列出的可选值都指向这一处，
# 以后想加第五档只改这里。
PermissionMode = Literal["accept edits", "bypass permissions", "default", "plan"]

PERMISSION_MODES: tuple[PermissionMode, ...] = get_args(PermissionMode)

# 档位从紧到松的顺序。**只用来判"这次切换要不要征求同意"**（见 permission_tool）：
# `decide_permission` 的判定表不是一个简单全序（bash 在 plan/default/accept 下都问），
# 别拿这个顺序去重写那张表。
PERMISSION_RANK: dict[PermissionMode, int] = {
    "plan": 0,
    "default": 1,
    "accept edits": 2,
    "bypass permissions": 3,
}

# 命令行上省事的短写法。**整词匹配**（归一化之后），不做前缀——`bypass` 是这一档的约定
# 短名，`bypas` 不是。键也留着归一化之后的样子（没有空格、小写）。
SHORT_FORMS: dict[str, PermissionMode] = {
    "accept": "accept edits",
    "bypass": "bypass permissions",
}


# 会改磁盘的那几个工具。模式判定（哪一档管它们）和"该给用户看改动还是看参数"都按这一份
# 名单分——两处各写一份迟早会漂。
EDIT_TOOLS = ("write_file", "edit_file", "replace_lines")

# 仓库里放"规矩"的那个文件：system_prompt.py 会把它的原文拼进 system prompt。改它要用户点头，
# **即使在 accept edits 档下**——它是"以后每一次都生效"的东西，和改一行代码不是一类。
RULES_FILE = "AGENTS.md"


@dataclass
class PermissionRequest:
    tool_name: str
    tool_args: dict


@dataclass
class PermissionDecision:
    decision: Literal["allow", "deny", "ask"]
    message:str

def parse_permission_mode(value: str) -> PermissionMode:
    """把用户敲的那一档解析成规范值。

    有两档的名字中间带着空格，命令行上敲起来别扭，所以这几种写法都算同一个：
    `accept edits`、`accept_edits`、`accept-edits`、`acceptEdits`——比的是"只留下字母数字、
    并转成小写"之后的样子，因此大小写和空格对数都不计较。`bypass permissions` 同理。

    看不懂的词直接报错，并把可选值列出来。**不比默认值兜底**：兜底的后果是用户以为
    切到了某一档、实际在另一档上跑，那种误解不该由默认值来消化。
    """
    if not isinstance(value, str):
        raise ValueError(f"权限模式得是字符串，收到的是 {type(value).__name__}：{value!r}")
    squashed = "".join(ch for ch in value.lower() if ch.isalnum())
    for mode in PERMISSION_MODES:
        if squashed == mode.replace(" ", ""):
            return mode
    if squashed in SHORT_FORMS:
        return SHORT_FORMS[squashed]
    options = "、".join(f"{mode!r}" for mode in PERMISSION_MODES)
    shorts = "、".join(f"{name!r}" for name in SHORT_FORMS)
    raise ValueError(f"{value!r} 不是权限模式。可选：{options}（短写法：{shorts}）")


def permission_rank(mode: str) -> int:
    """档位的松紧序号：plan 最紧，bypass 最松。"""
    return PERMISSION_RANK[parse_permission_mode(mode)]


def is_looser(current: str, target: str) -> bool:
    """从 current 换到 target 是不是一次放宽（需要用户当次同意）。

    同档返回 False；写法和 `parse_permission_mode` 一样宽松。
    """
    return permission_rank(target) > permission_rank(current)


def decide_permission(
    request: PermissionRequest, allow_shell: bool, permission_mode: str
) -> PermissionDecision:
    """判一次调用该不该放行：allow / deny / ask。

    - `bypass permissions` 一律放行，最先短路。它是唯一一档不把用户放进回路的——但也只管
      "要不要问"这一层：工具自己的闸门（围栏、敏感路径、读-before-写）不归它管
    - bash：`default` / `accept edits` / `plan` 下都要问用户（一条命令会干什么，用户事先
      看不出来）。没开 shell 时直接放行——它本来就会报"没开命令执行"，先弹一句询问、答完
      才发现跑不了，是白问
    - 其余的工具一律放行——读文件、列目录、搜索本来就只受路径闸门管，逐个问没人受得了
    - 改文件的那几个（EDIT_TOOLS）看档位：`accept edits` 放行、`default` 问、`plan` 拒绝
    - 有一个例外：改**规矩文件**（AGENTS.md）在 `accept edits` 下也要问——那是"以后每一次
      都生效"的东西，和改一行代码不是一类
    - 还有一个：**写长期记忆**（remember）不论哪一档都要问。同理——它也是"以后每次会话都
      生效"的，模型悄悄给自己立规矩比改错一行代码严重

    "ask" 不是答案，是"这事得问用户"；真的去问、以及用户说不行之后怎么办，在 harness 那层。

    参数都是权限自己的东西（工具名、shell 开没开、档位），不认识 ToolContext 那种工具层的
    环境对象——权限不是工具的属性。

    **档位现在是那个"工具层环境对象"里的一个字段，但上面这条没破**（`ToolContext` 里确实
    带着 `permission_mode`，见 tools/base.py 那个字段的说明）。之所以不冲突，是因为要在
    这里分清两样东西：

    - **能力**——"工具能不能自己问用户"，也就是 prompter。它照旧被挡在工具层外面，
      `ask_user` 依然是 harness 在 CLI 层注入的，test_prompt.py 里有用例钉着这条。
    - **环境**——"工具此刻跑在什么条件下"，也就是 cwd、allow_shell、以及档位。档位属于
      这一类，它进 `ToolContext` 是为了让**子 agent 继承主 agent 当前那一档**（不继承的话，
      plan 档下派一个 general-purpose 子 agent 就能改文件）。

    这个函数的签名因此一个字母都没变：它照旧只收那个字符串，不认识 ToolContext。
    """
    mode = parse_permission_mode(permission_mode)

    if mode == "bypass permissions":
        return PermissionDecision("allow", "")

    if request.tool_name == "remember":
        return PermissionDecision("ask", "写长期记忆要用户确认")

    if request.tool_name == "bash":
        if not allow_shell:
            # **这一支实际到不了**：没开 shell 时 bash 压根不发给模型（见 tools/base.py 的
            # Tool.available）。留着是因为这个函数单独拿出来也得是对的——它有自己的测试，
            # 而且"没开的工具直接放行"这个判断本身没毛病（先弹一句问、答完才发现跑不了，是白问）。
            return PermissionDecision("allow", "")
        return PermissionDecision("ask", "跑命令要用户确认")

    if request.tool_name not in EDIT_TOOLS:
        return PermissionDecision("allow", "")

    if mode == "plan":
        return PermissionDecision("deny", "plan 模式只读，不给改文件")
    if is_rules_file(request):
        # 排在 plan 后面：plan 是"拒"，比"问"更严，不该被这条翻过来。
        return PermissionDecision("ask", f"改 {RULES_FILE} 要确认——它是以后每次都生效的规矩")
    if mode == "accept edits":
        return PermissionDecision("allow", "")
    return PermissionDecision("ask", "改文件要用户确认")


def is_rules_file(request: PermissionRequest) -> bool:
    """这次调用是不是在改那个"规矩文件"（AGENTS.md）。

    只比文件名、不分大小写。宁可多问一次——Windows 上 `agents.md` 和 `AGENTS.md` 本来就是
    同一个文件，在 Linux 上多问一次也只是问一次。
    """
    path = request.tool_args.get("path")
    if not isinstance(path, str) or not path:
        return False
    return os.path.basename(os.path.normpath(path)).lower() == RULES_FILE.lower()


def _grant_key(request: PermissionRequest) -> tuple[str, str] | None:
    """这次调用能不能被记住；能的话，键是什么。

    只有改文件的那几个、而且带了 path 的才记。路径要**归一化**：同一个文件模型可能写成
    a.py、./a.py、src/../a.py，不归一化就会重复问——重复问是安全的那一侧，但很烦。
    normpath 只做字符串归约，不碰文件系统、也不判围栏（那是工具的事）；两边都用它，
    写法的差异就抹平了。
    """
    if request.tool_name not in EDIT_TOOLS:
        return None
    path = request.tool_args.get("path")
    if not isinstance(path, str) or not path:
        return None
    return request.tool_name, os.path.normpath(path)


@dataclass
class Grants:
    """本会话里用户说过"别再问了"的那些调用。进程退出就没了，不落盘。

    键是 (工具名, 路径)。按工具名记太宽了——"允许 write_file"等于允许写**任何**文件，
    比逐次问还危险；按路径记才最小可用：同一个文件连着改几处只问一次，换个文件再问。

    只记改文件那几个：bash 不在内——命令之间差异太大，按整条记没意义，按程序名
    （git / pytest）记又太粗，而它是唯一没有 diff 可看的调用，逐条点头是对的。
    """

    entries: set[tuple[str, str]] = field(default_factory=set)

    # 后台子 agent 也会走到这儿（`_gate` 先查 `covers`），而主线程可能在 `remember`、
    # `/grants` 可能在遍历。单条 `in` / `add` 在 CPython 下本身是原子的，但"遍历时另一条
    # 线程加了一条"会让 `sorted(entries)` 抛 RuntimeError——所以统一加锁，遍历走 snapshot()。
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)

    def covers(self, request: PermissionRequest) -> bool:
        key = _grant_key(request)
        if key is None:
            return False
        with self._lock:
            return key in self.entries

    def remember(self, request: PermissionRequest) -> None:
        """记下这次批准。记不了的（比如 bash）就当没说——调用方不必先判能不能记。"""
        key = _grant_key(request)
        if key is not None:
            with self._lock:
                self.entries.add(key)

    def clear(self) -> None:
        """把这一会话批过的全忘掉。之后同一个文件会重新问你——只会多问，不会少问。"""
        with self._lock:
            self.entries.clear()

    def snapshot(self) -> list[tuple[str, str]]:
        """排好序的一份副本。**不要在外面直接遍历 entries**：后台线程可能在改它。"""
        with self._lock:
            return sorted(self.entries)
