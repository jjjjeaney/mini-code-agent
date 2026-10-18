"""system prompt 怎么拼。

**分两半，静态那半在最前面：**

- **静态的**（`_STATIC`）：身份、做事的规矩、工具怎么选、说话什么调子。同一份给所有项目所有
  用户，**一个字都不随运行变**。放最前面是为了缓存——这个后端是自动前缀缓存，前缀一样就按
  便宜价算（实测：这个项目的提示词 + 工具 schema 那一大段里，2560 个 token 走缓存、只有 236
  按原价）。所以这半写得细一点是划算的，但它**每一发都要发**，不是草稿纸——加一条就问一次
  "模型照着这条会做错什么事，不加会做错什么事"。
- **动态的**：这次运行的实情（日期/平台/模型）、当前档位、从 cwd 到仓库根那几份 `AGENTS.md`
  的原文、技能清单、长期记忆的清单、git 状态。按"变得越勤越靠后"排，这样改一处不会把它后面
  的缓存全冲掉（git 状态最勤——模型每改一个文件它就变，所以垫底）。

单独一个模块，是因为它要读盘、要认识档位，而 `core/agent.py` 只该拿到一个现成的字符串。
它只读**环境**——读盘、读几个环境变量、跑几条只读的 git 命令——不碰 provider、不碰会话状态；
给定 cwd、档位、模型和当时的仓库状态就有确定的输出，测起来不用起 agent（起个临时仓库就够）。

**子 agent 那份也在这儿拼**（`build_subagent_prompt`）：它的静态头是另一份（身份不同，
而且它问不了用户），但 `project_context` 那几段和主 agent 共用——"这个项目自己是什么样"
对谁都是一样的。
"""

from __future__ import annotations

import os
import platform
import subprocess
from datetime import date
from pathlib import Path

from mini_code_agent.memory import index_text as memory_index
from mini_code_agent.permissions import RULES_FILE, parse_permission_mode
from mini_code_agent.skills import index_text as skill_index
from mini_code_agent.tools.safety import truncate_text

# 跑 git 那几条只读命令最多等多久。仓库太大时 `git status` 可能要好几秒——等不到就当没这回事
# （见 _git_note：拿不到就整段不写，不半途拼一句出来）。
GIT_TIMEOUT_SECONDS = 5.0

# 静态那半。**不出现任何随项目、随运行变的东西**——不提名具体文件、不提档位、不提记忆。
# 那些在下半段各自的位置上。这条界线要守住：混进一个动态的东西，它后面的缓存就全废了。
#
# "什么时候派子 agent"那一段也在这半里，而且**只能在这半**：`_MODE_HINTS` 是主 agent 和
# 子 agent **共用**的，往那儿写等于让子 agent 也去派子 agent——它池子里压根没有那个工具。
# 反过来，子 agent 那份静态头是 `_SUBAGENT_STATIC`，收不到这一段，正好。
_STATIC = """\
你是一个跑在终端里的 coding agent，在用户的项目目录里干活。工具就是你的手：读、搜、改、跑命令。

## 身份

- 你是干活的，不是聊天的：该看的文件看，该改的改，把事情做完，然后说清做了什么。
- 只说你验证过的事——没读过的文件别描述它的内容，没跑过的命令别说它过了。
- 拿不准的（用户想要什么、两个方案选哪个）就问，别替他决定。

## 系统

- 有些调用会**先问用户**（改文件、跑命令）。被拒绝是正常的人机协商，不是错误：**换个做法**，
  不要原样重试。
- 当前权限档位在下面单独一行写着，它决定"哪些调用要先问"。
- 工具参数里的路径一律相对于**当前工作目录**。

## 执行任务

- **改一个文件之前先读它。** 没读过的不许改，工具会拒绝。
- 贴着周围的写法改：命名、注释密度、惯用法都跟着这个文件已有的来，别顺手翻新。
- 做完说清改了什么、为什么；验证过就说结果（失败也算），没验证就说没验证。

## 执行操作

- `bash` 跑在**一次性容器**里：没有网络、根文件系统只读、只有项目目录挂进去（可写）、跑完就删。
  所以命令碰不到宿主机。
- **但项目目录是可写的**：`rm -rf`、重定向覆盖照样能毁掉还没提交的工作，而且那条路**不经过
  文件工具的备份**。破坏性的动作（删、覆盖、重置）动手前先想爆炸半径。
- 容器里的路径和宿主机不一样（项目在容器里挂在 `/workspace`）。**别把宿主机的路径塞给容器里
  的命令。**
- 文件工具有路径闸门：只能碰工作目录之内，`.env` 这类敏感文件读不了。

## 使用工具

- **读文件用 `read_file`，搜内容用 `grep`，找文件用 `glob` / `project_tree`。** 不要用 `cat`、
  `head`、`sed`、`find` 那类命令去干这些——文件工具有闸门、有截断说明、改动有备份，输出也更好读。
- **改文件用 `edit_file`（局部）或 `write_file`（整篇），不要用 `sed -i`。**
- **跑测试、装依赖、用命令行工具、git 操作**才用 `bash`（需要启动时打开 `--shell`）。
- `read_file` 的输出**每行前面带行号**（形如 `12 | 正文`）。那是给你按位置引用的，**不是文件
  内容**——拿它当 `old_string` 之前先把前缀削掉。
- 结果里写着"已截断"时，你手上不是全部：要看全就按提示再读一段。

## 什么时候派子 agent

**要翻很多文件才弄明白的事，交给子 agent（`agent` 工具）去做，别自己一个个读**——它读过的
文件留在**它自己**的上下文里，你只拿到一段结论。

- **用户要计划、方案、"该怎么改"时，先派 `plan` 子 agent 把现状和取舍查清楚**，再照它的
  结论讲给他听。自己一路读下来的话，那些文件会一直压在你上下文里，后面每一轮都要重发。
- 摸一块不熟的代码、查某个东西在哪儿 / 怎么实现的 / 哪儿用到了——派 `explore`。
- 整块要动手的活（改几个文件、跑测试、跑完验证）——派 `general-purpose`。

**但小事别派。** 一次 `read_file`、一条 `grep` 就够的，自己来——派出去比自己干慢，而且它
每一步都是另外的请求。

## 语气与风格

- **不用 emoji。**
- 引用代码位置写成 `路径:行号`（终端里可点击），例如 `src/cli.py:42`。
- 说结论，别复述工具输出，也别把刚读到的内容再贴一遍。

## 输出效率

- **先给答案或结果**，再给必要的解释。不要"好的，我来帮你看看…"这种开场白。
- 该短就短，别把一次改动写成报告；但也别为省字数牺牲完整性——用户看不到你的思考过程。
"""

# 每档一句。plan 那句是重点——"plan 只读"实际靠的就是它（见 known-issues F2）。
_MODE_HINTS = {
    "default": "当前权限档位 default：改文件和跑命令之前都会请用户确认。",
    "accept edits": "当前权限档位 accept edits：改文件不用确认，跑命令仍然会请用户确认。",
    "plan": (
        "当前权限档位 plan：这一档只做调查。只用读类工具和只读命令"
        "（read_file / grep / glob / list_files，以及 git status、git diff 这类不改东西的命令），"
        "不要请求改文件，也不要请求会改动任何东西的命令。把打算怎么做讲清楚，等用户切档再动手。"
    ),
    "bypass permissions": "当前权限档位 bypass permissions：什么都不会问用户，照常干活。",
}

# fork 开着的时候才加的一段。
#
# **它不能进 `_STATIC`**：那份是"同一份给所有项目所有用户、一个字都不随运行变"的，而 fork 是
# 启动时决定的开关——写进去会让没开 fork 的用户也看见一个根本不存在的工具。放在档位之后、
# 项目上下文之前：前面那几段一次运行里不动，它也是。
_FORK_HINT = """\
## 什么时候用 fork

有些事**必须带着现在这段对话的上下文**才做得对。这时候用 `fork`，不要用 `agent`：它拿到的是
和你一样的系统提示、工具表和历史（只少了你最后那条工具调用），所以背景不用重讲，缓存也直接
命中。

- **上下文依赖**：要靠前面说过的背景、读过的文件、已经定下的方向才做得对。
- **任务独立性**：想从**现在这个状态**出发，独立试另一种做法。
- **上下文成本**：这件事需要你已经付过费的那段上下文，重讲一遍既费 token 又会失真。
- 反过来，**一段自足的任务描述就能说清的，用 `agent`**——它从零开始，读完的东西不占你的
  上下文。

`fork` 是**同步**的；**同一批工具调用里别和别的工具并排**（它看不到同一批里其它调用的结果）；
**不能嵌套 fork**。它改过的文件，你之后改之前要先重读。
"""

# ——— 子 agent 的那半 ———
#
# 它**不复用上面那份 `_STATIC`**，理由不是懒：那里写着"你是干活的、拿不准就问用户"，而子
# agent **问不了用户**，身份也不是一回事。共用的只有 `project_context` 那几段——项目自己的
# 规矩、有哪些技能和记忆，对谁都得知道。
#
# 这么分还有一个好处：主 agent 提示词的**字节一个都不动**，它的缓存前缀照旧。所以以后要改
# 那份 `_STATIC`，改就是了，不会牵动这儿。
#
# 下面这份是**所有类型共用**的；每种类型自己那一段在 SUBAGENT_HEADS 里。
_SUBAGENT_STATIC = """\
你是一个**子 agent**，被主 agent 派出来做一件事。你看不到主 agent 和用户的对话，只看得到
下面这段任务。

你的最终回复会**原样交回**给主 agent，那是你唯一的产出——你的中间步骤它看不到（终端上那条
流水是打给用户看的，不是给它的）。所以结论要写全：查到了什么、在哪个文件的哪一行、还有
什么没查。别写"见上面""如前所述"这种指代，主 agent 手上没有你的上下文。

## 系统

- 有些调用会**先问用户**（改文件、跑命令）。被拒绝是正常的人机协商，不是错误：**换个做法**，
  不要原样重试。
- 当前权限档位在下面单独一行写着，它决定"哪些调用要先问"。
- 工具参数里的路径一律相对于**当前工作目录**。
- **你问不了用户。** 需要他拍板的事（两个方案选哪个、要不要动那个文件），写进最终回复里
  交给主 agent，由它去问。别自己替他决定，也别因为问不了就随手猜一个往下走。

## 做事的规矩

- **改一个文件之前先读它。** 没读过的不许改，工具会拒绝。
- 贴着周围的写法改：命名、注释密度、惯用法都跟着这个文件已有的来，别顺手翻新。
- 只说你验证过的事——没读过的文件别描述它的内容，没跑过的命令别说它过了。

## 语气与风格

- **不用 emoji。**
- 引用代码位置写成 `路径:行号`，例如 `src/cli.py:42`。
- 说结论，别复述工具输出，也别把刚读到的内容再贴一遍。
- 该短就短，但也别为省字数牺牲完整性——**主 agent 看不到你的思考过程**。
"""

# 每种类型那一段。**键要和 core/subagent.py 的 AGENT_TYPES 对得上**——那边有一条例出
# 两份键集合的测试，对不上会红。
SUBAGENT_HEADS: dict[str, str] = {
    "general-purpose": (
        "## 这件事\n\n"
        "从头做完：该看的看，该改的改，该跑的跑。你有全部工具。\n\n"
        "最后说清：改了哪些文件（写路径）、为什么这么改、验证过没有（跑没跑测试、结果是什么）。"
        "没做完的说清卡在哪儿。"
    ),
    "explore": (
        "## 这件事\n\n"
        "**在代码库里找到事实，然后报告。** 你只能读，改不了任何东西。\n\n"
        "报告里写清**路径和行号**（形如 `src/cli.py:42`）——主 agent 会照着这些去读原始文件，"
        "写不清它就得多查一遍。它要的是位置和内容，不是你的评价。\n\n"
        "没找到就说没找到，并列一下你搜过哪些地方：那本身也是信息，比猜一个答案强。"
    ),
    "plan": (
        "## 这件事\n\n"
        "**设计一个实施方案。** 你只能读，改不了任何东西。\n\n"
        "方案要具体到能直接照着做：改哪个文件的哪一处、加什么、按什么顺序、有哪几个选择、"
        "各自的代价是什么、有什么风险。别停在「建议重构一下某处」这种层面上。\n\n"
        "把关键现状写进去（相关代码现在是怎么写的、在哪儿），主 agent 才知道你为什么这么设计。"
    ),
}


def _session_note(model: str | None) -> str:
    """这次运行的实情：今天几号、什么平台、什么模型。

    **日期取到"天"、不取到秒。** 这段每轮重拼一次，取到秒就等于每轮换一个前缀——缓存从这一
    行往后全废。取到天的话，一天之内它一个字都不变。

    平台要说，因为这个项目两边不一样：`bash` 在 Linux 容器里跑，文件工具在宿主机上（这台是
    Windows）。不说的话，模型给路径、给命令会按错的那边来。

    **shell 只在环境告诉我们的时候才写**——Windows 上常常没有 `SHELL` 这个变量，那就不写：
    与其猜一个，不如不说。写的是 `SHELL` 的值，也就是**启动这次运行的那个** shell（通常就是
    用户的终端，但不是的话我们也不知道）。取 `stem` 是为了跨平台一致：Windows 上那几个带
    `.exe`，Linux 上不带，统一成 `bash` 更好认。

    结尾那句是**日期真正的用处**：模型不知道自己的训练数据截止到哪天，不提醒它就容易拿"记忆
    里的最新版本"当"现在的最新版本"答。
    """
    bits = [f"今天 {date.today().isoformat()}", f"宿主机 {platform.system()}"]
    shell = os.environ.get("SHELL")
    if shell:
        bits.append(f"终端 shell {Path(shell).stem}")
    if model:
        bits.append(f"模型 {model}")
    return (
        f"这次运行：{'，'.join(bits)}。\n"
        "（你的训练数据早于今天——“最新版本”“最近有什么变化”这类问题得查证，别凭记忆答。）"
    )


def _run_git(cwd: Path, *args: str) -> str | None:
    """跑一条**只读**的 git 命令，拿回输出（首尾空白去掉）。失败返回 None。

    三个都不能省：

    - **`-c core.quotepath=false`**：得放在**子命令前面**（放后面 git 会报 unknown switch c）。
      不加的话中文文件名会被转义成 `\\344\\275\\240` 那样，等于没给。
    - **`--no-optional-locks`**：`git status` 会顺手刷新索引，那是**在写** `.git/`。我们只是想看看。
    - **超时**：仓库大到 `git status` 要好几秒时，别把整份提示词拖住。

    失败一律返回 None——没装 git、不是仓库、超时、权限不足，在这一层不区分：分了对上面也没用，
    反正都是"这一段拿不到"。
    """
    try:
        done = subprocess.run(
            ["git", "-c", "core.quotepath=false", "--no-optional-locks", *args],
            cwd=cwd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=GIT_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.SubprocessError):
        # 没装 git（FileNotFoundError）、超时（TimeoutExpired）、cwd 不存在，都走这儿。
        return None
    if done.returncode != 0:
        return None
    return done.stdout.strip()


def _branch_of(header: str) -> str:
    """从 `git status --branch` 那第一行里取出分支名。

    三种写法：`## main`、`## main...origin/main`（有上游）、`## No commits yet on main`（空仓库）。
    HEAD 分离时是 `## HEAD (no branch)`——那种返回空串，让调用方说"没在任何分支上"。
    """
    name = header.removeprefix("## ").split("...")[0].strip()
    name = name.removeprefix("No commits yet on ").strip()
    return "" if not name or name.startswith("HEAD") else name


def _git_note(cwd: Path) -> str:
    """这个仓库现在什么样子：在哪条分支上、工作区脏不脏、最近几条提交。

    **拿不到就整段不写**——不是 git 仓库、没装 git、超时，都当这一节不存在。这跟 `AGENTS.md`
    读不出来要说一声**不一样**，得说清：那边是用户以为自己的规矩生效了、其实没有，不说就是
    骗人；这边只是少给一条背景（模型照样能干活），而每发都印一句"git 用不了"纯属占地方。

    **它排在最后一段**，因为它最勤：模型每改一个文件它就变。排最后面，变它时冲掉的缓存就只有
    它自己。

    提示词是**每轮重拼**的，它天然就是"这一轮开始时"的快照（一轮之内不会重拼）——所以开头
    明说了这句。模型在同一轮里改过文件之后，这段就旧了，它得知道。
    """
    # `--untracked-files=normal` 不能省：**宿主的全局配置能改掉这个**——
    # `status.showUntrackedFiles=no` 会让未跟踪的文件整个不出现，而输出看上去一切正常（实测）。
    # `--porcelain` 保证的是格式稳定，不保证这些开关还是默认值。
    status = _run_git(cwd, "status", "--porcelain", "--branch", "--untracked-files=normal")
    if status is None:
        return ""

    lines = status.splitlines()
    branch = _branch_of(lines[0]) if lines else ""
    changes = lines[1:]
    untracked = sum(1 for line in changes if line.startswith("??"))

    where = f"分支 {branch}" if branch else "没在任何分支上（HEAD 分离）"
    if changes:
        where += f"；工作区 {len(changes) - untracked} 个改动、{untracked} 个未跟踪"
    else:
        where += "；工作区干净"

    parts = [f"这个仓库现在（**这一轮开始时**的快照）：{where}。"]
    if changes:
        # 清单可能很长，交给 truncate_text——它会在正文里注明截了、原文多长，不静默截断。
        parts.append(truncate_text("\n".join(f"  {line}" for line in changes)))
    # 空仓库（init 之后还没提交）拿不到这个，"最近提交"那半就不写，别的照写。
    # `--no-show-signature` 同理：`log.showSignature=true` 会往每条里塞签名信息，把列表搅乱。
    log = _run_git(cwd, "log", "--oneline", "-5", "--no-decorate", "--no-show-signature")
    if log:
        parts.append("最近几条提交：\n" + "\n".join(f"  {line}" for line in log.splitlines()))
    return "\n".join(parts)


def project_context(cwd: Path) -> list[str]:
    """项目自己的那几段：AGENTS.md、技能清单、记忆清单、git 状态。

    **顺序按"变得多勤"排的**（勤的靠后）：`AGENTS.md` 跟着仓库走；技能清单只在人加/删
    skill 时变；记忆清单会随 `remember` 长一截；**git 状态最勤**——模型每改一个文件它就变，
    所以垫底。这样任何一处变了，它**后面**的才掉缓存，前面的照旧命中。

    单独拎出来是因为**两个 builder 共用它**：主 agent 那份（`build_system_prompt`）和子
    agent 那份（`build_subagent_prompt`）。两边身份和档位那几段不一样，但"这个项目自己是
    什么样"完全一样——子 agent 照样得知道 AGENTS.md 里的规矩、有哪些技能和记忆可以取。

    返回的是**段**（每段一个字符串），不是拼好的整体：调用方接在自己那几段后面一起 join
    就行。它在缓存前缀里的位置是固定的，别往中间插东西。
    """
    parts: list[str] = []
    rules = _read_rules(cwd)
    if rules:
        parts.append(f"项目自己的规矩（来自 {RULES_FILE}，要遵守）：\n\n{rules}")
    skills = skill_index(cwd)
    if skills:
        # 跟记忆那节同一个道理，而且更严重：光把清单摆出来，模型不会主动去取（实测过）。
        # skill 还多一层——**它不会觉得自己需要一个自己不知道存在的东西**，所以那句"先扫一遍"
        # 必须写，而且描述得是"什么时候用得上"，不是"这是什么"。
        parts.append(
            "你有一份**技能**清单（每份只有名字和一句话，正文要按需取）。**动手之前先扫一遍**："
            "哪份对得上手头这件事，就先用 read_skill 把正文读出来，**按它写的步骤做**——"
            "别照着那一句话猜，那只是个目录条目。skill 里可能还带附件（脚本、模板、参考文档），"
            "read_skill 会列出来，要哪个用 read_skill_file 取。\n\n"
            f"{skills}"
        )
    index = memory_index(cwd)
    if index:
        # 只有清单，正文靠 read_memory 按需取——这段每轮都要发，也要稳（缓存前缀）。
        # 口气是命令式的：光把清单摆出来，模型不会主动去取（实测过），得明说"动手之前先扫
        # 一遍"。这也解释了为什么不能只列出来就算完。
        parts.append(
            "你有一份长期记忆（清单在下面，每条只有名字和一句话）。**动手之前先扫一遍**："
            "只要跟手头这件事沾一点边，就先用 read_memory 把正文读出来再决定怎么做——"
            "一句话的描述不够，别照着它猜。\n\n"
            "这些是**以前**记下来的快照。跟眼前的代码、文件、命令输出对不上时，一律**以眼前的"
            "为准**，别拿记忆去跟现状争；真撞上不一致，就提一句「这条记忆可能过期了」、问用户"
            "要不要改——不要自己改，也不要照着旧的做。\n\n"
            f"{index}"
        )
    git = _git_note(cwd)
    if git:
        parts.append(git)
    return parts


def build_system_prompt(
    cwd: Path,
    permission_mode: str,
    model: str | None = None,
    allow_fork: bool = False,
) -> str:
    """拼主 agent 的 system prompt：静态那半 + 这次运行的实情 + 档位 + 项目自己的那几段。

    前三段在这儿拼，后面那四段（AGENTS.md、技能、记忆、git）由 `project_context` 给，
    和子 agent 那份共用——**它的顺序理由写在那儿**。

    前三段的顺序也是按"变得多勤"排的：静态那半永远不变；这次运行的实情一天一变（平台、
    模型一次运行里不动）；档位会在会话里换（`/permission_mode`），但一次运行里通常不动。

    `model` 不给就不写那半句——**不猜**。调用方手上有真 provider 就传 `provider.model`；
    `Provider` 协议故意只声明了 complete（见 providers/base.py），所以这里也不假设它有。

    `allow_fork` 是启动时的开关：开着才加 fork 那段。关着的时候输出**逐字节和以前一样**，
    没开这个功能的人不会在提示词里看到一个不存在的工具。

    交互那条路**每轮重拼一次**：记忆和 `AGENTS.md` 都可能中途变（模型刚 `remember` 了一条），
    拼一次用到底的话，这一轮记下的东西要重启进程才看得见。档位同理——`/permission_mode` 能
    中途换，所以它也得每轮现读。
    """
    mode = parse_permission_mode(permission_mode)
    parts = [_STATIC, _session_note(model), _MODE_HINTS[mode]]
    if allow_fork:
        parts.append(_FORK_HINT)
    parts.extend(project_context(cwd))
    return "\n\n".join(parts)


def build_subagent_prompt(
    cwd: Path, agent_type: str, permission_mode: str, model: str | None = None
) -> str:
    """拼子 agent 的 system prompt：公用静态头 + 这个类型那段 + 档位 + 项目那几段。

    **和主 agent 那份只共用后半段。** 前面几样都不一样：它看不到主 agent 和用户的对话、
    它问不了用户，所以"拿不准就问"那句对它是有害的。

    **档位照旧要给它**：子 agent 继承主 agent 当前那一档（见 core/subagent.py），所以 plan
    档下它得知道"改文件会被拒"——不然它会一路试到把步数烧完。

    `agent_type` 认不出来就**报错，不兜底**：调用方是 harness，这个字符串是从那张表里来的，
    到这一步还不认识说明是代码写错了。模型给的那个类型在 core/subagent.py 就已经拦下了，
    变成一句给模型看的普通结果。
    """
    head = SUBAGENT_HEADS.get(agent_type)
    if head is None:
        options = "、".join(SUBAGENT_HEADS)
        raise ValueError(f"没有名为 {agent_type!r} 的子 agent 类型。可选：{options}")
    mode = parse_permission_mode(permission_mode)
    parts = [
        _SUBAGENT_STATIC,
        head,
        _session_note(model),
        _MODE_HINTS[mode],
        *project_context(cwd),
    ]
    return "\n\n".join(parts)


def _rule_files(cwd: Path) -> list[Path]:
    """从 cwd 往上收集 AGENTS.md，**到仓库根为止（含）**。仓库根在前、cwd 在后。

    找不到 `.git`（不在仓库里）就只给 cwd 那一份——**不往上走**。这一条是刻意的：往上读就是
    读 cwd 之外，而这个项目的围栏本来只有一句话，"只能碰 cwd 之内"。见下面 _read_rules。
    """
    chain = [cwd.resolve(), *cwd.resolve().parents]
    stop = 1  # 不在仓库里：只读 cwd 那份（跟没这个功能时一样）
    for index, directory in enumerate(chain):
        # `.git` 可能是目录（普通仓库）也可能是文件（worktree / submodule），exists 两种都认。
        if (directory / ".git").exists():
            stop = index + 1  # 连仓库根那一层一起读
            break
    return [directory / RULES_FILE for directory in reversed(chain[:stop])]


def _read_rules(cwd: Path) -> str:
    """把从 cwd 到仓库根这一路上所有的 AGENTS.md 拼成一段。**越靠后的越贴近当前目录。**

    为什么往上读：项目落在子目录里的时候（monorepo、或者就一层子目录），规矩写在仓库根上写
    一次就够，不用每个子目录抄一份。

    为什么**到仓库根就停**：往上读是读 cwd 之外，围栏那条线本来是"只能碰 cwd 之内"。而
    AGENTS.md 特殊——它不是给模型"读"的，是**直接进 system prompt** 的，所以"谁能往上级目录
    写一个 AGENTS.md，谁就能给这个 agent 下指令"。停在仓库根，这个面就收在这个仓库里（不到
    文件系统根，`C:\\Users\\<你>\\AGENTS.md` 那种就不会被卷进来）。

    找不到 `.git` 就只读 cwd 那一份，等于这条功能不存在。

    读不出来要说一声，别让它悄悄失效——用户以为规矩生效了、其实没有，那是骗人。哪一份出问题
    就把哪一份的路径报出来（多份的时候不说是哪一份，等于没说）。
    """
    found = []
    for path in _rule_files(cwd):
        label = os.path.relpath(path, cwd)
        text = _read_one(path, label)
        if text:
            found.append((label, text))

    if not found:
        return ""
    if len(found) == 1:
        # 只有一份就不标路径——跟没这个功能时的输出一模一样，不白多两行。
        return found[0][1]
    # 多份时每一份都标清楚，并说明顺序：模型得知道哪条更贴近眼前这个目录。
    # 末尾那个空行不能省——不然第一个 `###` 会跟引导句挤成一段（`head` 不参与下面那次 join）。
    head = "从工作目录往上到仓库根，每一层的都在下面，**越靠后的越贴近当前目录**：\n\n"
    return head + "\n\n".join(f"### {label}\n\n{text}" for label, text in found)


def _read_one(path: Path, label: str) -> str:
    """读一份规矩文件。没有它是常态，不该因此报错；读不出来要说一声，别让它悄悄失效。"""
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return ""
    except UnicodeDecodeError:
        return f"（{label} 在，但不是 UTF-8，读不出来。）"
    except OSError as exc:
        return f"（{label} 读不出来：{exc}）"
    return truncate_text(text.strip())
