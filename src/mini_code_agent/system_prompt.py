"""system prompt 怎么拼。

**分两半，静态那半在最前面：**

- **静态的**（`_STATIC`）：身份、做事的规矩、工具怎么选、说话什么调子。同一份给所有项目所有
  用户，**一个字都不随运行变**。放最前面是为了缓存——这个后端是自动前缀缓存，前缀一样就按
  便宜价算（实测：这个项目的提示词 + 工具 schema 那一大段里，2560 个 token 走缓存、只有 236
  按原价）。所以这半写得细一点是划算的，但它**每一发都要发**，不是草稿纸——加一条就问一次
  "模型照着这条会做错什么事，不加会做错什么事"。
- **动态的**：当前档位、仓库里 `AGENTS.md` 的原文、长期记忆的清单。按"变得越勤越靠后"排，
  这样改一处不会把它后面的缓存全冲掉。

单独一个模块，是因为它要读盘、要认识档位，而 `core/agent.py` 只该拿到一个现成的字符串。
它是个纯函数——给定 cwd 和档位就有确定的输出，测起来不用起 agent。
"""

from __future__ import annotations

from pathlib import Path

from mini_code_agent.memory import index_text as memory_index
from mini_code_agent.permissions import RULES_FILE, parse_permission_mode
from mini_code_agent.tools.safety import truncate_text

# 静态那半。**不出现任何随项目、随运行变的东西**——不提名具体文件、不提档位、不提记忆。
# 那些在下半段各自的位置上。这条界线要守住：混进一个动态的东西，它后面的缓存就全废了。
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


def build_system_prompt(cwd: Path, permission_mode: str) -> str:
    """拼一份 system prompt：静态那半 + 当前档位 + AGENTS.md + 长期记忆。

    顺序是**按"变得多勤"排的**（勤的靠后）：静态那半永远不变；档位会在会话里换
    （`/permission_mode`），但一次运行里通常不动；`AGENTS.md` 跟着仓库走；记忆清单最勤——
    模型每 `remember` 一条它就长一截。这样任何一处变了，它**后面**的才掉缓存，前面的照旧命中。

    交互那条路**每轮重拼一次**：记忆和 `AGENTS.md` 都可能中途变（模型刚 `remember` 了一条），
    拼一次用到底的话，这一轮记下的东西要重启进程才看得见。档位同理——`/permission_mode` 能
    中途换，所以它也得每轮现读。
    """
    mode = parse_permission_mode(permission_mode)
    parts = [_STATIC, _MODE_HINTS[mode]]
    rules = _read_rules(cwd)
    if rules:
        parts.append(f"项目自己的规矩（来自 {RULES_FILE}，要遵守）：\n\n{rules}")
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
    return "\n\n".join(parts)


def _read_rules(cwd: Path) -> str:
    """读规矩文件。没有它是常态，不该因此报错；读不出来要说一声，别让它悄悄失效。"""
    try:
        text = (cwd / RULES_FILE).read_text(encoding="utf-8")
    except FileNotFoundError:
        return ""
    except UnicodeDecodeError:
        return f"（{RULES_FILE} 在，但不是 UTF-8，读不出来。）"
    except OSError as exc:
        return f"（{RULES_FILE} 读不出来：{exc}）"
    return truncate_text(text.strip())
