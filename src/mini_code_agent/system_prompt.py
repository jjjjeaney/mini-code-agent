"""system prompt 怎么拼。

四段：固定的那段（身份 + 做事的约定）、当前档位那一句、仓库里 `AGENTS.md` 的原文、以及长期
记忆的清单（`docs/memory-design.md`）。

单独一个模块，是因为它要读盘、要认识档位，而 `core/agent.py` 只该拿到一个现成的字符串。
它是个纯函数——给定 cwd 和档位就有确定的输出，测起来不用起 agent。
"""

from __future__ import annotations

from pathlib import Path

from mini_code_agent.memory import index_text as memory_index
from mini_code_agent.permissions import RULES_FILE, parse_permission_mode
from mini_code_agent.tools.safety import truncate_text

_BASE = """\
你是一个 coding agent，在用户的项目目录里干活。工具就是你的手：读、搜、改、跑命令。

几条做事的约定：
- 改一个文件之前先读它——没读过的不许改，工具会拒绝。
- 工具参数里的路径都相对于当前工作目录。
- bash 跑在 Linux 容器里、项目挂在 /workspace；文件工具操作的是宿主机。两边路径长得不一样，
  别把宿主机的路径塞给容器里的命令。
- 需要用户拍板的事（偏好、取舍、没有唯一答案的设计选择）用 ask_user 问他，不要替他决定。
- 改文件或跑命令可能被用户拒绝。那是他的决定：换个做法，不要原样重试。
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
    """拼一份 system prompt。

    交互那条路**每轮重拼一次**：记忆和 `AGENTS.md` 都可能中途变（模型刚 `remember` 了一条），
    拼一次用到底的话，这一轮记下的东西要重启进程才看得见。档位倒是变不了（现在还没有中途
    换档的入口）——哪天加了，这份照样得跟着重建。
    """
    mode = parse_permission_mode(permission_mode)
    parts = [_BASE, _MODE_HINTS[mode]]
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
