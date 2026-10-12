"""会话落盘：把 messages 写成 `.agent/sessions/<id>.jsonl`，也能读回来。

几条定下来的规矩，都是有意选的：

- **一行一条**：第一行是 meta（cwd / 档位 / allow_shell / 开始时间），之后每行是一条 message，
  原样存——就是发给 API 的那个结构，不做任何加工。
- **追加写**，不是每轮整篇重写：会话越长重写越贵，而且重写到一半崩了就全丢；追加最多丢最后
  一行。所以读的时候要**跳过解析不了的行**——一行 JSON 要么完整要么残，残的那行只可能是写
  到一半断了。
- **只存 messages 和 meta**（外加压缩时那行 summary）：trace 是派生物（信息都在 messages 里）；
  `Grants` 和 `read_state` 不存——重开一次会话就该重新问、重新读，把旧的 mtime 搬回来反而
  会骗人。
- **压缩是有损的，所以得记一笔**：文件里那些消息**一条不删**（它还是完整流水，人也能翻），
  只追加一行 `{"type": "summary", "up_to": N, "text": ...}`——"写到第 N 条为止，都被这段摘要
  代表了"。加载时按**最后一条** summary 剪掉它前面的消息，把摘要顶在最前面。
- id 用时间戳，和改前备份同一套。文件名自带顺序，"最近一次"不用另建索引。
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

# 和改前备份共用 .agent/：它已经在 .gitignore 里，也被挡在模型的视野之外（EXCLUDED_DIRS +
# SENSITIVE_DIRS）。
SESSIONS_DIR = Path(".agent") / "sessions"

# 摘要那条合成消息长这样。加载和压缩两边都用这个函数造，免得形状走样。
SUMMARY_PREFIX = "[前面这段对话已压缩成摘要]\n\n"


def summary_message(text: str) -> dict[str, Any]:
    """把一段摘要变成能放进 messages 的那一条。"""
    return {"role": "user", "content": f"{SUMMARY_PREFIX}{text}"}


@dataclass
class Session:
    """一次会话的落盘。id 是时间戳，path 是它那个文件。"""

    id: str
    path: Path

    # 文件里一共写了多少条消息（不含 meta 和 summary 那两行）。
    written: int = 0

    # 内存里那份的第一条**真实**消息，对应文件里的第几条；以及开头是不是顶着一条合成的摘要。
    # 这两个数是给 record_summary 换算用的：内存那份剪过枝，下标和文件对不上。
    covered_from: int = 0
    has_summary: bool = False

    def append(self, messages: Sequence[dict[str, Any]]) -> None:
        """把这一轮新增的消息追加上去。"""
        if not messages:
            return
        with self.path.open("a", encoding="utf-8") as fh:
            for message in messages:
                fh.write(json.dumps(message, ensure_ascii=False) + "\n")
        self.written += len(messages)

    def record_summary(self, covered: int, text: str) -> None:
        """记一笔：调用方手里那份 messages 的**前 covered 条**，被这段摘要代表了。

        写进文件的是**文件下标**，不是调用方那份的下标——内存那份可能已经剪过枝，而且开头还
        顶着一条合成的摘要（它不是文件里的消息，得扣掉一格）。

        文件里那些消息**一条都不删**：完整流水留着，剪枝是加载时的事。
        """
        self.covered_from += covered - (1 if self.has_summary else 0)
        self.has_summary = True
        line = {"type": "summary", "up_to": self.covered_from, "text": text}
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(line, ensure_ascii=False) + "\n")


def start(cwd: Path, *, permission_mode: str, allow_shell: bool) -> Session:
    """开一次新会话：建文件、写 meta 那一行。"""
    directory = cwd / SESSIONS_DIR
    directory.mkdir(parents=True, exist_ok=True)
    started = datetime.now()
    path = _free_path(directory, started.strftime("%Y%m%d-%H%M%S"))
    meta = {
        "type": "meta",
        "cwd": str(cwd),
        "permission_mode": permission_mode,
        "allow_shell": allow_shell,
        "started": started.isoformat(timespec="seconds"),
    }
    path.write_text(json.dumps(meta, ensure_ascii=False) + "\n", encoding="utf-8")
    return Session(id=path.stem, path=path)


def latest(cwd: Path) -> str | None:
    """最近一次的 id；一次都没有就是 None。id 是定宽时间戳，排序就是时间序。"""
    names = sorted(path.stem for path in (cwd / SESSIONS_DIR).glob("*.jsonl"))
    return names[-1] if names else None


def load(cwd: Path, session_id: str | None = None) -> tuple[Session, dict[str, Any], list[dict]]:
    """读回一次会话：Session、meta、messages。

    返回的那份 messages 是**剪过枝的**：被 summary 行盖住的老消息不在里面（文件里还在）。

    session_id 不传就取最近一次；找不到直接报错——**静默开一个新的**会让人以为接上了，
    那种错比报错难查得多。
    """
    if session_id is None:
        session_id = latest(cwd)
        if session_id is None:
            raise FileNotFoundError(f"{cwd / SESSIONS_DIR} 里还没有任何会话。")
    path = cwd / SESSIONS_DIR / f"{session_id}.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"没有 {session_id} 这个会话（找的是 {path}）。")

    meta: dict[str, Any] = {}
    messages: list[dict] = []
    summary = ""
    summary_from = 0
    for item in _read_lines(path):
        kind = item.get("type")
        if kind == "meta":
            meta = item
        elif kind == "summary":
            # 后一条涵盖前一条（它压的那段历史里已经含了前一段摘要），所以只认最后一条。
            summary = str(item.get("text") or "")
            summary_from = int(item.get("up_to") or 0)
        else:
            messages.append(item)

    kept = messages[summary_from:]
    if summary:
        kept = [summary_message(summary), *kept]
    return (
        Session(
            id=session_id,
            path=path,
            written=len(messages),
            covered_from=summary_from,
            has_summary=bool(summary),
        ),
        meta,
        kept,
    )


def _read_lines(path: Path) -> Iterable[dict[str, Any]]:
    """一行一条地读，跳过读不出来的那些（只可能是崩在写的中途）。

    空行也跳过：手工编辑过的话，末尾多个换行不该让整份会话读不出来。
    """
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def _free_path(directory: Path, session_id: str) -> Path:
    """撞名就往后排，绝不覆盖——同一秒里连开两次会撞上，而先来的那份已经记着东西了。

    和改前备份那条路同一个办法：不指望"时间戳够细"来避重（Windows 上连取两次可能拿到
    同一个值），靠计数器保证唯一。
    """
    path = directory / f"{session_id}.jsonl"
    counter = 1
    while path.exists():
        path = directory / f"{session_id}-{counter}.jsonl"
        counter += 1
    return path
