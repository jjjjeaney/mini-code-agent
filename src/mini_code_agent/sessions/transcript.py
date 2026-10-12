"""会话落盘：把 messages 写成 `.agent/sessions/<id>.jsonl`，也能读回来。

几条定下来的规矩，都是有意选的：

- **一行一条**：第一行是 meta（cwd / 档位 / allow_shell / 开始时间），之后每行是一条 message，
  原样存——就是发给 API 的那个结构，不做任何加工。
- **追加写**，不是每轮整篇重写：会话越长重写越贵，而且重写到一半崩了就全丢；追加最多丢最后
  一行。所以读的时候要**跳过解析不了的行**——一行 JSON 要么完整要么残，残的那行只可能是写
  到一半断了。
- **只存 messages 和 meta**：trace 是派生物（信息都在 messages 里）；`Grants` 和 `read_state`
  不存——重开一次会话就该重新问、重新读，把旧的 mtime 搬回来反而会骗人。
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


@dataclass
class Session:
    """一次会话的落盘。id 是时间戳，path 是它那个文件。"""

    id: str
    path: Path

    def append(self, messages: Sequence[dict[str, Any]]) -> None:
        """把这一轮新增的消息追加上去。"""
        with self.path.open("a", encoding="utf-8") as fh:
            for message in messages:
                fh.write(json.dumps(message, ensure_ascii=False) + "\n")


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
    for item in _read_lines(path):
        if item.get("type") == "meta":
            meta = item
        else:
            messages.append(item)
    return Session(id=session_id, path=path), meta, messages


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
