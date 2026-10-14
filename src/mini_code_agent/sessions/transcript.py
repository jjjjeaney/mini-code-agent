"""会话落盘：把 messages 写成 `.agent/sessions/<id>.jsonl`，也能读回来。

几条定下来的规矩，都是有意选的：

- **一行一条**：第一行是 meta（cwd / 档位 / allow_shell / 开始时间），之后每行是一条 message，
  原样存——就是发给 API 的那个结构，不做任何加工。
- **没说过话就不建文件**：`start()` 只算不写，第一轮真落了消息才建（见 `Session._open`）。
  以前是启动即建，于是"打开看了一眼就退出"也留下一个空文件——它们会攒起来，而 `--continue`
  取的是最新的那个名字，很容易接到空壳上。**文件名在第一次写的时候才定下来**，不是启动那
  一刻：推迟的只是"写"，"占名字"得等真写的时候做，不然两个同秒启动的会话会各自以为同一个
  名字是自己的。
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
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

# 和改前备份共用 .agent/：它已经在 .gitignore 里，也被挡在模型的视野之外（EXCLUDED_DIRS +
# SENSITIVE_DIRS）。
SESSIONS_DIR = Path(".agent") / "sessions"

# 列表里那句预览截多长。够认出"这是哪次"就行——它只是给你挑的，不是用来读的。
PREVIEW_CHARS = 40

# 摘要那条合成消息长这样。加载和压缩两边都用这个函数造，免得形状走样。
SUMMARY_PREFIX = "[前面这段对话已压缩成摘要]\n\n"


def summary_message(text: str) -> dict[str, Any]:
    """把一段摘要变成能放进 messages 的那一条。"""
    return {"role": "user", "content": f"{SUMMARY_PREFIX}{text}"}


@dataclass
class Session:
    """一次会话的落盘。id 是时间戳，path 是它那个文件。

    **文件是懒建的**：`start()` 不碰磁盘，第一轮真的攒下消息才写（见 `_open`）。所以"启动过
    但一句话没说"不会在盘上留下任何东西。
    """

    id: str
    path: Path

    # 落盘时要写进第一行的那个 meta。`load()` 读回来的会话没有它——那份文件里早就写过了。
    meta: dict[str, Any] = field(default_factory=dict)

    # 文件真在盘上吗。刚 `start()` 出来是 False；`load()` 回来的、以及写过一次之后的，是 True。
    opened: bool = False

    # 文件里一共写了多少条消息（不含 meta 和 summary 那两行）。
    written: int = 0

    # 内存里那份的第一条**真实**消息，对应文件里的第几条；以及开头是不是顶着一条合成的摘要。
    # 这两个数是给 record_summary 换算用的：内存那份剪过枝，下标和文件对不上。
    covered_from: int = 0
    has_summary: bool = False

    def _open(self) -> None:
        """第一次真要写之前：建目录、定下文件名、把 meta 写成第一行。只跑一次。

        **名字在这里才定下来**，不是 `start()` 那一刻。不是因为我们懒，是因为两个同秒启动的
        会话在启动时都还看不见对方——各写各的就会撞进同一个文件。等到真写的时候再按目录现在
        的样子定一次，先来的那份就占住名字了。

        meta 必须是文件的第一行，所以"建文件"和"写 meta"是同一件事，分不开。
        """
        if self.opened:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path = _free_path(self.path.parent, self.id)
        self.id = self.path.stem
        self.path.write_text(json.dumps(self.meta, ensure_ascii=False) + "\n", encoding="utf-8")
        self.opened = True

    def append(self, messages: Sequence[dict[str, Any]]) -> None:
        """把这一轮新增的消息追加上去。第一次调用会把文件（连同 meta）建出来。

        空的那次不算：`runagent` 前一步就炸了的话，一条消息都没有，这时**不该**留下一个空
        会话——没有发生过的事情不值得记一笔。
        """
        if not messages:
            return
        self._open()
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
        self._open()
        self.covered_from += covered - (1 if self.has_summary else 0)
        self.has_summary = True
        line = {"type": "summary", "up_to": self.covered_from, "text": text}
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(line, ensure_ascii=False) + "\n")


def start(cwd: Path, *, permission_mode: str, allow_shell: bool) -> Session:
    """开一次新会话。**不碰磁盘**——第一轮真攒下消息才建文件（见 Session._open）。

    id 和 path 先按时间戳做成一个"苗子"，真落盘那一刻才定下来。在那之前，盘上什么都没有。
    """
    started = datetime.now()
    stamp = started.strftime("%Y%m%d-%H%M%S")
    meta = {
        "type": "meta",
        "cwd": str(cwd),
        "permission_mode": permission_mode,
        "allow_shell": allow_shell,
        "started": started.isoformat(timespec="seconds"),
    }
    return Session(id=stamp, path=cwd / SESSIONS_DIR / f"{stamp}.jsonl", meta=meta)


def latest(cwd: Path) -> str | None:
    """最近一次**有内容**的会话的 id；一次都没有就是 None。

    空文件（只剩 meta 那行）不算：它们是"建了却从没说过话"的残骸，接上也没有东西可接。
    新代码不会再造出这种文件，但盘上以前留下的还在。

    比的是**id 字符串**、不是文件路径：同一个 id 撞了会加个 `-1` 后缀，而 `"…-1.jsonl"` 和
    `"….jsonl"` 按路径比是在 `-` 和 `.` 上分的（`-` < `.`），那样排出来 `-1` 反而更早。
    id 是定宽时间戳，按字符串排就是时间序，撞了的那份跟在它前面那次的后面。
    """
    directory = cwd / SESSIONS_DIR
    for name in sorted((path.stem for path in directory.glob("*.jsonl")), reverse=True):
        if _has_messages(directory / f"{name}.jsonl"):
            return name
    return None


def _has_messages(path: Path) -> bool:
    """这个会话文件里有没有真正的对话。

    用生成器一读到就收，不把整个文件读完——每列一次目录，每个文件都要过这一道。
    """
    for item in _read_lines(path):
        if item.get("type") not in ("meta", "summary"):
            return True
    return False


@dataclass
class SessionInfo:
    """列表里一行要用的那几样。

    **不是**用来接着聊的那份——那是 `load()` 的活儿，它要的是完整的 messages。这里只要能
    认出"这是哪一次、聊了多少、最后停在哪儿"。
    """

    id: str
    messages: int
    has_summary: bool

    # **最后**一次人说的话，不是第一句。第一句常常认不出这是哪次——实测盘上连着三次会话的
    # 第一句都是「读取main.py的文件内容」；而"我上次停在哪儿"正好是翻这个列表要回答的问题。
    # 代价是收尾那句如果只是个「继续」，单看它确实没什么信息量——那也没办法，列表只能给你
    # 一个认得出的钩子，认不出就靠 id 和条数。
    preview: str


def list_sessions(cwd: Path) -> list[SessionInfo]:
    """这个目录下有过哪些会话，最新的在前。

    每个文件完整读一遍——计数和预览都得知道里面有什么，没有更省的算法（好处是顺便把
    `_read_lines` 那套"跳过读不出来的行"也用上了，手工编辑坏了的文件不至于炸掉整个列表）。

    坏在**打不开**的文件直接跳过：列个表不该因为一个读不出来的文件整个报错。（`load()` 的
    契约正相反，它是"读不回来就报错"——那个是要接着聊，静默给个空会话比报错糟得多。）
    """
    directory = cwd / SESSIONS_DIR
    infos = []
    # 按 id 字符串排，不是按文件路径——同秒撞名那份带 `-1` 后缀，按路径比会排到前头去。
    for name in sorted((path.stem for path in directory.glob("*.jsonl")), reverse=True):
        try:
            infos.append(_describe(directory / f"{name}.jsonl"))
        except OSError:
            continue
    return infos


def _describe(path: Path) -> SessionInfo:
    """一个会话文件撮成一行。"""
    messages = 0
    has_summary = False
    preview = ""
    for item in _read_lines(path):
        kind = item.get("type")
        if kind == "meta":
            continue
        if kind == "summary":
            has_summary = True
            continue
        messages += 1
        if item.get("role") == "user":
            # 用户那句话存的是字符串；工具结果那一批虽然也是 user，但 content 是块列表，
            # 不是人说的话，不能拿来当预览。
            content = item.get("content")
            if isinstance(content, str):
                # 一路覆盖，不留第一句：最后留下的就是这个人**最后**说的一句话。
                preview = _one_line(content)
    return SessionInfo(
        id=path.stem, messages=messages, has_summary=has_summary, preview=preview
    )


def _one_line(text: str) -> str:
    """折成一行再截短。**折行是必须的**：一句粘贴进来的多行 prompt 会把整个列表撕开。"""
    flat = " ".join(text.split())
    return flat if len(flat) <= PREVIEW_CHARS else flat[:PREVIEW_CHARS] + "…"


def describe(cwd: Path, session_id: str) -> SessionInfo:
    """撮一次会话，只为看清它是什么——`--delete` 在动手之前用它。

    找不着就报错，不返回 None：调用方要拿它做决定，一个"看起来正常但什么都没有"的空壳比
    一句报错难查得多（和 `load()` 同一条理由）。
    """
    path = cwd / SESSIONS_DIR / f"{session_id}.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"没有 {session_id} 这个会话（找的是 {path}）。")
    return _describe(path)


def delete(cwd: Path, session_id: str) -> None:
    """删掉一次会话——删的是那个 jsonl 文件本身。

    **没有回收站、也没有备份，这一步不可逆**。所以调用方要**先给人看过再动手**（cli.py 里
    就是这么用的：先 describe 打印一遍、问过用户，才调这里）。

    落了盘的工具结果（`.agent/tool-results/`）**不跟着删**：那些文件名里没有会话 id，认不出
    是谁的。等它们按会话分目录了才谈得上一起删（见 docs/known-issues.md 的 G2）。
    """
    # 先撮一遍再删：报错的措辞和 describe 一致，而"文件在不在"这件事也只判一次。
    describe(cwd, session_id)
    (cwd / SESSIONS_DIR / f"{session_id}.jsonl").unlink()


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
            # **必须是 True**：这份文件已经在盘上了。标成 False 的话，下一次 append 会走
            # _open 那条路，而它是拿 "w" 建文件的——把读上来的整个会话当场截掉。
            opened=True,
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
