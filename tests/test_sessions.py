"""sessions/transcript.py 的单元测试：落盘、读回、坏行、撞名。全部本地执行。"""

from __future__ import annotations

import json

import pytest

from mini_code_agent.sessions import SESSIONS_DIR, Session, latest, list_sessions, load, start
from mini_code_agent.sessions.transcript import PREVIEW_CHARS


def a_session(tmp_path, **kwargs):
    return start(tmp_path, **{"permission_mode": "default", "allow_shell": False, **kwargs})


def test_starting_a_session_writes_nothing(tmp_path):
    """"开工了"不算数——第一轮真攒下消息之前，盘上不该有任何东西。

    以前是启动即建，于是"打开看一眼就退出"也留下一个空文件；攒多了之后 `--continue`
    很容易接到空壳上（它取的是最新的那个名字）。
    """
    session = a_session(tmp_path)

    assert not (tmp_path / SESSIONS_DIR).exists()
    assert session.opened is False
    assert latest(tmp_path) is None


def test_the_file_appears_with_its_meta_when_the_first_messages_land(tmp_path):
    session = a_session(tmp_path, permission_mode="plan", allow_shell=True)

    session.append([{"role": "user", "content": "第一句"}])

    assert session.opened is True
    lines = session.path.read_text(encoding="utf-8").splitlines()
    assert json.loads(lines[0])["type"] == "meta"  # meta 必须是第一行
    _, meta, messages = load(tmp_path, session.id)
    assert meta["permission_mode"] == "plan"
    assert meta["allow_shell"] is True
    assert meta["cwd"] == str(tmp_path)
    assert messages == [{"role": "user", "content": "第一句"}]
    assert session.path.name == f"{session.id}.jsonl"


def test_appended_messages_come_back_whole(tmp_path):
    """原样进、原样出——工具调用和结果那些嵌套块一个都不能丢。"""
    session = a_session(tmp_path)
    sent = [
        {"role": "user", "content": "改一下 a.py"},
        {"role": "assistant", "content": [{"type": "text", "text": "好"}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1"}]},
    ]

    session.append(sent)
    _, _, messages = load(tmp_path, session.id)

    assert messages == sent


def test_non_ascii_survives(tmp_path):
    session = a_session(tmp_path)

    session.append([{"role": "user", "content": "中文 + emoji 🎯"}])

    _, _, messages = load(tmp_path, session.id)
    assert messages[0]["content"] == "中文 + emoji 🎯"


def test_two_sessions_in_the_same_second_do_not_collide(tmp_path):
    first = a_session(tmp_path)
    second = a_session(tmp_path)

    first.append([{"role": "user", "content": "先来的"}])
    second.append([{"role": "user", "content": "后来的"}])

    assert second.id != first.id
    assert "先来的" in first.path.read_text(encoding="utf-8")  # 先来那份没被盖掉
    assert "后来的" in second.path.read_text(encoding="utf-8")
    assert latest(tmp_path) == second.id


def test_the_name_is_settled_at_the_first_write_not_at_start(tmp_path):
    """两个同秒启动的会话，在启动那一刻都还看不见对方——名字得等真写的时候再定一次。

    这里把两个 Session 都按同一个苗子 id 造出来，为的是不靠"两次 start() 落在同一秒"这种
    时钟上的运气。
    """
    stamp = "20260101-000000"
    meta = {"type": "meta", "cwd": str(tmp_path)}
    first = Session(id=stamp, path=tmp_path / SESSIONS_DIR / f"{stamp}.jsonl", meta=meta)
    second = Session(id=stamp, path=tmp_path / SESSIONS_DIR / f"{stamp}.jsonl", meta=meta)

    first.append([{"role": "user", "content": "先来的"}])
    second.append([{"role": "user", "content": "后来的"}])

    assert second.id == f"{stamp}-1"
    # 关键的一条：后来那个没有把内容追进先来那份文件里
    assert "后来的" not in first.path.read_text(encoding="utf-8")
    assert "后来的" in second.path.read_text(encoding="utf-8")


def test_appending_after_a_load_does_not_clobber_the_file(tmp_path):
    """`load()` 回来的会话盘上已经有一份了——惰性建文件那条路不能拿 "w" 把它重来一遍。"""
    opened = a_session(tmp_path)
    opened.append([{"role": "user", "content": "第一句"}])

    session, _, _ = load(tmp_path, opened.id)
    session.append([{"role": "user", "content": "第二句"}])

    _, _, messages = load(tmp_path, session.id)

    assert [message["content"] for message in messages] == ["第一句", "第二句"]


def test_a_session_with_nothing_in_it_is_never_the_latest(tmp_path):
    """空壳（只剩 meta 那行）不算数——接上也没有东西可接。新代码不再造它们，但旧的还在。"""
    real = a_session(tmp_path)
    real.append([{"role": "user", "content": "真聊过"}])
    # 手造一个名字更晚、但没有任何对话的残骸
    (tmp_path / SESSIONS_DIR / "29991231-235959.jsonl").write_text(
        '{"type": "meta", "cwd": "."}', encoding="utf-8"
    )

    assert latest(tmp_path) == real.id


def test_load_without_an_id_takes_the_latest(tmp_path):
    first = a_session(tmp_path, permission_mode="plan")
    first.append([{"role": "user", "content": "早的"}])
    second = a_session(tmp_path)
    second.append([{"role": "user", "content": "最近的"}])

    session, meta, messages = load(tmp_path)

    assert session.id == second.id
    assert meta["permission_mode"] == "default"
    assert messages == [{"role": "user", "content": "最近的"}]


def test_no_sessions_at_all(tmp_path):
    assert latest(tmp_path) is None

    with pytest.raises(FileNotFoundError, match="还没有任何会话"):
        load(tmp_path)


def test_an_unknown_id_says_so(tmp_path):
    """找不到就直接报错——静默开一个新的会让人以为接上了。"""
    a_session(tmp_path).append([{"role": "user", "content": "一"}])

    with pytest.raises(FileNotFoundError, match="没有 nope 这个会话"):
        load(tmp_path, "nope")


def test_a_torn_last_line_is_skipped(tmp_path):
    """崩在写的中途最多留半行；读的时候跳过它，前面的照样能用。"""
    session = a_session(tmp_path)
    session.append([{"role": "user", "content": "第一句"}])
    with session.path.open("a", encoding="utf-8") as fh:
        fh.write('{"role": "assistant", "content": [{"ty')

    _, _, messages = load(tmp_path, session.id)

    assert messages == [{"role": "user", "content": "第一句"}]


def test_a_blank_line_is_skipped(tmp_path):
    """手工编辑过留了个空行，不该让整份会话读不出来。"""
    session = a_session(tmp_path)
    session.append([{"role": "user", "content": "第一句"}])
    with session.path.open("a", encoding="utf-8") as fh:
        fh.write("\n")

    _, _, messages = load(tmp_path, session.id)

    assert len(messages) == 1


class TestListing:
    """列表：最新的在前，带上聊了多少、压过没有、开头说的是什么。"""

    def test_newest_comes_first(self, tmp_path):
        older = a_session(tmp_path)
        older.append([{"role": "user", "content": "早的"}])
        newer = a_session(tmp_path)
        newer.append([{"role": "user", "content": "新的"}])

        assert [info.id for info in list_sessions(tmp_path)] == [newer.id, older.id]

    def test_it_counts_messages_and_flags_the_summary(self, tmp_path):
        session = a_session(tmp_path)
        session.append(
            [
                {"role": "user", "content": "第一句"},
                {"role": "assistant", "content": [{"type": "text", "text": "好"}]},
            ]
        )
        session.record_summary(2, "压过了")

        (info,) = list_sessions(tmp_path)

        assert info.messages == 2
        assert info.has_summary is True

    def test_the_preview_is_the_last_thing_the_user_said(self, tmp_path):
        """第一句常常认不出这是哪次——停在哪儿才认得出来。"""
        session = a_session(tmp_path)
        session.append(
            [
                {"role": "user", "content": "给我看看那个文件"},
                {"role": "assistant", "content": [{"type": "text", "text": "好"}]},
                {"role": "user", "content": "再看看另一个"},
            ]
        )

        assert list_sessions(tmp_path)[0].preview == "再看看另一个"

    def test_a_tool_result_is_not_mistaken_for_something_the_user_said(self, tmp_path):
        """工具结果那一批的 role 也是 user，但 content 是块列表——那不是人说的话。

        这条比取第一句的时候**更要紧**：会话多半以一串工具结果收尾，滤不掉的话预览就变成
        一大坨输出。
        """
        session = a_session(tmp_path)
        session.append(
            [
                {"role": "user", "content": "这才是我说的"},
                {
                    "role": "user",
                    "content": [
                        {"type": "tool_result", "tool_use_id": "t1", "content": "一大坨输出"}
                    ],
                },
            ]
        )

        assert list_sessions(tmp_path)[0].preview == "这才是我说的"

    def test_a_long_preview_is_flattened_and_cut(self, tmp_path):
        """粘贴进来的多行 prompt 会把整个列表撕开，所以先折成一行再截。"""
        session = a_session(tmp_path)
        session.append([{"role": "user", "content": "第一行\n第二行\n" + "x" * 200}])

        preview = list_sessions(tmp_path)[0].preview

        assert "\n" not in preview
        assert preview.endswith("…")
        assert len(preview) == PREVIEW_CHARS + 1

    def test_the_ones_that_never_said_anything_are_listed_too(self, tmp_path):
        """筛掉它们是**显示**那一层的事——列表本身要能报出"另有 N 次没说过话的"。"""
        quiet = a_session(tmp_path)
        quiet.append([{"role": "user", "content": "聊过"}])
        # 名字更晚、但只有 meta 的残骸
        (tmp_path / SESSIONS_DIR / "29991231-235959.jsonl").write_text(
            '{"type": "meta", "cwd": "."}', encoding="utf-8"
        )

        assert [info.messages for info in list_sessions(tmp_path)] == [0, 1]

    def test_a_file_that_cannot_be_read_is_skipped(self, tmp_path):
        """列个表不该因为一个读不出来的文件整个报错。（建个目录冒充会话文件就行。）"""
        good = a_session(tmp_path)
        good.append([{"role": "user", "content": "好的那份"}])
        (tmp_path / SESSIONS_DIR / "20200101-000000.jsonl").mkdir()

        assert [info.id for info in list_sessions(tmp_path)] == [good.id]


class TestSummaries:
    """压缩是有损的，所以文件里留一条标注；剪枝发生在加载时。"""

    def six_messages(self, session) -> None:
        session.append([{"role": "user", "content": f"第 {i} 句"} for i in range(6)])

    def test_a_summary_prunes_what_loads(self, tmp_path):
        session = a_session(tmp_path)
        session.append(
            [
                {"role": "user", "content": "第一句"},
                {"role": "assistant", "content": [{"type": "text", "text": "好"}]},
            ]
        )
        session.record_summary(2, "前面聊了第一句")
        session.append([{"role": "user", "content": "第二句"}])

        _, _, messages = load(tmp_path, session.id)

        assert "前面聊了第一句" in messages[0]["content"]
        assert messages[1]["content"] == "第二句"
        assert len(messages) == 2

    def test_only_the_covered_part_is_pruned(self, tmp_path):
        """剪掉的是摘要盖住的那几条，留着的**一条都不能少**。

        （up_to 要是记成"文件里一共写过多少条"，留下的会被一起剪掉——这条就是钉那个。）
        """
        session = a_session(tmp_path)
        self.six_messages(session)

        session.record_summary(4, "前四句的摘要")

        _, _, messages = load(tmp_path, session.id)

        assert len(messages) == 3  # 摘要 + 留着的两条
        assert [message["content"] for message in messages[1:]] == ["第 4 句", "第 5 句"]

    def test_a_second_summary_accounts_for_the_first(self, tmp_path):
        """第二次压缩时，内存那份开头已经顶着一条合成的摘要——它不是文件里的消息，得扣一格。"""
        first = a_session(tmp_path)
        self.six_messages(first)
        first.record_summary(4, "第一次")

        # 接着上一次：内存里是 [摘要, 第4句, 第5句]
        session, _, messages = load(tmp_path, first.id)
        messages.append({"role": "user", "content": "第 6 句"})
        session.append([{"role": "user", "content": "第 6 句"}])
        session.record_summary(3, "第二次")  # 摘要 + 第4句 + 第5句

        _, _, messages = load(tmp_path, session.id)

        assert len(messages) == 2  # 新摘要 + 第 6 句
        assert "第二次" in messages[0]["content"]
        assert messages[1]["content"] == "第 6 句"

    def test_a_summary_does_not_delete_anything_from_the_file(self, tmp_path):
        """剪枝是加载时的事——文件里一条不删，人还能翻。"""
        session = a_session(tmp_path)
        session.append([{"role": "user", "content": "第一句"}])

        session.record_summary(1, "压过了")

        assert "第一句" in session.path.read_text(encoding="utf-8")

    def test_written_counts_every_message_line(self, tmp_path):
        session = a_session(tmp_path)
        session.append([{"role": "user", "content": "一"}])
        session.record_summary(1, "压")
        session.append([{"role": "user", "content": "二"}])

        assert session.written == 2
        assert load(tmp_path, session.id)[0].written == 2

    def test_the_last_summary_wins(self, tmp_path):
        """第二条摘要压的那段历史里已经含了第一条，所以只认最后一条。"""
        session = a_session(tmp_path)
        self.six_messages(session)
        session.record_summary(2, "第一次")
        session.record_summary(4, "第二次")

        _, _, messages = load(tmp_path, session.id)

        assert "第二次" in messages[0]["content"]
        assert not any("第一次" in str(message.get("content")) for message in messages)
