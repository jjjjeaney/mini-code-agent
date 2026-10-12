"""sessions/transcript.py 的单元测试：落盘、读回、坏行、撞名。全部本地执行。"""

from __future__ import annotations

import pytest

from mini_code_agent.sessions import latest, load, start


def a_session(tmp_path, **kwargs):
    return start(tmp_path, **{"permission_mode": "default", "allow_shell": False, **kwargs})


def test_a_new_session_writes_its_meta(tmp_path):
    session = a_session(tmp_path, permission_mode="plan", allow_shell=True)

    _, meta, messages = load(tmp_path, session.id)

    assert meta["permission_mode"] == "plan"
    assert meta["allow_shell"] is True
    assert meta["cwd"] == str(tmp_path)
    assert messages == []
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

    assert second.id != first.id
    assert first.path.read_text(encoding="utf-8") != ""  # 先来那份没被盖掉
    assert latest(tmp_path) == second.id


def test_load_without_an_id_takes_the_latest(tmp_path):
    a_session(tmp_path, permission_mode="plan")
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
    a_session(tmp_path)

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
