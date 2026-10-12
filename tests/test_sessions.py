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
