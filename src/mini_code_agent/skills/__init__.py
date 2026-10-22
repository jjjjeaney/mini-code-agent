"""技能：一份一份"照着做"的步骤，跨会话留着。

和长期记忆的分界是**内容**——记忆是事实（拿来判断），skill 是步骤（拿来照做）。存储形状也
不一样：记忆是平铺的文件，skill 是目录，能带附件。

设计、边界和取舍见 `docs/skill-design.md`。
"""

from mini_code_agent.skills.store import (
    INDEX_NAME,
    SKILL_DIR,
    SKILL_FILE,
    Skill,
    attachments,
    index_text,
    names,
    read,
    skills_dir,
    sync_skill_index,
)

__all__ = [
    "INDEX_NAME",
    "SKILL_DIR",
    "SKILL_FILE",
    "Skill",
    "attachments",
    "index_text",
    "names",
    "read",
    "skills_dir",
    "sync_skill_index",
]
