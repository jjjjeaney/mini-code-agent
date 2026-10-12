"""长期记忆：跨会话保留的事实，比如项目约定、用户偏好。

和 sessions 的分界是生命周期：会话结束就归档的进 sessions，一直留着的进这里。

存储、召回、提取、整理、分类这些设计上的取舍见 `docs/memory-design.md`。
"""

from mini_code_agent.memory.store import Memory, index_text, names, read, write

__all__ = ["Memory", "index_text", "names", "read", "write"]
