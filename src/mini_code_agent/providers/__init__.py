"""LLM 后端适配层：agent 只依赖 base.Provider 这个 Protocol。

`complete` 给内部回合（后台续跑、子 agent、压缩摘要），`stream` 给前台终端回合。
具体后端各自成模块。
"""
