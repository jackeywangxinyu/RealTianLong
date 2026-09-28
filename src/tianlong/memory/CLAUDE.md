# memory/
> L2 | 父级: /CLAUDE.md

角色可回忆的经历。权威记录（MemoryRecord）由权威写入器在世界提交的同一事务里写入 outbox；向量索引只是可重建的派生数据，绝不裁定物品位置等当前事实。"以前在哪里"不能覆盖"现在在哪里"。

成员清单
records.py: records_for() 记忆写入策略，只记事件（看到/听到/做过/被告知）与意外（原以为在的东西不见了及其新下落），不记"一切如常"
__init__.py: 包入口（qdrant-client 为可选依赖，子模块按需导入）

[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
