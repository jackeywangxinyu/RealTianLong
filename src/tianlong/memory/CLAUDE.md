# memory/
> L2 | 父级: /CLAUDE.md

角色可回忆的经历。权威记录（MemoryRecord）由权威写入器在世界提交的同一事务里写入 outbox；向量索引只是可重建的派生数据，绝不裁定物品位置等当前事实。"以前在哪里"不能覆盖"现在在哪里"。

成员清单
records.py: records_for() 记忆写入策略，只记事件（看到/听到/做过/被告知）与意外（原以为在的东西不见了及其新下落），不记"一切如常"
embedder.py: Embedder 协议 + HashingEmbedder 字符 1~3 gram 带符号特征哈希，确定性零依赖；语义模型可按协议替换
index.py: MemoryScope 服务端强制检索边界（世界/分支/主人/known_at<=now/类别）+ QdrantMemoryIndex，记录 ID 经 uuid5 映射为点 ID 使重放幂等
recall.py: Recall 读路径，工作记忆（权威存储近期记录，不论是否已索引）∪ 长期联想（向量检索）并去重
indexer.py: MemoryIndexer outbox → 索引同步，先写索引后确认，崩溃重放只覆盖不重复
__init__.py: 包入口（qdrant-client 为可选依赖，子模块按需导入）

[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
