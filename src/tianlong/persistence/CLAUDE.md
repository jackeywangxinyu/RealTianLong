# persistence/
> L2 | 父级: /CLAUDE.md

事实持久化。WorldStore 协议的唯一写路径是 commit()：乐观版本检查 + 世界新版本 + 事件 + 观察 + 变化了的认知 + 待索引经历（outbox），一次原子提交。LangGraph 检查点不能代替它；向量索引从它派生、可重建。

成员清单
store.py: WorldRef（world_id + branch_id）、CommitBatch、VersionConflict、UnknownWorld、WorldStore 协议
memory_store.py: InMemoryWorldStore，测试/训练/离线默认后端，锁只保护"检查版本 + 写入"临界区
__init__.py: 包入口（Neo4j 实现按需导入，不强制依赖驱动）

[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
