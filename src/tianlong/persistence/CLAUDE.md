# persistence/
> L2 | 父级: /CLAUDE.md

事实持久化。WorldStore 协议的唯一写路径是 commit()：乐观版本检查 + 世界新版本 + 事件 + 观察 + 变化了的认知 + 待索引经历（outbox），一次原子提交。LangGraph 检查点不能代替它；向量索引从它派生、可重建。

成员清单
store.py: WorldRef（world_id + branch_id）、CommitBatch、VersionConflict、UnknownWorld、WorldStore 协议
memory_store.py: InMemoryWorldStore，测试/训练/离线默认后端，锁只保护"检查版本 + 写入"临界区
neo4j_store.py: Neo4jWorldStore 图数据库后端；实体为带种类标签的节点、世界关系为类型化边；命题节点与 BELIEVES 边分离"内容"与"谁相信"；commit 先锁 World 节点再比版本（read-committed 下由锁保证串行）；关系按 net_relation_diff() 净差异落库；环顾类观察不落库；认知节点另存勘察记录（surveyed/searched，旧存档缺省为空）；标签与关系类型只取自枚举白名单
codec.py: core/cognition 值对象 ⇄ JSON 的逐字段显式编解码（草图带 seen，旧存档缺省为亲见），不用反射与 pickle——数据库内容不能决定构造哪个类
__init__.py: 包入口（Neo4j 实现按需导入，不强制依赖驱动）

[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
