# persistence/
> L2 | 父级: /CLAUDE.md

事实持久化。WorldStore 协议的唯一写路径是 commit()：乐观版本检查 + 请求绑定检查 + 世界新版本 + 事件 + 观察 + 变化了的认知 + 待索引经历（outbox）+ 请求进度（TurnEnvelope）+ 会话运行态，一次原子提交——“世界推进了”与“这个请求走到哪了”不会只落一半，同一请求的重复投递也不会各结算一次。叙述文字另走幂等的 record_render()（只补写一次），文字失败不回滚世界。LangGraph 检查点不能代替它；向量索引从它派生、可重建。

成员清单
store.py: WorldRef（world_id + branch_id）、TurnEnvelope（request_id + payload_hash 绑定的请求进度：玩家行动模板、多步计划的后续步骤 followups、是否追加反应 tick、计划 tick、已提交版本与时钟、玩家感知、初见描写键、完结标记、叙述）、CommitBatch（可选 request / session_state 随同一事务写入）、VersionConflict、RequestConflict（同 ID 异内容，或进度接不上已落库的那一份）、UnknownWorld、WorldStore 协议（新增 request()/session_state()/save_versions() 读与 record_render() 旁路写，create() 记下存档版本）、check_request_progress()（两个后端共用的请求绑定检查：首 tick 建立绑定，此后每次提交的进度必须恰好是已落库进度加本次版本，已完结或异内容即拒——放在提交的临界区 / 事务里，查询与结算之间没有空隙）
memory_store.py: InMemoryWorldStore，测试/训练/离线默认后端，锁只保护"检查版本 + 检查请求绑定 + 写入"临界区；请求进度与会话运行态同一临界区写入，会话运行态按 JSON 往返存取（与 Neo4j 取回的形状一致）
neo4j_store.py: Neo4jWorldStore 图数据库后端；实体为带种类标签的节点、世界关系为类型化边；命题节点与 BELIEVES 边分离"内容"与"谁相信"；commit 先锁 World 节点再比版本（read-committed 下由锁保证串行），持锁读已落库的请求进度做绑定检查，不合即整个事务回滚；关系按 net_relation_diff() 净差异落库；环顾类观察不落库；认知节点另存勘察记录（surveyed/searched）、承诺状态（obligations/said）与社交状态（cues/attitudes/company/allies/yielded），经历记录带 informant/verdict，旧存档缺省为空；(:Request {data, narration}) 存请求进度，World 节点存存档版本（versions）与会话运行态（session），三者与世界变化同一事务；标签与关系类型只取自枚举白名单
codec.py: core/cognition 值对象 ⇄ JSON 的逐字段显式编解码（草图带 seen，旧存档缺省为亲见；意图与感知带言语行为 social，旧记录缺省为无；承诺（待回话无命题、带对方的言语行为）与说过的话（闲话无命题、带言语行为）；社交线索 cue 与态度 attitudes（旧存档缺省为空）；envelope_to/from 请求进度含后续步骤与反应 tick，叙述不入其中），不用反射与 pickle——数据库内容不能决定构造哪个类
__init__.py: 包入口（Neo4j 实现按需导入，不强制依赖驱动）

[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
