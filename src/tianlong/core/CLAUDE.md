# core/
> L2 | 父级: /CLAUDE.md

领域语言层。只依赖标准库，是整座依赖图的最底层：kernel 用它裁定事实，cognition 用它表达信念，persistence 用它存储，learning 用它构造标签。所有值对象不可变、可哈希、可 pickle，查询一律返回排序元组——回放确定性在这一层就被保证。

成员清单
schema.py: 领域词汇表，Kind/Rel/Op/Manner 枚举 + RelSpec（函数型关系决定信念互斥槽位）+ OpSignature（行动语法）+ OBSERVABLE_ATTRS（肉眼可见属性白名单）
entities.py: Entity/Relation 值对象，属性存为有序元组以保证可哈希与 repr 稳定
changes.py: 世界变化语言 AddRelation/RemoveRelation/SetAttr，带前置条件（删除要求存在、改值要求旧值匹配），relocate() 表达 AT 的一删一增
world.py: WorldState 不可变实际世界图，apply() 只改事实、stamp() 才推进版本，fingerprint() 是回放验收判据
propositions.py: Proposition（命题内容，不含"谁相信"）与 Fact（带极性），slot 定义函数型谓词下的互斥槽位
events.py: 因果链数据 Intent → Event（真相）→ Observation（服务端溯源）→ Percept（角色可见的片面内容，刻意不带来源 ID）；言语原话 utterance 随意图与感知传递，只是修辞
grammar.py: signature_error() 行动语法检查，kernel 用真实种类、cognition 用已知种类调用同一把尺子；只查"能不能这样说"，不查"能不能做成"
profiles.py: Goal/Profile 角色设定卡，目标角色条件化（守护/获取/递送）+ 对他人的信任度，供脚本策略、RL 奖励、信念修正与对白渲染共用
memories.py: MemoryRecord 经历权威记录，known_at 用于检索时的时间过滤，向量索引只是它的派生
ids.py: blake2b 确定性派生 digest/derive_seed/make_id，拒绝 hash() 与 uuid4 以保证可回放
clock.py: 游戏时间为整数分钟，1 tick = 1 分钟，clock_label() 渲染"第1日 08:10"
__init__.py: 再导出全部公共类型

[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
