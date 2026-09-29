# core/
> L2 | 父级: /CLAUDE.md

领域语言层。只依赖标准库，是整座依赖图的最底层：kernel 用它裁定事实，cognition 用它表达信念，persistence 用它存储，learning 用它构造标签。所有值对象不可变、可哈希、可 pickle，查询一律返回排序元组——回放确定性在这一层就被保证。

成员清单
schema.py: 领域词汇表，Kind/Rel/Op/Manner 枚举（含动手/研读/施用）+ RelSpec（函数型关系决定信念互斥槽位）+ OpSignature（行动语法，MOVE = 目的地 + 路线门）+ OBSERVABLE_ATTRS（静态外观白名单）/ STATUS_ATTRS（伤、毒、被制，环顾可见）/ PRIVATE_ATTRS（内力、点穴时限、修习进度，永不外泄）
entities.py: Entity/Relation 值对象，属性存为有序元组以保证可哈希与 repr 稳定
changes.py: 世界变化语言 AddRelation/RemoveRelation/SetAttr，带前置条件（删除要求存在、改值要求旧值匹配），relocate() 表达 AT 的一删一增
world.py: WorldState 不可变实际世界图，apply() 只改事实、stamp() 才推进版本，fingerprint() 是回放验收判据
propositions.py: Proposition（命题内容，不含"谁相信"）与 Fact（带极性），slot 定义函数型谓词下的互斥槽位
events.py: 因果链数据 Intent → Event（真相）→ Observation（服务端溯源）→ Percept（角色可见的片面内容，刻意不带来源 ID）；MOVE 的 obj 是所走的路线（门）；EntitySketch.seen 区分亲眼所见（有外观）与只闻其名（外观未知）；言语原话 utterance 随意图与感知传递，只是修辞
grammar.py: signature_error() 行动语法检查，kernel 用真实种类、cognition 用已知种类调用同一把尺子；只查"能不能这样说"，不查"能不能做成"
profiles.py: Goal/Profile 角色设定卡，目标角色条件化（守护/获取/递送/守地/寻仇/灭口/护人）+ 时间闸门 not_before + 寻仇了结条件 until + 盟友 + 信任度；interests() 汇总关注的人与物
memories.py: MemoryRecord 经历权威记录，known_at 用于检索时的时间过滤，向量索引只是它的派生
ids.py: blake2b 确定性派生 digest/derive_seed/make_id，拒绝 hash() 与 uuid4 以保证可回放
clock.py: 游戏时间为整数分钟，1 tick = 1 分钟，clock_label() 渲染"第1日 08:10"，is_night()/minutes_until_night() 定义昼夜（戌时入夜、卯时破晓）
__init__.py: 再导出全部公共类型

[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
