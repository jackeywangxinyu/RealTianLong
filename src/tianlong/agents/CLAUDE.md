# agents/
> L2 | 父级: /CLAUDE.md

角色的决策流程与多智能体编排（LangGraph）。只产出结构化意图，从不写世界；提交与冲突结算归 runtime/WorldAuthority。隔离边界是 AgentPort：节点只能读"自己的认知、自己的回忆"。依赖（port/policy/predictor/speaker）经 LangGraph runtime context 注入，不进检查点；检查点只存决策轨迹，恢复它不会撤销已提交的世界事件，所以意图 ID 由 (世界, 分支, 角色, 版本) 派生，重试即去重。

数据流：observe（近期经历凝成线索）→ recall（工作记忆 ∪ 长期联想；只为读 Situation.memories 的策略而跑，现有策略都不读，默认跳过）→ propose（候选集）→ predict（后果）→ decide（策略在候选集中选）→ express（言语润色）→ submit（意图）。

成员清单
port.py: AgentPort 一个角色能触碰的全部外部能力（认知读取器、回忆器、长期记忆摘要、决策依据版本与时间）
predictors.py: Prediction v2（成功率、有效新观察、目标进展、风险、不确定性）+ PRED_FIELDS（进入策略观测的列，登记即生效）+ OutcomePredictor 协议（带 profile）+ imagine()/branch_value()：在自己的认知上假想一条“得手之后”的短分支、用同一套目标语义求势能差（不写回、不碰真相）+ HeuristicPredictor 信念先验（锁信念定通过率、位置可信度定拿取率、没看过/没翻过的地方才有新观察、自己走过不算“那边有动静”、动手胜负难料）；GNN 预测器实现同一协议即可替换
policy_kit.py: Situation/Choice（带结构化标签：等待原因 WAIT_REASONS 或 explore；social 给所选候选附上言语行为；free 是候选之外唯一允许的行动——不带命题的闲话，此时 index 指向 WAIT，只认下标的学习层把它当等待；chosen() 给出最终交给内核的行动）/Policy 协议 + PolicyKit 共享积木（在候选集中挑选、沿自己的地图连门带路走一步——有绕得开自知锁门的路就绕，没有才面对那扇门：先试认为配的、再试看着像钥匙的，打不开就不去撞，记忆旧了才再推；凭个人勘察记录探索：先翻此处、再去最近的没看过的地方、都看过就向眼前人打听；近期经历、谁对谁动过手、_whereabouts() 矛盾说法并存时按“可信度 × 说话者可靠度（长期记忆）”取舍、信念查询）
tactics.py: MartialTactics 江湖行为积木：服解药自救、为自己与盟友还手、盟友中毒则搜出下毒者身上的解药施救、寻仇（受伤即解气或须制住；不知仇人在哪就去找）、守地、灭口（只灭落单的撞见者，满堂同门前悄悄溜走）、护人（不知被护者在哪就去找）
policies.py: ScriptedPolicy 作曲者：自救 → 还手 → 救治 → 回应提问 → 按目标（受时间闸门约束）→ 查探响动 → 等待；不知下落≠丢失（守护者从没见过守护之物就去原处仔细看一眼；确知不在、或亲手翻过原处仍不见，才盘问搜身），失主讨要、旁人报告、说过不重复（持久的 said 记录）；欠着的问题（obligations）问话人在眼前就作答；要找的东西或人下落不明就去探索；等待带结构化原因（没事/未到时辰/自以为已达成/不知道而卡住/无可行候选/想不出办法）；也是 RL 模仿学习的示范者
npc_graph.py: 单角色 LangGraph 决策图 NpcState/NpcContext（recall 开关，默认关：观察与向量回忆节点空转，省下每人每 tick 一次检索）/build_npc_graph（预测带上角色设定、决策带上长期记忆摘要；决策经 Choice.chosen() 落定，闲话不在此处措辞——留给主持人之声），checkpoint_serde() 以白名单限制检查点可反序列化的类型
orchestrator.py: Orchestrator 以 Send 扇出并行运行多个角色的决策图并汇总 Deliberation（意图 + 理由 + 回忆 + 候选数）；决策轨迹检查点须显式开启（checkpoint=True：剖析显示它占决策耗时的 65~75%，会话从不读它），开启后每角色每 tick 一条线程，按条数修剪，长局内存有界
scheduler.py: Scheduler 节流阀，有新经历/手头有事/约定时辰已到/闲置过久才完整决策，其余例行等待；to_state()/from_state() 以 JSON 兼容形状存取调度标记，随世界提交落库、读档恢复（否则读档那一刻人人“该决策”，调度与连续运行分叉）
__init__.py: 包入口（langgraph 为可选依赖）

[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
