# agents/
> L2 | 父级: /CLAUDE.md

角色的决策流程与多智能体编排（LangGraph）。只产出结构化意图，从不写世界；提交与冲突结算归 runtime/WorldAuthority。隔离边界是 AgentPort：节点只能读"自己的认知、自己的回忆"。依赖（port/policy/predictor/speaker）经 LangGraph runtime context 注入，不进检查点；检查点只存决策轨迹，恢复它不会撤销已提交的世界事件，所以意图 ID 由 (世界, 分支, 角色, 版本) 派生，重试即去重。

数据流：observe（近期经历凝成线索）→ recall（工作记忆 ∪ 长期联想）→ propose（候选集）→ predict（后果）→ decide（策略在候选集中选）→ express（言语润色）→ submit（意图）。

成员清单
port.py: AgentPort 一个角色能触碰的全部外部能力（认知读取器、回忆器、决策依据版本与时间）
predictors.py: Prediction + OutcomePredictor 协议 + HeuristicPredictor 信念先验（门锁信念定通过率、位置可信度定拿取率、下落不明时查看更有价值）；GNN 预测器实现同一协议即可替换
policies.py: Situation/Choice/Policy 协议 + ScriptedPolicy 角色条件化规则策略（回应提问 → 按目标守护/获取/递送 → 查探守护范围响动 → 等待）；不知下落≠丢失，失主讨要、旁人报告、说过不重复；也是 RL 模仿学习的示范者
npc_graph.py: 单角色 LangGraph 决策图 NpcState/NpcContext/build_npc_graph，checkpoint_serde() 以白名单限制检查点可反序列化的类型
orchestrator.py: Orchestrator 以 Send 扇出并行运行多个角色的决策图并汇总 Deliberation（意图 + 理由 + 回忆 + 候选数）
scheduler.py: Scheduler 节流阀，有新经历/手头有事/闲置过久才完整决策，其余例行等待
__init__.py: 包入口（langgraph 为可选依赖）

[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
