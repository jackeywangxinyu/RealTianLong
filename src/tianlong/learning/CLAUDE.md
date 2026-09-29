# learning/
> L2 | 父级: /CLAUDE.md

GNN 学习"接下来可能发生什么"（阶段 B），RL 学习"应该选择什么"（阶段 C，见 rl/CLAUDE.md）。两者的决策输入都只来自角色认知：featurize 只接受 GraphView，角色入口在类型上拿不到 WorldState。

设计决策：编码器用 TransformerConv(edge_dim) 而非 RGCN/HGT——关系类型（含极性与方向）与可信度/时效/传闻都在边特征里，RGCN 类卷积会丢掉这些认知语义；节点种类作为特征，小图上与异构图表达力等价、且能与 RL 的定长观测互通。环境动态与角色视角两类样本分开构造、分开训练：前者学真相如何演化（输入含规则用到的全部机制变量——表示丢信息与真实随机性必须分开），后者学"做了这件事之后我会相信什么"（含"仍然不知道"与"确知不在原处"）。标签只来自内核实际执行的结果。维度与语义只在 schema.py 定义一次。

成员清单
schema.py: 类型化特征规格（Schema v2）唯一一处：节点列布局（种类 | 自身 | 每个属性一块值列 + known 三态列（1 已知 / 0 未知 / -1 不适用），数值以 value/scale、类别以独热进入 | 事件的操作/渠道/结果/原因/命题谓词/极性/提问）、关系词表（认知关系含极性与方向、ONEWAY_TO、事件关系含命题主语宾语）、行动编码字段（含言语命题）、预测目标声明 TARGETS（位置指针 + UNKNOWN/GONE、动态布尔三态、动态数值、发现、有效新观察数）；SCHEMA 指纹写进每个检查点，check_schema() 对规格不符或视角不符（全知模型冒充角色预测）以 StaleModel 明确拒绝
featurize.py: 按 schema 把 GraphView 编码为 GraphTensors（numpy 中立格式），每条边配反向边；bool_tri()/num_known() 从节点列读回动态属性（标签与输入同一定义）；encode_action() 把候选编码为 操作/方式/目标/对象（MOVE 为路线门）/行动者 + 言语命题（谓词、主语、宾语、极性、是否提问）
samples.py: 监督信号（StateDelta）：env_sample() 真实状态（含全部机制变量）→ 真实下一状态，agent_sample() 个人认知 → 结算后的认知，agent_query() 推理输入；“不知道”拆成仍不知道（UNKNOWN）/确知不在原处去向不明（GONE）/认识新实体（discover）；动态属性按类型给标签（布尔三态、数值 + 是否已知：修习进度、学成技能、被吸走的内力）；有效新观察数由 datagen 给出；DynData 让所有指针字段随批偏移、空类用独立掩码
task.py: TaskConfig 任务分布契约（江湖化比例、规模、人数、修习覆盖旋钮、启用目标族、时限）——GNN 数据、模仿示范、PPO 的每个 env runner、评测与检查点读同一份；registry() 只注册启用目标族，fingerprint() 进 manifest
datagen.py: 数据工厂，按 TaskConfig 取样的程序化世界里“脚本策略 + 分层随机探索”行动、内核执行，每 tick 只一人行动（孤立行动效果模型）；own_effect_slots()/observation_gain() 把“信念变化中扣除本行动直接效果”定义为有效新观察数；记录样本所属世界供按世界切分
model.py: RelationalEncoder（残差 TransformerConv 栈）、ActionEncoder（操作/方式/命题谓词嵌入 + 五个引用的节点表示 + 命题极性，策略网络同构）与 DynamicsModel（行动条件化 + 成败头 + 带 UNKNOWN/GONE 两空类与惯性项的位置指针头 + 动态布尔三态头 + 数值残差头与已知性惯性 + 发现头 + 有效新观察数头）；loss_terms() 每个目标一项，角色专有目标只在角色样本上计
train.py: 训练与验收 CLI（python -m tianlong.learning.train --view env|agent，任务字段展平为参数），按世界切分；指标按 TARGETS 逐项：位置召回只叫 holder_*（不冒充“全部事实”）、布尔属性逐属性、数值属性 MAE 对照“不变”、成败按操作的 Brier 对照**训练集**常数（测试集常数只作诊断）、发现与有效新观察数；coverage 报告每类机制在数据里出现几次；检查点带 schema 指纹与视角；device=auto 有 GPU 即用
predictor.py: GNNPredictor 以 OutcomePredictor 协议接入 LangGraph 决策图；加载时核对规格与视角（只收 agent 模型）；成功率来自成败头，预期获知来自有效新观察数头（确定地走到已知处不算获知）
rl/: 强化学习（见 rl/CLAUDE.md）
__init__.py: 包入口（torch / torch_geometric / ray 为可选依赖）

[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
