# learning/
> L2 | 父级: /CLAUDE.md

GNN 学习"接下来可能发生什么"（阶段 B），RL 学习"应该选择什么"（阶段 C，见 rl/CLAUDE.md）。两者的决策输入都只来自角色认知：featurize 只接受 GraphView，角色入口在类型上拿不到 WorldState。

设计决策：编码器用 TransformerConv(edge_dim) 而非 RGCN/HGT——关系类型（含极性与方向）与可信度/时效/传闻都在边特征里，RGCN 类卷积会丢掉这些认知语义；节点种类作为特征，小图上与异构图表达力等价、且能与 RL 的定长观测互通。环境动态与角色视角两类样本分开构造、分开训练：前者学真相如何演化，后者学"做了这件事之后我会相信什么"（含"仍然不知道"）。标签只来自内核实际执行的结果。

成员清单
featurize.py: 特征词表与 GraphView → GraphTensors（numpy 中立格式）；每条边配反向边；encode_action() 把候选行动编码为操作/方式/目标/对象/行动者在图中的位置；VOCAB 词表指纹写进每个检查点，check_vocab() 让过期模型在加载时以 StaleModel 明确拒绝
samples.py: Sample 监督样本；env_sample() 真实状态→真实下一状态，agent_sample() 个人认知→结算后的认知，agent_query() 推理输入；DynData 让指针字段随批偏移、"未知"用独立掩码表示
datagen.py: 数据工厂，程序化世界（默认一半江湖化）里"脚本策略 + 分层随机探索"行动、内核执行，每 tick 只一人行动使环境标签干净；记录样本所属世界供按世界切分
model.py: RelationalEncoder（残差 TransformerConv 栈）与 DynamicsModel（行动条件化 + 成败头 + 带"未知"类与可学习惯性项的位置指针头 + 属性三态头）
train.py: 训练与验收 CLI（python -m tianlong.learning.train --view env|agent），按世界切分，指标对照"什么都不变"基线：变化召回、未变保持、未知被错误确定化率；成败按操作分项 + Brier 校准（动手胜负带随机），属性变化按属性分项；device=auto 有 GPU 即用
predictor.py: GNNPredictor 以 OutcomePredictor 协议接入 LangGraph 决策图，输出成功概率与预期获知量
rl/: 强化学习（见 rl/CLAUDE.md）
__init__.py: 包入口（torch / torch_geometric / ray 为可选依赖）

[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
