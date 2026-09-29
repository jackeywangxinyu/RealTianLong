# learning/rl/
> L2 | 父级: /src/tianlong/learning/CLAUDE.md

角色策略的强化学习（RLlib 新 API 栈）。强化学习发生在训练阶段；游玩时运行训练好的网络（LearnedPolicy），不在玩家输入后临时重训。
训练环境与线上共用同一个内核与同一套认知修正，观测只来自各自的认知图；奖励来自真实目标进展（势函数差分），并惩罚失败与"冤枉人"。

动作空间是"指针式"候选打分：候选对象由认知图提供（cognition.candidates），策略只学在候选集中选哪一个；掩码只屏蔽候选集之外的空位，从不按真相屏蔽行动。
冻结的世界模型（启发式或 GNN）为每个候选提供 (成功率, 预期获知) 特征——"在每批策略采样期间固定其版本"。

成员清单
observation.py: ObsSpec 与定长观测契约（节点/边/焦点/目标/候选/候选预测/掩码），RL 环境与游戏内 LearnedPolicy 共用
env.py: TianlongEnv（MultiAgentEnv），同一 tick 同时出招、统一结算；expert_actions() 给出脚本示范；last_events 只供评测数行为；obs_spec 避开 gymnasium 的 spec 属性；训练世界不江湖化（奖励只懂物品目标）
rewards.py: potential() 目标进度势函数，step_reward() = 势差 − 步长成本 − 失败成本 − 冤枉人惩罚
module.py: GraphPolicyNet（纯 torch：定长观测还原为稀疏批图 → RelationalEncoder → 逐候选打分 + 价值头）与 CandidateScoringModule（TorchRLModule + ValueFunctionAPI 外壳）
train.py: 模仿学习初始化 → PPO（所有角色共享参数、各自观测）→ 留出种子上对照 随机/脚本/模仿/PPO + 去掉世界模型特征的消融；--entropy 与 --bc-smoothing 控制“模仿后策略过尖、PPO 无从探索”的问题；评测数冤枉人与动手（奖励不禁动手，“制住再搜”是否被学会要如实报告）；--env-runners/--gpus 供 Colab 放大；CLI python -m tianlong.learning.rl.train
policy.py: LearnedPolicy 以 Policy 协议接入 LangGraph 决策图，替换 ScriptedPolicy 而不改图
__init__.py: 包入口（ray[rllib] 为可选依赖）

[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
