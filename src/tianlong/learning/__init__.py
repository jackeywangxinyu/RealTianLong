"""
[INPUT]: 无（子模块按需导入：torch / torch_geometric / ray 均为可选依赖）
[OUTPUT]: 子模块 featurize / samples / datagen / model / train / predictor / rl
[POS]: learning 包入口；GNN 学习“接下来可能发生什么”，RL 学习“应该选择什么”——两者都只接受角色认知作为决策输入
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""
