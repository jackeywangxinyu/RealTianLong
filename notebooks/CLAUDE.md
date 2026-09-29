# notebooks/
> L2 | 父级: /CLAUDE.md

在别人的算力上跑本仓库的训练入口。笔记本只是“检出固定提交 → 安装 → 全量测试 → 调用同一套 CLI → 产物写进 Drive”的外壳，不含任何训练逻辑——逻辑只在 src/tianlong/learning 里一处。笔记本本身由 scripts/make_colab_notebook.py 生成，不手改。

成员清单
train_colab.ipynb: Colab GPU 训练与评测（生成物，COMMIT 为占位，交给 Colab 的版本由生成器填入 40 位提交）。检出：代码来自 Drive 上的完整 git bundle 或 GitHub，已有目录不是干净仓库即停、从不嵌套克隆、checkout --detach 后核对 HEAD；私有仓库的令牌只经环境变量注入一次性请求头，remote、.git/config、日志里都没有（最后一格扫描运行目录确认）。sh() 非零即抛异常，安装（constraints.txt 锁定版本）或全量测试失败后续不再执行。速度由 CPU 核数决定（世界模拟、认知、GNN 预测、评测与示范都在 CPU 上按世界/按局并行），各阶段按本机核数自动分配 --workers 与 PPO 采样进程，GPU 只加速学习器与 GNN 训练。阶段：剖析 → GNN 两视角（内存 ≥ 24 GB 时经 sh_all 同时训练，否则逐个；按轮续训、按校准损失选轮）→ RL 主实验三种子 + 训练期无预测 / 无记忆 + E04 探查有/无预测（PPO 断点续训）→ results（含跨运行配对）→ 部署包；全部写在 MyDrive/RealTianLong/runs/<提交前十位>-<规模>/

[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
