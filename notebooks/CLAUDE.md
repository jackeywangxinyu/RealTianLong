# notebooks/
> L2 | 父级: /CLAUDE.md

在别人的算力上跑本仓库的训练入口。笔记本只是“克隆 → 安装 → 调用同一套 CLI（参数放大）→ 取回产物”的外壳，不含任何训练逻辑——逻辑只在 src/tianlong/learning 里一处。

成员清单
train_colab.ipynb: Colab GPU 放大训练，GH_TOKEN 从 Colab Secrets 读取且不回显；GNN 两视角（3000 世界 × 40 轮）→ RL 模仿 + PPO（GPU 学习器、以 GNN 预测为候选特征）→ 打印报告 → 打包下载，本地以 --artifacts 指向即可接入游戏

[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
