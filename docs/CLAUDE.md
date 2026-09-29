# docs/
> L2 | 父级: /CLAUDE.md

给人看的产出物，不被代码引用。这里的文件由脚本生成、提交入库，是“效果”的存档；改了引擎行为就重跑脚本，让存档与地形一致。

成员清单
demo/wuliang.md: scripts/play_demo.py 录制的天龙八部·无量山一局（Gemini 叙述）——开场冲突链、原著路线学成凌波微步、玩家离场后入夜私奔在幕后发生，结尾是段誉以为与真相的对照表
results/<运行名>/: 一次 Colab 训练运行的机器可读存档，按运行名（提交号前 10 位-规模）分目录、目录结构与 Drive 上的运行目录同构；gnn/dynamics_{agent,env}.json 与 rl/<实验>/policy_ppo*.json 是 learning.train / learning.rl.train 写出的报告（带 manifest 与逐世界记录，--pair 配对比较的原料），RESULTS.md 由 learning.results 从这些报告生成（README 的“当前结果”只引用这个目录：表格照抄 RESULTS.md，其余指标取自报告），bundle.json 记下上线用的两份检查点的 sha256 与训练时语义版本（检查点本身不入库，全部留在 Drive），env.json / profile.json 是运行环境与分段计时。只增不改：新的运行开新目录，旧目录是历史
results/f1d30c93ef-full/: 第一次全规模运行（Schema v2 / reward-v2，A100 + 12 核）——两个视角的动态模型、三个训练种子的主实验、训练期无预测/无记忆消融、“先探查”任务及其无预测消融

[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
