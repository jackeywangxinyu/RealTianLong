# scripts/
> L2 | 父级: /CLAUDE.md

本地开发基础设施与演示工具。不含引擎逻辑：只准备环境，或把引擎跑一遍给人看。

成员清单
neo4j_local.sh: 下载并启动 Neo4j Community（仅监听 127.0.0.1，限定堆与页缓存），要求预设 NEO4J_PASSWORD；供契约测试与 `--store neo4j` 持久化游玩
make_colab_notebook.py: notebooks/train_colab.ipynb 的唯一源头（单元格以 Python 字符串维护，可审阅可测试）；--commit 填入固定提交生成交给 Colab 的那一份，仓库里的版本不填（tests/test_notebook.py 核对一致）
play_demo.py: 按固定指令序列录制一局（默认无量山原著路线），写出 docs/demo/{world}.md：每回合玩家所见的叙述 / 内核裁定的全部事件 / NPC 各自理由，结尾对照玩家以为与真相；叙述经 CachedLLM 缓存，重跑不再花费调用

[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
