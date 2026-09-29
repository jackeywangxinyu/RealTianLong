# .github/
> L2 | 父级: /CLAUDE.md

持续集成。只跑仓库里已有的检查（ruff + pytest），不含任何只在 CI 里才存在的逻辑——本地能跑通的，CI 就应该跑通。

成员清单
workflows/ci.yml: 两个作业。core：只装 dev，验证“核心零依赖”（内核、认知、持久化契约的内存后端必须独立通过，学习层与 Neo4j 用例整模块跳过）；full：CPU 版 torch + `constraints.txt` 锁定的学习层依赖 + Neo4j 5 服务容器（一次性密码只存在于该次运行），跑全量套件——含笔记本与生成器一致性、C02–C05、断点续训逐位一致与双后端契约

[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
