# runtime/
> L2 | 父级: /CLAUDE.md

装配层。把内核、认知、存储、记忆、智能体与语言串成可玩的会话。角色决策可以并行，事实提交只经 WorldAuthority 串行发生。

成员清单
authority.py: WorldAuthority 每个世界实例唯一的权威写入器，settle() = 幂等检查 → 内核裁定 → 认知折叠 → 经历提炼 → 原子提交；found() 由场景建世界（信任度取自角色设定）
session.py: GameSession 装配中心（存储里已有该世界则读档接续，并从经历记录重建向量索引），一回合 = 解析玩家输入（失败不推进时间）→ 调度器分出需决策者 → 同一版本扇出 NPC 决策、其余例行等待 → 权威结算 → outbox 同步索引 → 玩家视角叙述；“等到天黑”跨越多个 tick、身边一有动静即停；记录已描写过的实体只在初见时描写；TurnReport 携带真相、NPC 理由与分阶段耗时供调试
cli.py: 终端前端 main()，--world 选世界（默认天龙八部·无量山），/beliefs 看玩家自己的认知、/debug 看真相与 NPC 理由，两种视角刻意分开；--llm auto 有密钥即启用 Gemini；--store neo4j --save 名称 实现跨进程存档；load_dotenv() 读取 .env 只补缺不覆盖
__init__.py: 包入口

[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
