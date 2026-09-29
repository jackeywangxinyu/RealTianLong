# runtime/
> L2 | 父级: /CLAUDE.md

装配层。把内核、认知、存储、记忆、智能体与语言串成可玩的会话。角色决策可以并行，事实提交只经 WorldAuthority 串行发生。重试与读档不让世界多走一步：意图幂等（意图 ID）之上还有请求幂等（TurnEnvelope），会话运行态随提交落库、读档原样恢复，存档版本不符即拒绝。

成员清单
authority.py: WorldAuthority 每个世界实例唯一的权威写入器，settle() = 幂等检查 → 内核裁定 → 认知折叠 → 经历提炼 → annotate 附注（调用方看到未落库的结算后给出请求进度与会话运行态）→ 原子提交；found() 由场景建世界（信任度取自角色设定）并记下存档版本
session.py: GameSession 装配中心（存储里已有该世界则先过存档版本闸门再读档接续，恢复调度标记与已描写实体，并从经历记录重建向量索引），一回合 = 解析玩家输入（失败不推进时间）→ 调度器分出需决策者 → 同一版本扇出 NPC 决策、其余例行等待 → 权威结算 → outbox 同步索引 → 玩家视角叙述；“等到天黑”跨越多个 tick、身边一有动静即停，经过的时辰连同玩家原话一并交给叙述者；记录已描写过的实体只在初见时描写；每个 tick 的请求进度与会话运行态（调度标记 + 已描写实体）先在副本上推进、随世界同一事务落库，提交成功才替换内存；turn(text, request_id) 同 ID 同内容返回既有结果、异内容抛 RequestConflict，提交后崩溃的重试只按已持久化的感知重写文字，多 tick 等待中途崩溃只走剩下的 tick，request_id=None 行为不变；请求绑定由存储在提交内检查，并发的重复投递每个 tick 只有一次能提交，被越过的一方即停（不多走 tick、会话运行态改回落库的那一份），以落库的请求为准返回，对方未走完时只给目前的文字、不落库；进度与世界版本两次读一致才判定“别人越过了它”；建档后尚无提交就读档，开场已描写的实体按开场规则补回；LLMSpeaker 拿到场景名字全集与别称作对白闸门的拒绝全集；TurnReport 携带真相、NPC 理由、分阶段耗时，另记文字侧结果 render（来源 + 闸门结论）与 replayed
versions.py: 存档版本闸门，current_versions() = 存档格式 + KERNEL_VERSION + 属性规格摘要（种类集合排序后摘要，跨进程稳定）+ GOALS_VERSION；check_save() 不一致即抛 IncompatibleSave，allow_migration=True 才接续且不改写、不补写旧档
cli.py: 终端前端 main()，--world 选世界（默认天龙八部·无量山），/beliefs 看玩家自己的认知、/debug 看真相与 NPC 理由，两种视角刻意分开；--llm auto 有密钥即启用 Gemini；--store neo4j --save 名称 实现跨进程存档（版本不符一句话说明并退出，--allow-migration 显式接续）；--predictor gnn / --policy learned 缺模型或模型词表过期时给出一句话提示并退出；load_dotenv() 读取 .env 只补缺不覆盖
__init__.py: 包入口

[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
