# runtime/
> L2 | 父级: /CLAUDE.md

装配层。把内核、认知、存储、记忆、智能体与语言串成可玩的会话。角色决策可以并行，事实提交只经 WorldAuthority 串行发生。

成员清单
authority.py: WorldAuthority 每个世界实例唯一的权威写入器，settle() = 幂等检查 → 内核裁定 → 认知折叠 → 经历提炼 → 原子提交；found() 由场景建世界
__init__.py: 包入口

[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
