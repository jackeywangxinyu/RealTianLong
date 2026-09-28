# src/tianlong/
> L2 | 父级: /CLAUDE.md

引擎包根。子包按依赖方向分层：core ← kernel, cognition ← persistence, memory, language ← agents, learning ← runtime。每个子包有自己的 L2 地图。

成员清单
__init__.py: 包版本号
__main__.py: `python -m tianlong` 入口，转交 runtime/cli
core/: 领域语言（见 core/CLAUDE.md）
kernel/: 世界规则内核（见 kernel/CLAUDE.md）
cognition/: 角色心智（见 cognition/CLAUDE.md）
persistence/: 事实持久化（见 persistence/CLAUDE.md）
memory/: 可回忆经历（见 memory/CLAUDE.md）
language/: 开放语义与文字表达（见 language/CLAUDE.md）
agents/: 智能体决策与编排（见 agents/CLAUDE.md）
scenarios/: 内容（见 scenarios/CLAUDE.md）
runtime/: 装配（见 runtime/CLAUDE.md）

[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
