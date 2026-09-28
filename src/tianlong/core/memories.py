"""
[INPUT]: 依赖标准库 dataclasses
[OUTPUT]: 对外提供 MemoryRecord（角色可回忆经历的权威记录）
[POS]: core 的经历记录；由权威写入器在世界提交的同一事务里写入（outbox），再由 memory 层索引进向量库——向量索引只是可重建的派生数据
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class MemoryRecord:
    id: str
    world_id: str
    branch_id: str
    owner: str
    kind: str                     # event / discovery / speech
    text: str                     # 角色视角的自然语言
    occurred_at: int              # 事情发生的游戏时间
    known_at: int                 # 角色获知的游戏时间：检索时强制 known_at <= now
    source: str                   # 来源观察 ID（溯源用，不交给角色）
    subjects: tuple[str, ...] = ()  # 涉及的实体
