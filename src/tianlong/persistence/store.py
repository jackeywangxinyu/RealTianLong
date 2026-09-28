"""
[INPUT]: 依赖 core 的 WorldState / Event / Observation / MemoryRecord，cognition 的 BeliefStore
[OUTPUT]: 对外提供 WorldRef / CommitBatch / VersionConflict / UnknownWorld / WorldStore 协议
[POS]: persistence 的契约；内存实现与 Neo4j 实现都遵守它。commit 是唯一写路径：版本检查 + 世界变化 + 事件 + 观察 + 认知 + 待索引经历，一次原子提交
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

from tianlong.cognition import BeliefStore
from tianlong.core import Event, Observation, WorldState
from tianlong.core.memories import MemoryRecord


@dataclass(frozen=True, slots=True)
class WorldRef:
    world_id: str
    branch_id: str = "main"

    def __str__(self) -> str:
        return f"{self.world_id}@{self.branch_id}"


@dataclass(frozen=True, slots=True)
class CommitBatch:
    ref: WorldRef
    expected_version: int              # 乐观并发：只有 head 仍是这个版本才允许提交
    state: WorldState                  # 新版本
    events: tuple[Event, ...]
    observations: tuple[Observation, ...]
    beliefs: Mapping[str, BeliefStore]  # 本次发生变化的角色认知（整份替换）
    memories: tuple[MemoryRecord, ...]  # outbox：待索引的经历


class VersionConflict(Exception):
    """head 已不是 expected_version：有别的写入者抢先提交了。"""


class UnknownWorld(KeyError):
    pass


class WorldStore(Protocol):
    # ---- 生命周期 ----
    def create(self, ref: WorldRef, state: WorldState, beliefs: Mapping[str, BeliefStore]) -> None: ...
    def exists(self, ref: WorldRef) -> bool: ...

    # ---- 读 ----
    def head(self, ref: WorldRef) -> WorldState: ...
    def beliefs(self, ref: WorldRef, agent: str) -> BeliefStore: ...
    def event_for_intent(self, ref: WorldRef, intent_id: str) -> Event | None: ...
    def events(self, ref: WorldRef) -> tuple[Event, ...]: ...
    def recent_memories(self, ref: WorldRef, owner: str, since: int) -> tuple[MemoryRecord, ...]: ...

    # ---- 写（唯一路径）----
    def commit(self, batch: CommitBatch) -> None: ...

    # ---- outbox ----
    def pending_memories(self, limit: int = 256) -> tuple[MemoryRecord, ...]: ...
    def mark_indexed(self, ids: Sequence[str]) -> None: ...
