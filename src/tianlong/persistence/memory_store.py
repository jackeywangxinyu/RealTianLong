"""
[INPUT]: 依赖 persistence/store 的协议与值对象，core / cognition 的不可变类型
[OUTPUT]: 对外提供 InMemoryWorldStore
[POS]: persistence 的内存实现；测试、训练、离线游玩的默认后端。与 Neo4j 实现遵守同一协议，用同一组契约测试验证
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

from __future__ import annotations

import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from tianlong.cognition import BeliefStore
from tianlong.core import Event, Observation, WorldState
from tianlong.core.memories import MemoryRecord
from tianlong.persistence.store import CommitBatch, UnknownWorld, VersionConflict, WorldRef


@dataclass
class _World:
    head: WorldState
    beliefs: dict[str, BeliefStore]
    events: list[Event] = field(default_factory=list)
    observations: list[Observation] = field(default_factory=list)
    intent_index: dict[str, Event] = field(default_factory=dict)
    memories: list[MemoryRecord] = field(default_factory=list)


class InMemoryWorldStore:
    """所有值都是不可变对象，直接保存引用即可；锁只保护“检查版本 + 写入”这一临界区。"""

    def __init__(self) -> None:
        self._worlds: dict[WorldRef, _World] = {}
        self._pending: dict[str, MemoryRecord] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------
    #  生命周期
    # ------------------------------------------------------------

    def create(self, ref: WorldRef, state: WorldState, beliefs: Mapping[str, BeliefStore]) -> None:
        with self._lock:
            if ref in self._worlds:
                raise ValueError(f"世界已存在: {ref}")
            self._worlds[ref] = _World(state, dict(beliefs))

    def exists(self, ref: WorldRef) -> bool:
        return ref in self._worlds

    # ------------------------------------------------------------
    #  读
    # ------------------------------------------------------------

    def _world(self, ref: WorldRef) -> _World:
        try:
            return self._worlds[ref]
        except KeyError:
            raise UnknownWorld(str(ref)) from None

    def head(self, ref: WorldRef) -> WorldState:
        return self._world(ref).head

    def beliefs(self, ref: WorldRef, agent: str) -> BeliefStore:
        return self._world(ref).beliefs.get(agent) or BeliefStore(agent)

    def event_for_intent(self, ref: WorldRef, intent_id: str) -> Event | None:
        return self._world(ref).intent_index.get(intent_id)

    def events(self, ref: WorldRef) -> tuple[Event, ...]:
        return tuple(self._world(ref).events)

    def observations(self, ref: WorldRef) -> tuple[Observation, ...]:
        return tuple(self._world(ref).observations)

    def recent_memories(self, ref: WorldRef, owner: str, since: int) -> tuple[MemoryRecord, ...]:
        return tuple(m for m in self._world(ref).memories if m.owner == owner and m.known_at >= since)

    # ------------------------------------------------------------
    #  写
    # ------------------------------------------------------------

    def commit(self, batch: CommitBatch) -> None:
        with self._lock:
            w = self._world(batch.ref)
            if w.head.version != batch.expected_version:
                raise VersionConflict(f"{batch.ref}: head={w.head.version} expected={batch.expected_version}")
            if batch.state.version != batch.expected_version + 1:
                raise ValueError("新状态版本必须恰好 +1")
            w.head = batch.state
            w.beliefs.update(batch.beliefs)
            w.events.extend(batch.events)
            w.observations.extend(batch.observations)
            for e in batch.events:
                w.intent_index[e.intent.id] = e
            w.memories.extend(batch.memories)
            for m in batch.memories:
                self._pending[m.id] = m

    # ------------------------------------------------------------
    #  outbox
    # ------------------------------------------------------------

    def pending_memories(self, limit: int = 256) -> tuple[MemoryRecord, ...]:
        with self._lock:
            return tuple(list(self._pending.values())[:limit])

    def mark_indexed(self, ids: Sequence[str]) -> None:
        with self._lock:
            for i in ids:
                self._pending.pop(i, None)
