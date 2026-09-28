"""
[INPUT]: 依赖 persistence 的 WorldStore / WorldRef，memory/index 的 MemoryIndex / MemoryScope / Recollection
[OUTPUT]: 对外提供 Recall（回忆服务）、RecallResult
[POS]: memory 的读路径；工作记忆（权威存储中最近的记录，不论是否已索引）∪ 长期联想（向量检索），
       杜绝“刚看见却因索引延迟而忘记”
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

from __future__ import annotations

from dataclasses import dataclass

from tianlong.core.memories import MemoryRecord
from tianlong.memory.index import MemoryIndex, MemoryScope, Recollection
from tianlong.persistence import WorldRef, WorldStore


@dataclass(frozen=True, slots=True)
class RecallResult:
    recent: tuple[MemoryRecord, ...]     # 工作记忆：最近 window 分钟内获知的经历，新的在前
    related: tuple[Recollection, ...]    # 长期联想：与当前处境相似的过往经历（不含已在 recent 中的）


class Recall:
    def __init__(self, store: WorldStore, index: MemoryIndex, recent_window: int = 30) -> None:
        self.store = store
        self.index = index
        self.recent_window = recent_window

    def recall(self, scope: MemoryScope, query: str, limit: int = 5) -> RecallResult:
        ref = WorldRef(scope.world_id, scope.branch_id)
        recent = [
            m for m in self.store.recent_memories(ref, scope.owner, scope.now - self.recent_window)
            if m.known_at <= scope.now and (scope.kinds is None or m.kind in scope.kinds)
        ]
        recent.sort(key=lambda m: (-m.known_at, m.id))
        seen = {m.id for m in recent}
        related = [r for r in self.index.search(scope, query, limit + len(seen)) if r.record.id not in seen]
        return RecallResult(tuple(recent), tuple(related[:limit]))
