"""
[INPUT]: 依赖 persistence 的 WorldStore（outbox 读取与确认），memory/index 的 MemoryIndex
[OUTPUT]: 对外提供 MemoryIndexer（outbox → 向量索引的同步器）
[POS]: memory 的写路径后半段；先写索引、后确认 outbox，崩溃重放只会重复覆盖同一 UUID，不会丢也不会重
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

from __future__ import annotations

from tianlong.memory.index import MemoryIndex
from tianlong.persistence import WorldStore


class MemoryIndexer:
    def __init__(self, store: WorldStore, index: MemoryIndex) -> None:
        self.store = store
        self.index = index

    def drain(self, batch: int = 256) -> int:
        total = 0
        while True:
            pending = self.store.pending_memories(batch)
            if not pending:
                return total
            self.index.upsert(pending)
            self.store.mark_indexed([m.id for m in pending])
            total += len(pending)
