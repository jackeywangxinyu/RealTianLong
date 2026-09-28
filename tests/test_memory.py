"""
[INPUT]: 依赖 tianlong.memory 的 embedder / index / recall / indexer，conftest 的 authority / act
[OUTPUT]: 记忆层测试：检索边界强制过滤、outbox 幂等同步、最近缓冲弥补索引延迟、嵌入确定性
[POS]: tests 的记忆层；验证“回忆不是现状”与“向量索引只是派生数据”
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

from __future__ import annotations

import pytest

pytest.importorskip("qdrant_client")

from tianlong.core import Op  # noqa: E402
from tianlong.core.memories import MemoryRecord  # noqa: E402
from tianlong.memory.embedder import HashingEmbedder  # noqa: E402
from tianlong.memory.index import MemoryScope, QdrantMemoryIndex  # noqa: E402
from tianlong.memory.indexer import MemoryIndexer  # noqa: E402
from tianlong.memory.recall import Recall  # noqa: E402


def rec(rid, owner="guard", text="仓库那边传来一阵响动", known_at=10, branch="main", world="w", kind="event"):
    return MemoryRecord(rid, world, branch, owner, kind, text, known_at, known_at, f"obs-{rid}", ())


def test_embedder_is_deterministic_and_lexical():
    e = HashingEmbedder(dim=64)
    a, b, c = e.embed(["听到仓库那边传来一阵响动", "仓库有响动", "船长在港口喝茶"])
    assert a == e.embed(["听到仓库那边传来一阵响动"])[0]
    dot = lambda x, y: sum(i * j for i, j in zip(x, y, strict=True))  # noqa: E731
    assert dot(a, b) > dot(a, c)


def test_scope_filters_are_enforced():
    idx = QdrantMemoryIndex()
    idx.upsert([
        rec("mine"),
        rec("captains", owner="captain"),
        rec("future", known_at=99),
        rec("other_branch", branch="alt"),
        rec("other_world", world="w2"),
    ])
    hits = idx.search(MemoryScope("w", "main", "guard", now=20), "响动", limit=10)
    assert [h.record.id for h in hits] == ["mine"]


def test_kind_filter():
    idx = QdrantMemoryIndex()
    idx.upsert([rec("e"), rec("s", kind="speech")])
    hits = idx.search(MemoryScope("w", "main", "guard", now=20, kinds=("speech",)), "响动")
    assert [h.record.id for h in hits] == ["s"]


def test_outbox_drain_is_idempotent_and_recall_bridges_index_lag(authority, act):
    index = QdrantMemoryIndex()
    recall = Recall(authority.store, index)
    act(("player", Op.TAKE, "key"))
    now = authority.head().clock
    scope = MemoryScope(authority.ref.world_id, authority.ref.branch_id, "guard", now)

    # 索引尚未同步：工作记忆里已经有了
    before = recall.recall(scope, "响动")
    assert any("响动" in m.text for m in before.recent)
    assert not index.search(scope, "响动")

    indexer = MemoryIndexer(authority.store, index)
    n = indexer.drain()
    assert n > 0 and indexer.drain() == 0
    index.upsert(authority.store.recent_memories(authority.ref, "guard", 0))  # 重放覆盖同一 UUID
    hits = index.search(scope, "响动", limit=10)
    assert len({h.record.id for h in hits}) == len(hits)
    assert all(h.record.owner == "guard" for h in hits)


def test_recall_dedups_recent_and_related(authority, act):
    index = QdrantMemoryIndex()
    act(("player", Op.TAKE, "key"))
    MemoryIndexer(authority.store, index).drain()
    now = authority.head().clock
    r = Recall(authority.store, index).recall(MemoryScope("warehouse", "main", "guard", now), "响动")
    assert not {m.id for m in r.recent} & {x.record.id for x in r.related}
