"""
[INPUT]: 依赖 tianlong.persistence 的 InMemoryWorldStore / Neo4jWorldStore / net_relation_diff，tianlong.runtime.authority
[OUTPUT]: WorldStore 契约测试：两种后端跑同一组断言——往返一致、幂等、版本冲突、outbox、跨后端确定性
[POS]: tests 的持久化层；Neo4j 用例在 NEO4J_URI 不可达时自动跳过
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

from __future__ import annotations

import os
import uuid
from dataclasses import replace

import pytest

from tianlong.cognition import BeliefStore
from tianlong.core import Intent, Op, Rel, Relation
from tianlong.persistence import CommitBatch, InMemoryWorldStore, VersionConflict
from tianlong.runtime.authority import WorldAuthority
from tianlong.scenarios import build_warehouse

SCRIPT = [
    [("player", Op.TAKE, "key"), ("guard", Op.MOVE, "warehouse", "door_main")],
    [("player", Op.UNLOCK, "door_store", "key"), ("guard", Op.INSPECT, "player")],
    [("player", Op.MOVE, "storeroom", "door_store"), ("guard", Op.TELL, "player")],   # 语法非法 → 被拒也要可回放
    [("player", Op.PUT, "storeroom", "key")],
]


def _neo4j_store():
    pytest.importorskip("neo4j")
    if not os.environ.get("NEO4J_URI"):
        pytest.skip("NEO4J_URI 未配置")
    from neo4j.exceptions import ServiceUnavailable

    from tianlong.persistence.neo4j_store import Neo4jWorldStore
    try:
        return Neo4jWorldStore.from_env()
    except (ServiceUnavailable, OSError) as e:  # pragma: no cover
        pytest.skip(f"Neo4j 不可达: {e}")


@pytest.fixture(params=["memory", pytest.param("neo4j", marks=pytest.mark.neo4j)])
def store(request):
    if request.param == "memory":
        yield InMemoryWorldStore()
        return
    s = _neo4j_store()
    s.created = []  # type: ignore[attr-defined]
    yield s
    for ref in s.created:  # type: ignore[attr-defined]
        s.drop_world(ref)
    s.close()


def found(store):
    sc = replace(build_warehouse(), world_id=f"t-{uuid.uuid4().hex[:8]}")
    auth = WorldAuthority.found(store, sc)
    if hasattr(store, "created"):
        store.created.append(auth.ref)
    return sc, auth


def play(auth):
    for tick, specs in enumerate(SCRIPT):
        v = auth.head().version
        auth.settle([Intent(f"s{tick}-{s[0]}", s[0], s[1], *s[2:], based_on=v) for s in specs])


def test_create_round_trip(store):
    sc, auth = found(store)
    assert auth.head().fingerprint() == sc.state.fingerprint()
    for agent in sc.profiles:
        expected = BeliefStore(agent, trust=dict(sc.profiles[agent].trust)).revise_all(sc.priors[agent])[0]
        assert store.beliefs(auth.ref, agent) == expected
    with pytest.raises(ValueError):
        store.create(auth.ref, sc.state, {})


def test_backends_agree_after_play(store):
    _, auth = found(store)
    play(auth)
    _, ref_auth = found(InMemoryWorldStore())
    play(ref_auth)
    assert auth.head().fingerprint() == ref_auth.head().fingerprint()
    assert [e.id for e in store.events(auth.ref)] == [e.id for e in ref_auth.store.events(ref_auth.ref)]
    for agent in ("player", "guard", "captain"):
        assert store.beliefs(auth.ref, agent) == ref_auth.store.beliefs(ref_auth.ref, agent)
    assert auth.head().target("key", Rel.AT) == "storeroom"


def test_idempotency_and_conflict(store):
    _, auth = found(store)
    it = Intent("once", "player", Op.TAKE, "key", based_on=0)
    auth.settle([it])
    assert auth.settle([it]).replayed
    assert store.event_for_intent(auth.ref, "once").outcome.value == "success"
    head = store.head(auth.ref)
    stale = auth.kernel.step(head, [Intent("x", "guard", Op.WAIT, based_on=head.version)])
    auth.settle([Intent("y", "guard", Op.WAIT, based_on=head.version)])
    with pytest.raises(VersionConflict):
        store.commit(CommitBatch(auth.ref, head.version, stale.state, stale.events, (), {}, ()))


def test_outbox_cycle(store):
    _, auth = found(store)
    auth.settle([Intent("t", "player", Op.TAKE, "key", based_on=0)])
    mine = [m for m in store.pending_memories(1000) if m.world_id == auth.ref.world_id]
    assert any(m.owner == "guard" and "响动" in m.text for m in mine)
    store.mark_indexed([m.id for m in mine])
    assert not [m for m in store.pending_memories(1000) if m.world_id == auth.ref.world_id]
    assert store.recent_memories(auth.ref, "guard", 0)


def test_net_relation_diff_folds_chains():
    pytest.importorskip("neo4j")
    from tianlong.core import relocate
    from tianlong.persistence.neo4j_store import net_relation_diff
    changes = [*relocate("key", "table", "player"), *relocate("key", "player", "guard")]
    removed, added = net_relation_diff(changes)
    assert removed == {Relation("key", Rel.AT, "table")}
    assert added == {Relation("key", Rel.AT, "guard")}
