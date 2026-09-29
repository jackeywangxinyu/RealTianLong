"""
[INPUT]: 依赖 conftest 的 authority / act / make_intent，tianlong.runtime.authority，tianlong.persistence
[OUTPUT]: 设计验收用例“仓库钥匙”的端到端测试 + 幂等、并发、回放
[POS]: tests 的验收层；逐段对照设计文档第七节：事实、观察、记忆、推测、行动与叙述通过明确接口连接
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

from __future__ import annotations

import pytest

from tianlong.core import Modality, Op, Proposition, Rel, Relation
from tianlong.persistence import CommitBatch, InMemoryWorldStore, VersionConflict
from tianlong.runtime.authority import WorldAuthority
from tianlong.scenarios import build_warehouse

from .conftest import make_intent


def beliefs(authority, agent):
    return authority.store.beliefs(authority.ref, agent)


def memories(authority, agent):
    return [m.text for m in authority.store.recent_memories(authority.ref, agent, 0)]


def test_design_acceptance_story(authority, act):
    # ---- 第一段：玩家行动进入真实世界 ----
    r = act(("player", Op.TAKE, "key"))
    s = authority.head()
    assert s.target("key", Rel.AT) == "player"
    assert s.holds(Relation("captain", Rel.OWNS, "key"))
    assert s.attr("door_store", "locked") is True

    # ---- 第二段：不同角色获得不同信息 ----
    assert beliefs(authority, "player").location_of("key") == "player"
    guard_percepts = [o.percept for o in r.observations_of("guard")]
    assert [p.modality for p in guard_percepts if p.event] == [Modality.SOUND]
    assert guard_percepts[0].event.actor is None, "听到响动，不知道是谁"
    assert beliefs(authority, "guard").location_of("key") == "table", "守卫仍以为钥匙在桌上"
    assert beliefs(authority, "captain").location_of("key") == "table", "船长毫不知情"
    assert not r.observations_of("captain") or all(p.percept.event is None for p in r.observations_of("captain"))
    assert any("响动" in t for t in memories(authority, "guard"))

    # ---- 第四段：守卫进去查看，行动再次进入实际结算 ----
    act(("guard", Op.MOVE, "warehouse", "door_main"))
    g = beliefs(authority, "guard")
    assert not g.believed(Proposition.rel("key", Rel.AT, "table")).holds, "发现钥匙不见了"
    assert g.location_of("key") is None, "但不知道谁拿走了它"
    assert not g.holds(Proposition.rel("key", Rel.AT, "player"))
    assert any("钥匙不在桌面" in t for t in memories(authority, "guard"))

    # ---- 搜身之后才知道 ----
    act(("guard", Op.INSPECT, "player"))
    assert beliefs(authority, "guard").location_of("key") == "player"
    assert beliefs(authority, "captain").location_of("key") == "table", "船长依旧不知道"


def test_settle_is_idempotent(authority):
    it = make_intent("player", Op.TAKE, "key", based_on=0, intent_id="once")
    first = authority.settle([it])
    again = authority.settle([it])
    assert again.replayed and again.events == tuple(e for e in first.events if e.intent.id == "once")
    assert authority.head().version == 1
    assert authority.head().target("key", Rel.AT) == "player"


def test_rejected_intent_is_also_idempotent(authority):
    it = make_intent("player", Op.TAKE, "key", based_on=99, intent_id="stale")
    authority.settle([it])
    assert authority.settle([it]).replayed


def test_second_writer_gets_version_conflict():
    store = InMemoryWorldStore()
    a = WorldAuthority.found(store, build_warehouse())
    head = store.head(a.ref)
    # 另一个写入者基于同一版本算出了结果……
    stale = a.kernel.step(head, [make_intent("player", Op.WAIT, based_on=0)])
    # ……但权威写入器抢先提交了
    a.settle([make_intent("guard", Op.WAIT, based_on=0)])
    with pytest.raises(VersionConflict):
        store.commit(CommitBatch(a.ref, head.version, stale.state, stale.events, (), {}, ()))


def test_replay_reproduces_world():
    script = [
        [("player", Op.TAKE, "key"), ("guard", Op.MOVE, "warehouse", "door_main")],
        [("player", Op.UNLOCK, "door_store", "key"), ("guard", Op.INSPECT, "player")],
        [("player", Op.MOVE, "storeroom", "door_store"), ("captain", Op.MOVE, "entrance", "path")],
    ]

    def run():
        auth = WorldAuthority.found(InMemoryWorldStore(), build_warehouse())
        for tick, specs in enumerate(script):
            v = auth.head().version
            auth.settle([make_intent(*spec, based_on=v, intent_id=f"t{tick}-{spec[0]}") for spec in specs])
        return auth

    a, b = run(), run()
    assert a.head().fingerprint() == b.head().fingerprint()
    assert [e.id for e in a.store.events(a.ref)] == [e.id for e in b.store.events(b.ref)]
    assert a.head().target("player", Rel.AT) == "storeroom"
