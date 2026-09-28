"""
[INPUT]: 依赖 tianlong.kernel / cognition / scenarios，conftest 的 make_intent
[OUTPUT]: 认知隔离的性质测试：扰动角色未观察到的世界事实，其认知与认知视图必须逐字节不变
[POS]: tests 的隔离层；“没收到消息的人不能提前知道结果”被写成可证伪的断言
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

from __future__ import annotations

from dataclasses import fields

import pytest

from tianlong.cognition import BeliefStore, belief_view
from tianlong.core import Entity, Kind, Op, Percept, Rel, Relation, WorldState
from tianlong.kernel import Kernel
from tianlong.scenarios import build_warehouse

from .conftest import make_intent


def _perturbed(variant: str) -> WorldState:
    """在守卫看不到的地方改动真相。"""
    s = build_warehouse().state
    ents = dict(s.entities)
    rels = set(s.relations)
    if variant == "ledger_moved":
        rels.remove(Relation("ledger", Rel.AT, "storeroom"))
        rels.add(Relation("ledger", Rel.AT, "harbor"))
    elif variant == "door_unlocked":
        ents["door_store"] = ents["door_store"].with_attr("locked", False)
    elif variant == "extra_secret":
        ents["gold"] = Entity.make("gold", Kind.ITEM, "金条")
        rels.add(Relation("gold", Rel.AT, "storeroom"))
    return WorldState.build(s.seed, s.clock, ents.values(), rels)


def _guard_after(state: WorldState, ticks: int = 3) -> BeliefStore:
    sc = build_warehouse()
    store = BeliefStore("guard").revise_all(sc.priors["guard"])[0]
    k = Kernel()
    for _ in range(ticks):
        r = k.step(state, [make_intent("guard", Op.WAIT, based_on=state.version, intent_id=f"w{state.version}")])
        store = store.revise_all(o.percept for o in r.observations if o.observer == "guard")[0]
        state = r.state
    return store


@pytest.mark.parametrize("variant", ["ledger_moved", "door_unlocked", "extra_secret"])
def test_unobserved_truth_does_not_leak(variant):
    base = _guard_after(build_warehouse().state)
    other = _guard_after(_perturbed(variant))
    assert base == other
    assert belief_view(base, 999) == belief_view(other, 999)


def test_percept_has_no_provenance_field():
    names = {f.name for f in fields(Percept)}
    assert "source_event" not in names and "id" not in names, "角色拿到的感知不能顺着 ID 摸到真相"


def test_sound_percept_reveals_nothing_about_actor():
    s = build_warehouse().state
    r = Kernel().step(s, [make_intent("player", Op.TAKE, "key", based_on=0)])
    heard = [o.percept for o in r.observations if o.observer == "guard" and o.percept.event]
    assert heard and heard[0].facts == () and {sk.id for sk in heard[0].sketches} <= {"warehouse"}
