"""
[INPUT]: 依赖 tianlong.kernel 的 Kernel / violations，tianlong.core 的类型，conftest 的 make_intent
[OUTPUT]: 世界规则内核的单元测试：前置条件、失败获知、冲突结算、藏匿与搜查、准入拒绝、不变量、确定性
[POS]: tests 的内核层；验证“事实只由规则裁定”
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

from __future__ import annotations

import pytest

from tianlong.core import (
    Entity,
    Fact,
    Kind,
    Manner,
    Modality,
    Op,
    Outcome,
    Proposition,
    Rel,
    Relation,
    WorldState,
)
from tianlong.kernel import Kernel, violations
from tianlong.kernel.perception import audibility

from .conftest import make_intent


@pytest.fixture
def kernel():
    return Kernel()


def only(result, observer, modality=None):
    return [o.percept for o in result.observations
            if o.observer == observer and (modality is None or o.percept.modality == modality)]


def event_of(result, actor):
    return next(e for e in result.events if e.actor == actor)


# ============================================================
#  位置与所有权正交
# ============================================================


def test_take_moves_location_not_ownership(kernel, warehouse):
    s = warehouse.state
    r = kernel.step(s, [make_intent("player", Op.TAKE, "key", based_on=s.version)])
    assert r.state.target("key", Rel.AT) == "player"
    assert r.state.holds(Relation("captain", Rel.OWNS, "key")), "拿走不等于拥有"
    assert r.state.attr("door_store", "locked") is True, "钥匙离开桌面，门不会自动打开"
    assert r.state.version == s.version + 1 and r.state.clock == s.clock + 1


def test_locked_door_blocks_and_teaches(kernel, warehouse):
    s = warehouse.state
    r = kernel.step(s, [make_intent("player", Op.MOVE, "storeroom", "door_store", based_on=s.version)])
    e = event_of(r, "player")
    assert (e.outcome, e.reason) == (Outcome.FAILURE, "door_locked")
    assert r.state.target("player", Rel.AT) == "warehouse"
    self_p = only(r, "player", Modality.SELF)[0]
    assert Fact(Proposition.attr("door_store", "locked", True)) in self_p.facts


def test_unlock_then_enter(kernel, warehouse):
    s = warehouse.state
    s = kernel.step(s, [make_intent("player", Op.TAKE, "key", based_on=s.version)]).state
    r = kernel.step(s, [make_intent("player", Op.UNLOCK, "door_store", "key", based_on=s.version)])
    assert event_of(r, "player").outcome == Outcome.SUCCESS
    assert r.state.attr("door_store", "locked") is False
    s = r.state
    r = kernel.step(s, [make_intent("player", Op.MOVE, "storeroom", "door_store", based_on=s.version)])
    assert r.state.target("player", Rel.AT) == "storeroom"


def test_wrong_key_learns_mismatch(kernel):
    ents = [
        Entity.make("a", Kind.PLACE, "甲"), Entity.make("b", Kind.PLACE, "乙"),
        Entity.make("d", Kind.DOOR, "门", locked=True), Entity.make("k", Kind.ITEM, "错钥匙"),
        Entity.make("p", Kind.PERSON, "人"),
    ]
    rels = [Relation("d", Rel.CONNECTS, "a"), Relation("d", Rel.CONNECTS, "b"),
            Relation("p", Rel.AT, "a"), Relation("k", Rel.AT, "p")]
    s = WorldState.build(1, 0, ents, rels)
    r = kernel.step(s, [make_intent("p", Op.UNLOCK, "d", "k", based_on=0)])
    assert event_of(r, "p").reason == "wrong_key"
    assert Fact(Proposition.rel("k", Rel.MATCHES, "d"), False) in only(r, "p", Modality.SELF)[0].facts


# ============================================================
#  同时行动：统一结算，唯一物品只有一人能得到
# ============================================================


def _two_grabbers(seed: int) -> WorldState:
    ents = [Entity.make("room", Kind.PLACE, "屋"), Entity.make("t", Kind.SURFACE, "桌"),
            Entity.make("gem", Kind.ITEM, "宝石"),
            Entity.make("a", Kind.PERSON, "甲", agility=0.5), Entity.make("b", Kind.PERSON, "乙", agility=0.5)]
    rels = [Relation("t", Rel.AT, "room"), Relation("gem", Rel.AT, "t"),
            Relation("a", Rel.AT, "room"), Relation("b", Rel.AT, "room")]
    return WorldState.build(seed, 0, ents, rels)


@pytest.mark.parametrize("seed", range(20))
def test_simultaneous_take_has_exactly_one_winner(kernel, seed):
    s = _two_grabbers(seed)
    r = kernel.step(s, [make_intent("a", Op.TAKE, "gem", based_on=0), make_intent("b", Op.TAKE, "gem", based_on=0)])
    outcomes = sorted(e.outcome for e in r.events)
    assert outcomes == [Outcome.FAILURE, Outcome.SUCCESS]
    loser = next(e for e in r.events if e.outcome == Outcome.FAILURE)
    assert loser.reason == "held_by_other"
    assert not violations(r.state)


def test_winner_decided_by_rules_not_submission_order(kernel):
    s = _two_grabbers(3)
    ia = make_intent("a", Op.TAKE, "gem", based_on=0, intent_id="x-a")
    ib = make_intent("b", Op.TAKE, "gem", based_on=0, intent_id="x-b")
    r1 = kernel.step(s, [ia, ib])
    r2 = kernel.step(s, [ib, ia])
    assert r1.state.fingerprint() == r2.state.fingerprint()


def test_rough_manner_wins_initiative(kernel):
    s = _two_grabbers(0)
    r = kernel.step(s, [make_intent("a", Op.TAKE, "gem", based_on=0),
                        make_intent("b", Op.TAKE, "gem", based_on=0, manner=Manner.ROUGH)])
    assert r.state.target("gem", Rel.AT) == "b"


# ============================================================
#  藏匿与搜查
# ============================================================


def test_hide_then_inspect_reveals(kernel, warehouse):
    s = warehouse.state
    s = kernel.step(s, [make_intent("player", Op.TAKE, "key", based_on=s.version)]).state
    r = kernel.step(s, [make_intent("player", Op.PUT, "table", "key", based_on=s.version, manner=Manner.CAREFUL)])
    s = r.state
    assert s.attr("key", "hidden") is True
    # 守卫走进来环顾：看不见藏起来的钥匙
    s = kernel.step(s, [make_intent("guard", Op.MOVE, "warehouse", "door_main", based_on=s.version)]).state
    r = kernel.step(s, [make_intent("guard", Op.WAIT, based_on=s.version)])
    scene = only(r, "guard", Modality.SCENE)[0]
    assert not any(f.prop.subject == "key" for f in scene.facts)
    # 仔细查看桌面才能发现
    r = kernel.step(r.state, [make_intent("guard", Op.INSPECT, "table", based_on=r.state.version)])
    found = only(r, "guard", Modality.SELF)[0]
    assert Fact(Proposition.rel("key", Rel.AT, "table")) in found.facts
    assert "table" in found.scopes


def test_search_person_finds_concealed_small_item(kernel, warehouse):
    s = warehouse.state
    s = kernel.step(s, [make_intent("player", Op.TAKE, "key", based_on=s.version)]).state
    s = kernel.step(s, [make_intent("guard", Op.MOVE, "warehouse", "door_main", based_on=s.version)]).state
    r = kernel.step(s, [make_intent("guard", Op.INSPECT, "player", based_on=s.version)])
    assert Fact(Proposition.rel("key", Rel.AT, "player")) in only(r, "guard", Modality.SELF)[0].facts
    assert only(r, "player", Modality.SIGHT), "被搜身的人当然知道自己被搜了"


# ============================================================
#  准入：过时、重复、语法非法的意图不进入世界
# ============================================================


def test_stale_duplicate_and_syntax_rejected(kernel, warehouse):
    s = warehouse.state
    r = kernel.step(s, [
        make_intent("player", Op.TAKE, "key", based_on=s.version + 5, intent_id="a1"),
        make_intent("guard", Op.WAIT, based_on=s.version, intent_id="b1"),
        make_intent("guard", Op.WAIT, based_on=s.version, intent_id="b2"),
        make_intent("captain", Op.TAKE, "harbor", based_on=s.version, intent_id="c1"),
    ])
    reasons = {e.intent.id: (e.outcome, e.reason) for e in r.events}
    assert reasons["a1"] == (Outcome.REJECTED, "stale")
    assert reasons["b2"] == (Outcome.REJECTED, "duplicate_actor")
    assert reasons["c1"][0] == Outcome.REJECTED and reasons["c1"][1].startswith("syntax")
    assert r.state.target("key", Rel.AT) == "table"


# ============================================================
#  感知物理与确定性
# ============================================================


def test_audibility_curve():
    assert audibility(0.8, 1, 1.0) == 1.0          # 警觉守卫隔一道门必然听见正常拿取
    assert 0 < audibility(0.8 * 0.4, 1, 1.0) < 1   # 小心拿取：可能听见
    assert audibility(0.8, 3, 0.5) == 0.0


def test_careful_take_is_sometimes_unheard(kernel):
    heard = 0
    for seed in range(60):
        s = WorldState.build(seed, 0, *_warehouse_parts())
        r = kernel.step(s, [make_intent("player", Op.TAKE, "key", based_on=0, manner=Manner.CAREFUL)])
        heard += bool(only(r, "guard", Modality.SOUND))
    assert 10 < heard < 55


def _warehouse_parts():
    from tianlong.scenarios.warehouse import _entities, _relations
    return _entities(), _relations()


def test_step_is_deterministic(kernel, warehouse):
    s = warehouse.state
    intents = [make_intent("player", Op.TAKE, "key", based_on=0, intent_id="p"),
               make_intent("guard", Op.MOVE, "warehouse", "door_main", based_on=0, intent_id="g")]
    a, b = kernel.step(s, intents), kernel.step(s, list(reversed(intents)))
    assert a.state.fingerprint() == b.state.fingerprint()
    assert [o.id for o in a.observations] == [o.id for o in b.observations]


# ============================================================
#  模糊测试：程序化世界 + 随机行动，内核永不抛错、不变量永远成立
# ============================================================


def test_random_rollouts_never_break_invariants(kernel):
    import random

    from tianlong.cognition import BeliefStore, candidates
    from tianlong.scenarios.procedural import random_scenario

    for seed in range(60):
        sc = random_scenario(seed, jianghu=float(seed % 2))    # 一半江湖化：动手、毒、解药、秘籍、单向通道也要经得起乱来
        rng = random.Random(seed)
        s = sc.state
        stores = {a: BeliefStore(a).revise_all(sc.priors[a])[0] for a in sc.profiles}
        for t in range(20):
            intents = []
            for a in sorted(sc.profiles):
                c = rng.choice(candidates(stores[a]))
                intents.append(c.to_intent(f"{seed}-{t}-{a}", a, s.version))
            r = kernel.step(s, intents)          # 内核内部已断言不变量
            assert not violations(r.state)
            for o in r.observations:
                stores[o.observer], _ = stores[o.observer].revise(o.percept)
            s = r.state
