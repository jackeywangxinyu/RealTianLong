"""
[INPUT]: 依赖 tianlong.kernel 的 Kernel / rules.combat，tianlong.core 的类型，conftest 的 make_intent
[OUTPUT]: 武侠机制测试：暗门（夜现）与单向断崖、动手的伤→制与限时自解、闪避与吸功、毒与解药、研读累积与私密、搜走被制者之物、
          程序化世界的江湖层不改底图、私密属性永不外泄
[POS]: tests 的内核扩展层；验证新机制都走同一条“规则插件 + 感知 + 不变量”的路，而非特判
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

from __future__ import annotations

import pytest

from tianlong.core import (
    Entity,
    Fact,
    Kind,
    Modality,
    Op,
    Outcome,
    Proposition,
    Rel,
    Relation,
    WorldState,
    at,
    is_private_attr,
)
from tianlong.kernel import Kernel
from tianlong.kernel.rules.combat import SUBDUE_TICKS

from .conftest import make_intent


@pytest.fixture
def kernel():
    return Kernel()


def dojo(seed: int = 1, clock: int = at(1, 12, 0), **attrs) -> WorldState:
    """两人一室，外加崖顶—崖底（单向）与崖底暗门（夜现）。"""
    ents = [
        Entity.make("room", Kind.PLACE, "练武场"), Entity.make("top", Kind.PLACE, "崖顶"),
        Entity.make("bottom", Kind.PLACE, "崖底"), Entity.make("cave", Kind.PLACE, "石洞"),
        Entity.make("stair", Kind.DOOR, "石阶"), Entity.make("cliff", Kind.DOOR, "断崖", oneway="bottom"),
        Entity.make("crack", Kind.DOOR, "石缝", hidden=True, night_only=True, clue="wall"),
        Entity.make("wall", Kind.SURFACE, "石壁"),
        Entity.make("a", Kind.PERSON, "甲", martial=attrs.get("a", 0.6)),
        Entity.make("b", Kind.PERSON, "乙", martial=attrs.get("b", 0.0), evasion=attrs.get("evasion"),
                    absorb=attrs.get("absorb")),
        Entity.make("fang", Kind.ITEM, "毒牙", small=True, weapon=True, venom=True, edge=0.4),
        Entity.make("pill", Kind.ITEM, "解药", small=True, cures="poisoned"),
        Entity.make("book", Kind.ITEM, "秘籍", small=True, teaches="evasion", difficulty=3),
    ]
    rels = [Relation("stair", Rel.CONNECTS, "room"), Relation("stair", Rel.CONNECTS, "top"),
            Relation("cliff", Rel.CONNECTS, "top"), Relation("cliff", Rel.CONNECTS, "bottom"),
            Relation("crack", Rel.CONNECTS, "bottom"), Relation("crack", Rel.CONNECTS, "cave"),
            Relation("wall", Rel.AT, "bottom"),
            Relation("a", Rel.AT, "room"), Relation("b", Rel.AT, "room"),
            Relation("pill", Rel.AT, "b"), Relation("book", Rel.AT, "b")]
    if attrs.get("fanged"):
        rels.append(Relation("fang", Rel.AT, "a"))
    else:
        rels.append(Relation("fang", Rel.AT, "room"))
    return WorldState.build(seed, clock, ents, rels)


def step(kernel, s, *specs, **kw):
    return kernel.step(s, [make_intent(*spec, based_on=s.version, **kw) for spec in specs])


def self_percept(r, who):
    return next(o.percept for o in r.observations if o.observer == who and o.percept.modality == Modality.SELF)


# ============================================================
#  暗门与单向通道
# ============================================================


def _at_bottom(s: WorldState) -> WorldState:
    rels = set(s.relations) - {Relation("b", Rel.AT, "room")} | {Relation("b", Rel.AT, "bottom")}
    return WorldState.build(s.seed, s.clock, s.entities.values(), rels)


def test_hidden_passage_only_by_night_and_never_in_scene(kernel):
    day = _at_bottom(dojo())
    r = step(kernel, day, ("b", Op.INSPECT, "wall"))
    assert not any(f.prop.subject == "crack" for f in self_percept(r, "b").facts), "白日里玉璧只是一面石壁"
    scene = next(o.percept for o in r.observations if o.observer == "b" and o.percept.modality == Modality.SCENE)
    assert not any(f.prop.subject == "crack" for f in scene.facts), "暗门永不出现在环顾里"
    night = _at_bottom(dojo(clock=at(1, 20, 0)))
    r = step(kernel, night, ("b", Op.INSPECT, "wall"))
    assert Fact(Proposition.rel("crack", Rel.CONNECTS, "cave")) in self_percept(r, "b").facts
    r = step(kernel, r.state, ("b", Op.MOVE, "cave", "crack"))
    assert r.state.target("b", Rel.AT) == "cave"


def test_cliff_is_one_way(kernel):
    s = step(kernel, dojo(), ("b", Op.MOVE, "top", "stair")).state
    s = step(kernel, s, ("b", Op.MOVE, "bottom", "cliff")).state
    assert s.target("b", Rel.AT) == "bottom"
    r = step(kernel, s, ("b", Op.MOVE, "top", "cliff"))
    e = next(e for e in r.events if e.actor == "b")
    assert (e.outcome, e.reason) == (Outcome.FAILURE, "one_way")
    assert Fact(Proposition.attr("cliff", "oneway", "bottom")) in self_percept(r, "b").facts


# ============================================================
#  动手：伤 → 制 → 限时自解
# ============================================================


def test_wound_then_subdue_then_recover(kernel):
    s = dojo()
    r = step(kernel, s, ("a", Op.ATTACK, "b"))
    assert r.state.attr("b", "wounded") is True
    r = step(kernel, r.state, ("a", Op.ATTACK, "b"))
    s = r.state
    assert s.attr("b", "subdued_until") == s.clock - 1 + SUBDUE_TICKS
    r = step(kernel, s, ("b", Op.MOVE, "top", "stair"))
    assert next(e for e in r.events if e.actor == "b").reason == "subdued"
    r = step(kernel, r.state, ("b", Op.TELL, "a", None), topic=Fact(Proposition.rel("book", Rel.AT, "b")))
    assert next(e for e in r.events if e.actor == "b").outcome == Outcome.SUCCESS, "穴道被制，嘴还能说"
    later = r.state.stamp(r.state.version, r.state.clock + SUBDUE_TICKS)
    r = step(kernel, later, ("b", Op.MOVE, "top", "stair"))
    assert r.state.target("b", Rel.AT) == "top", "时辰一到，穴道自解"


def test_evasion_makes_attacks_miss():
    k = Kernel()
    hits_plain = sum(step(k, dojo(seed, a=0.5, b=0.3), ("a", Op.ATTACK, "b")).state.attr("b", "wounded") is True
                     for seed in range(40))
    hits_dodge = sum(step(k, dojo(seed, a=0.5, b=0.3, evasion=True),
                          ("a", Op.ATTACK, "b")).state.attr("b", "wounded") is True for seed in range(40))
    assert hits_plain > 30 and hits_dodge < 5


def test_absorb_drains_bare_handed_attacker(kernel):
    r = step(kernel, dojo(b=0.1, absorb=True), ("a", Op.ATTACK, "b"))
    assert r.state.attr("a", "martial") == pytest.approx(0.5) and r.state.attr("b", "martial") == pytest.approx(0.2)
    armed = step(kernel, dojo(b=0.1, absorb=True, fanged=True), ("a", Op.ATTACK, "b"))
    assert armed.state.attr("a", "martial") == pytest.approx(0.6), "持兵刃者不受吸功"


def test_venom_poisons_and_antidote_cures(kernel):
    r = step(kernel, dojo(fanged=True), ("a", Op.ATTACK, "b"))
    assert r.state.attr("b", "poisoned") is True and not r.state.attr("b", "wounded")
    r = step(kernel, r.state, ("b", Op.USE, "b", "pill"))
    assert r.state.attr("b", "poisoned") is None
    r = step(kernel, r.state, ("b", Op.USE, "a", "pill"))
    assert next(e for e in r.events if e.actor == "b").reason == "no_effect"


def test_take_from_subdued_but_not_from_standing(kernel):
    s = dojo()
    r = step(kernel, s, ("a", Op.TAKE, "pill"))
    assert next(e for e in r.events if e.actor == "a").reason == "held_by_other"
    s = step(kernel, step(kernel, r.state, ("a", Op.ATTACK, "b")).state, ("a", Op.ATTACK, "b")).state
    r = step(kernel, s, ("a", Op.TAKE, "pill"))
    assert r.state.target("pill", Rel.AT) == "a"


# ============================================================
#  研读：累积、学成、私密
# ============================================================


def test_study_accumulates_privately(kernel):
    s = dojo()
    reasons = []
    for _ in range(3):
        r = step(kernel, s, ("b", Op.STUDY, "book"))
        reasons.append(next(e for e in r.events if e.actor == "b").reason)
        witness = next(o.percept for o in r.observations if o.observer == "a" and o.percept.event)
        assert witness.event.reason is None and witness.facts == (), "旁人看不出学没学成"
        s = r.state
    assert reasons == ["progress", "progress", "mastered"] and s.attr("b", "evasion") is True
    assert Fact(Proposition.attr("b", "evasion", True)) in self_percept(r, "b").facts


def test_jianghu_layer_keeps_base_world_intact():
    """江湖层用独立随机流叠加：底图（布局、锁、原有物品与目标）不变，只多出身手、兵刃、解药等。"""
    from tianlong.scenarios.procedural import random_scenario

    for seed in range(40):
        base, plain, wuxia = random_scenario(seed), random_scenario(seed, jianghu=0.0), random_scenario(seed, jianghu=1.0)
        assert base.state.fingerprint() == plain.state.fingerprint(), "jianghu=0 与旧版逐字节相同"
        assert set(base.state.relations) <= set(wuxia.state.relations)
        assert {"w0", "c0"} <= set(wuxia.state.entities) and "w0" not in base.state.entities
        assert all(wuxia.state.attr(h, "martial") is not None for h in wuxia.profiles)
        for h, p in base.profiles.items():
            assert wuxia.profiles[h].goals[:len(p.goals)] == p.goals, "原有目标保留，江湖目标只追加"


def test_private_attrs_never_reach_other_people(kernel):
    """私密属性（内力、进度、点穴时限）永不外泄给旁人；本人的环顾里有自我感知。"""
    s = dojo(b=0.1, absorb=True)
    for spec in [("b", Op.STUDY, "book"), ("a", Op.ATTACK, "b"), ("a", Op.ATTACK, "b"), ("a", Op.ATTACK, "b")]:
        r = step(kernel, s, spec)
        for o in r.observations:
            leaked = [f for f in o.percept.facts
                      if f.prop.is_attr and is_private_attr(f.prop.attr_key) and f.prop.subject != o.observer]
            assert not leaked, o
            if o.percept.modality != Modality.SCENE:
                assert not any(f.prop.is_attr and is_private_attr(f.prop.attr_key) for f in o.percept.facts), o
        s = r.state
    mine = next(o.percept for o in r.observations if o.observer == "b" and o.percept.modality == Modality.SCENE)
    assert Fact(Proposition.attr("b", "progress_evasion", 1)) in mine.facts, "自己读到哪了，自己清楚"
