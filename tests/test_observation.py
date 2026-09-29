"""
[INPUT]: 依赖 tianlong.kernel 的 Kernel，tianlong.cognition 的 BeliefStore / candidates / belief_view，tianlong.language 的 parser / narrator，
         tianlong.scenarios 的 build_warehouse / build_wuliang，conftest 的 make_intent
[OUTPUT]: 观察投影验收（P01–P07）：远端失败移动不泄露、锁门只在门这侧留下动作与响动、成功移动两端各得其片段、
          传闻外观与真实外观隔离、只闻其名的外观是“未知”、传闻地点不触发亲眼描写、暗门不被“去某处”自动穿过、过时路线可以尝试并失败
[POS]: tests 的观察层；验证“观察由实际片段与感知渠道决定，而不是由行动的目标与真实世界决定”
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

from __future__ import annotations

import importlib.util

import pytest

from tianlong.cognition import BeliefStore, belief_view, candidates
from tianlong.core import (
    Entity,
    EntitySketch,
    Fact,
    Kind,
    Modality,
    Op,
    Outcome,
    Percept,
    Proposition,
    Rel,
    Relation,
    WorldState,
    at,
)
from tianlong.kernel import Kernel
from tianlong.scenarios import build_warehouse, build_wuliang

from .conftest import make_intent


def observed(r, who):
    return [o.percept for o in r.observations if o.observer == who and o.percept.modality != Modality.SCENE]


def _warehouse_with_guard_in(place: str) -> WorldState:
    s = build_warehouse().state
    rels = set(s.relations) - {Relation("guard", Rel.AT, "entrance")} | {Relation("guard", Rel.AT, place)}
    return WorldState.build(s.seed, s.clock, s.entities.values(), rels)


# ============================================================
#  P01 / P02：失败的移动不在目的地留下任何“看见”
# ============================================================


@pytest.mark.parametrize("door, reason", [("door_main", "route_mismatch"), ("path", "route_not_here")])
def test_p01_failed_remote_move_leaks_nothing_to_destination(door, reason):
    s = build_warehouse().state                      # 玩家在仓库，船长在港口，两地不相邻
    r = Kernel().step(s, [make_intent("player", Op.MOVE, "harbor", door, based_on=0)])
    e = r.events[0]
    assert (e.outcome, e.reason) == (Outcome.FAILURE, reason)
    assert r.state.target("player", Rel.AT) == "warehouse"
    assert observed(r, "captain") == [], "港口的人对一次没发生的抵达一无所知"
    assert not any(p.event for p in observed(r, "guard")), "隔壁也听不见：没触及世界的尝试不出声"


def test_p02_locked_door_gives_only_door_side_feedback():
    s = _warehouse_with_guard_in("storeroom")        # 守卫在锁着的门那一边
    s = WorldState.build(s.seed, s.clock, [*s.entities.values(), Entity.make("clerk", Kind.PERSON, "伙计")],
                         {*s.relations, Relation("clerk", Rel.AT, "warehouse")})
    r = Kernel().step(s, [make_intent("player", Op.MOVE, "storeroom", "door_store", based_on=0)])
    assert r.events[0].reason == "door_locked" and r.state.target("player", Rel.AT) == "warehouse"
    far = observed(r, "guard")
    assert all(p.modality == Modality.SOUND for p in far), "门那边只可能听见响动，看不见、也不知道是谁"
    assert all(p.event.actor is None for p in far)
    near = observed(r, "clerk")
    assert [p.modality for p in near] == [Modality.SIGHT]
    v = near[0].event
    assert (v.actor, v.target, v.obj, v.outcome, v.reason) == ("player", None, "door_store", Outcome.FAILURE, None), \
        "同屋的人看见他去推那道门、没推开；他想去哪、为何失败都没说出口"


def test_successful_move_gives_each_side_its_fragment():
    s = _warehouse_with_guard_in("storeroom")
    unlocked = s.entity("door_store").with_attr("locked", False)
    s = WorldState.build(s.seed, s.clock, [unlocked, *(e for e in s.entities.values() if e.id != "door_store")],
                         s.relations)
    r = Kernel().step(s, [make_intent("player", Op.MOVE, "storeroom", "door_store", based_on=0)])
    arrive = observed(r, "guard")
    assert [p.modality for p in arrive] == [Modality.SIGHT] and arrive[0].event.place == "storeroom"
    assert Fact(Proposition.rel("player", Rel.AT, "storeroom")) in arrive[0].facts


# ============================================================
#  P03 / P04 / P05：言语只传递说出的命题，外观只给亲眼所见
# ============================================================


def _tell_about_vial(cures: str):
    sc = build_warehouse()
    s = sc.state
    vial = Entity.make("vial", Kind.ITEM, "药瓶", small=True, cures=cures)
    s = WorldState.build(s.seed, s.clock, [*s.entities.values(), vial], {*s.relations, Relation("vial", Rel.AT, "harbor")})
    topic = Fact(Proposition.rel("vial", Rel.AT, "harbor"))
    r = Kernel().step(s, [make_intent("player", Op.WAIT, based_on=0),
                          make_intent("captain", Op.WAIT, based_on=0)])
    s = r.state
    s = WorldState.build(s.seed, s.clock, s.entities.values(),
                         set(s.relations) - {Relation("guard", Rel.AT, "entrance")} | {Relation("guard", Rel.AT, "warehouse")},
                         version=s.version)
    r = Kernel().step(s, [make_intent("guard", Op.TELL, "player", topic=topic, based_on=s.version)])
    heard = [p for p in observed(r, "player") if p.modality == Modality.SPEECH]
    store = BeliefStore("player").revise_all(sc.priors["player"])[0].revise_all(heard)[0]
    return heard, store


def test_p03_hearsay_does_not_carry_true_appearance():
    heard_a, store_a = _tell_about_vial("poisoned")
    heard_b, store_b = _tell_about_vial("wounded")
    assert heard_a and heard_a == heard_b, "同一句话、同样的个人知识：远处药瓶的真实外观变了，听者的感知不变"
    assert belief_view(store_a, 999) == belief_view(store_b, 999)
    sk = store_a.sketch("vial")
    assert sk is not None and not sk.seen and sk.attrs == ()


def test_p04_unseen_appearance_is_unknown_not_false():
    _, store = _tell_about_vial("poisoned")
    node = next(n for n in belief_view(store, 999).nodes if n.id == "vial")
    assert not node.knows("small") and not node.knows("weapon") and not node.knows("cures"), \
        "没见过：未知（缺席），而不是“不小/不是兵刃/不治什么”"
    seen = next(n for n in belief_view(store, 999).nodes if n.id == "key")
    assert seen.value("small") is True and seen.value("weapon") is False, "亲眼见过：外观确知"
    if importlib.util.find_spec("numpy") is not None:
        from tianlong.learning.featurize import featurize
        from tianlong.learning.schema import ATTR_BLOCK
        g = featurize(belief_view(store, 999))
        vial, key = g.x[g.index_of("vial")], g.x[g.index_of("key")]
        assert vial[ATTR_BLOCK["small"].known] == 0 and vial[ATTR_BLOCK["small"].start] == 0, "张量里：未知 = known 0"
        assert key[ATTR_BLOCK["weapon"].known] == 1 and key[ATTR_BLOCK["weapon"].start] == -1, "已知的否"
        assert vial[ATTR_BLOCK["locked"].known] == -1, "药瓶锁没锁：不适用，不是未知"


def test_p05_hearsay_place_is_not_described_as_seen():
    from tianlong.language.narrator import lore_keys
    from tianlong.scenarios.tianlong.lore import LORE

    sc = build_wuliang()
    s = sc.state
    topic = Fact(Proposition.rel("zhongling", Rel.AT, "langhuan"))
    r = Kernel().step(s, [make_intent("mawude", Op.TELL, "duanyu", topic=topic, based_on=0),
                          make_intent("zuozimu", Op.MOVE, "houyuan", "d_corridor", based_on=0)])
    percepts = [o.percept for o in r.observations if o.observer == "duanyu"]
    keys = lore_keys("duanyu", percepts, LORE)
    assert "langhuan" not in keys, "只听说的地点不写成亲眼所见"
    assert "houyuan" not in keys, "看着别人走向的地点，自己并没看见"


# ============================================================
#  P06 / P07：路线绑定在角色知道的门上
# ============================================================


def _grotto(clock: int = at(1, 12, 0)) -> tuple[WorldState, BeliefStore]:
    ents = [Entity.make("pool", Kind.PLACE, "湖畔"), Entity.make("cave", Kind.PLACE, "石洞"),
            Entity.make("top", Kind.PLACE, "崖顶"), Entity.make("stair", Kind.DOOR, "石阶"),
            Entity.make("crack", Kind.DOOR, "石缝", hidden=True), Entity.make("me", Kind.PERSON, "我")]
    rels = [Relation("stair", Rel.CONNECTS, "pool"), Relation("stair", Rel.CONNECTS, "top"),
            Relation("crack", Rel.CONNECTS, "pool"), Relation("crack", Rel.CONNECTS, "cave"),
            Relation("me", Rel.AT, "pool")]
    s = WorldState.build(1, clock, ents, rels)
    from tianlong.kernel.perception import scene_percept
    # 听人说起过“石洞”这个地方（只闻其名），但从不知道石缝
    told = Percept(clock - 1, Modality.SPEECH, sketches=(EntitySketch("cave", Kind.PLACE, "石洞", seen=False),))
    store = BeliefStore("me").revise_all([told, scene_percept(s, "me")])[0]
    return s, store


def test_p06_known_place_name_does_not_open_a_secret_passage():
    s, store = _grotto()
    assert store.knows("cave") and not store.knows("crack")
    assert not any(c.op == Op.MOVE and c.target == "cave" for c in candidates(store)), "不知道路，就想不到走过去"
    from tianlong.language.parser import rule_parse
    assert rule_parse("去石洞", store).candidate is None
    r = Kernel().step(s, [make_intent("me", Op.MOVE, "cave", "stair", based_on=0)])
    assert r.events[0].reason == "route_mismatch" and r.state.target("me", Rel.AT) == "pool", "内核不替角色挑暗门"
    r = Kernel().step(s, [make_intent("me", Op.INSPECT, "pool", based_on=0)])
    store = store.revise_all(o.percept for o in r.observations if o.observer == "me")[0]
    assert any(c.op == Op.MOVE and c.target == "cave" and c.obj == "crack" for c in candidates(store)), \
        "仔细查看发现石缝之后，才有这条路"


def test_p07_stale_route_can_be_tried_and_fails():
    sc = build_warehouse()
    s = sc.state
    player = BeliefStore("player").revise_all(sc.priors["player"])[0]
    # 玩家曾看见仓库大门开着能走；后来有人把它锁上了，玩家不知道
    s = WorldState.build(s.seed, s.clock, [s.entity("door_main").with_attr("locked", True),
                                           *(e for e in s.entities.values() if e.id != "door_main")], s.relations)
    move = next(c for c in candidates(player) if c.op == Op.MOVE and c.target == "entrance")
    assert move.obj == "door_main", "按旧认知，这条路还在"
    r = Kernel().step(s, [move.to_intent("m", "player", 0)])
    assert r.events[0].reason == "door_locked"
    player = player.revise_all(o.percept for o in r.observations if o.observer == "player")[0]
    assert player.holds(Proposition.attr("door_main", "locked", True)), "试过才知道门锁了"


def test_p08_asking_about_an_unseen_item_does_not_reveal_it():
    """问起一件从没见过的东西：它就算藏在这屋里、揣在对面那人身上，问话的人也不会因此“看见”它的样子（评审回归）。"""
    from tianlong.core import Entity, Relation, WorldState
    from tianlong.core import at as clock_at
    for hidden_on_floor in (True, False):
        ents = [Entity.make("hall", Kind.PLACE, "大堂"), Entity.make("a", Kind.PERSON, "甲"),
                Entity.make("b", Kind.PERSON, "乙"),
                Entity.make("key", Kind.ITEM, "钥匙", small=True, weapon=True, hidden=True if hidden_on_floor else None)]
        rels = [Relation("a", Rel.AT, "hall"), Relation("b", Rel.AT, "hall"),
                Relation("key", Rel.AT, "hall" if hidden_on_floor else "b")]
        s = WorldState.build(1, clock_at(1, 9, 0), ents, rels)
        r = Kernel().step(s, [make_intent("a", Op.ASK, "b", topic=Fact(Proposition.rel("key", Rel.AT, None)),
                                          based_on=0)])
        a = BeliefStore("a").revise_all(o.percept for o in r.observations if o.observer == "a")[0]
        sk = a.sketch("key")
        assert sk is None or (not sk.seen and sk.attrs == ()), (hidden_on_floor, sk)
        node = next((n for n in belief_view(a, 999).nodes if n.id == "key"), None)
        assert node is None or not node.knows("weapon")
