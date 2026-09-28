"""
[INPUT]: 依赖 tianlong.cognition 的 BeliefStore / candidates / belief_view，tianlong.core 的感知类型
[OUTPUT]: 信念修正规则的单元测试：函数型槽位取代、传闻不覆盖亲见、矛盾说法并存、负证据、藏匿物豁免、候选集只依认知
[POS]: tests 的认知层；验证“相信不等于真实”被实现成了数据结构
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

from __future__ import annotations

from tianlong.cognition import BeliefStore, belief_view, candidates
from tianlong.core import EntitySketch, Fact, Kind, Modality, Op, Percept, Proposition, Rel

SK = (
    EntitySketch("me", Kind.PERSON, "我"), EntitySketch("liar", Kind.PERSON, "骗子"),
    EntitySketch("room", Kind.PLACE, "屋"), EntitySketch("hall", Kind.PLACE, "厅"),
    EntitySketch("table", Kind.SURFACE, "桌"), EntitySketch("key", Kind.ITEM, "钥匙", (("small", True),)),
)


def at(s, o, holds=True):
    return Fact(Proposition.rel(s, Rel.AT, o), holds)


def percept(modality, *facts, tick=0, scopes=(), informant=None):
    return Percept(tick, modality, None, tuple(facts), tuple(scopes), SK, informant)


def base():
    store, _ = BeliefStore("me").revise(percept(Modality.SCENE, at("me", "room"), at("table", "room"),
                                                at("key", "table"), scopes=("room", "table")))
    return store


def test_functional_slot_supersedes():
    s, changes = base().revise(percept(Modality.SIGHT, at("key", "liar"), tick=5))
    assert s.location_of("key") == "liar"
    assert s.believed(Proposition.rel("key", Rel.AT, "table")) is None
    assert any(c.before and c.before.prop.value == "table" and c.after is None for c in changes)


def test_hearsay_does_not_override_firsthand_but_coexists():
    s, _ = base().revise(percept(Modality.SPEECH, at("key", "hall"), tick=5, informant="liar"))
    claims = s.positives("key", "AT")
    assert [b.prop.value for b in claims] == ["table", "hall"], "亲见排前，传闻并存"
    assert claims[1].hearsay and claims[1].confidence < 1.0
    assert s.location_of("key") == "table"


def test_firsthand_evicts_weaker_hearsay():
    s, _ = BeliefStore("me").revise(percept(Modality.SPEECH, at("key", "hall"), informant="liar"))
    s, _ = s.revise(percept(Modality.SIGHT, at("key", "table"), tick=3))
    assert [b.prop.value for b in s.positives("key", "AT")] == ["table"]


def test_negative_evidence_from_scene():
    s, changes = base().revise(percept(Modality.SCENE, at("me", "room"), at("table", "room"), tick=9,
                                       scopes=("room", "table")))
    b = s.believed(Proposition.rel("key", Rel.AT, "table"))
    assert b is not None and not b.holds
    assert s.location_of("key") is None, "知道不在桌上 ≠ 知道在哪里"
    assert any(c.before and c.before.holds and c.after and not c.after.holds for c in changes)


def test_scene_does_not_refute_believed_hidden_item():
    s, _ = base().revise(percept(Modality.SELF, Fact(Proposition.attr("key", "hidden", True)), tick=2))
    s, _ = s.revise(percept(Modality.SCENE, at("me", "room"), at("table", "room"), tick=3, scopes=("room", "table")))
    assert s.location_of("key") == "table"
    # 但仔细查看（SELF + scopes）的“看清”可以否定它
    s, _ = s.revise(percept(Modality.SELF, tick=4, scopes=("table",)))
    assert s.location_of("key") is None


def test_candidates_depend_only_on_beliefs():
    s = base()
    ops = {(c.op, c.target) for c in candidates(s)}
    assert (Op.TAKE, "key") in ops, "以为钥匙在桌上，就会想去拿——哪怕实际上它已经不在了"
    s2, _ = s.revise(percept(Modality.SCENE, at("me", "room"), at("table", "room"), tick=9, scopes=("room", "table")))
    assert (Op.TAKE, "key") not in {(c.op, c.target) for c in candidates(s2)}
    assert candidates(s2)[0].op == Op.WAIT


def test_belief_view_keeps_negation_confidence_and_age():
    s, _ = base().revise(percept(Modality.SPEECH, at("key", "hall"), tick=5, informant="liar"))
    v = belief_view(s, now=10)
    hall = next(e for e in v.edges if e.src == "key" and e.dst == "hall")
    table = next(e for e in v.edges if e.src == "key" and e.dst == "table")
    assert hall.hearsay and hall.age == 5 and hall.confidence < table.confidence
    assert next(n for n in v.nodes if n.id == "me").is_self
