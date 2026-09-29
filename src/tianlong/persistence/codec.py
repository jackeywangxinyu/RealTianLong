"""
[INPUT]: 依赖 core 的全部值对象，cognition 的 Belief / Episode / Obligation / Said，persistence/store 的 TurnEnvelope
[OUTPUT]: 对外提供 core 值对象 ⇄ JSON 兼容 dict 的显式编解码函数（intent / change / event / percept / fact / sketch / belief / episode /
          obligation / said / envelope 请求进度）
[POS]: persistence 的序列化边界；逐字段手写而非反射或 pickle——数据库里的内容不能决定构造哪个类，这是安全边界也是版本边界
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

from __future__ import annotations

from typing import Any

from tianlong.cognition import Belief, Episode, Obligation, Said
from tianlong.core import (
    AddRelation,
    Change,
    EntitySketch,
    Event,
    Fact,
    Intent,
    Kind,
    Manner,
    Modality,
    Op,
    Outcome,
    PerceivedEvent,
    Percept,
    Proposition,
    Rel,
    Relation,
    RemoveRelation,
    SetAttr,
    Social,
)
from tianlong.persistence.store import TurnEnvelope

J = dict[str, Any]


# ============================================================
#  命题与事实
# ============================================================


def prop_to(p: Proposition) -> J:
    return {"s": p.subject, "p": p.predicate, "v": p.value}


def prop_from(d: J) -> Proposition:
    return Proposition(d["s"], d["p"], d["v"])


def fact_to(f: Fact | None) -> J | None:
    return None if f is None else {"prop": prop_to(f.prop), "holds": f.holds}


def fact_from(d: J | None) -> Fact | None:
    return None if d is None else Fact(prop_from(d["prop"]), bool(d["holds"]))


# ============================================================
#  意图、变化、事件
# ============================================================


def _social(v: str | None) -> Social | None:
    return Social(v) if v else None       # 旧记录没有这个键：None


def intent_to(it: Intent) -> J:
    return {"id": it.id, "actor": it.actor, "op": it.op.value, "target": it.target, "obj": it.obj,
            "manner": it.manner.value, "topic": fact_to(it.topic), "based_on": it.based_on, "utterance": it.utterance,
            "social": it.social.value if it.social else None}


def intent_from(d: J) -> Intent:
    return Intent(d["id"], d["actor"], Op(d["op"]), d.get("target"), d.get("obj"), Manner(d["manner"]),
                  fact_from(d.get("topic")), int(d["based_on"]), d.get("utterance"), _social(d.get("social")))


def change_to(c: Change) -> J:
    if isinstance(c, SetAttr):
        return {"t": "set", "e": c.entity, "k": c.key, "old": c.old, "new": c.new}
    tag = "add" if isinstance(c, AddRelation) else "remove"
    return {"t": tag, "src": c.rel.src, "rel": c.rel.type.value, "dst": c.rel.dst}


def change_from(d: J) -> Change:
    if d["t"] == "set":
        return SetAttr(d["e"], d["k"], d["old"], d["new"])
    rel = Relation(d["src"], Rel(d["rel"]), d["dst"])
    return AddRelation(rel) if d["t"] == "add" else RemoveRelation(rel)


def event_to(e: Event) -> J:
    return {"id": e.id, "tick": e.tick, "intent": intent_to(e.intent), "place": e.place,
            "outcome": e.outcome.value, "reason": e.reason, "changes": [change_to(c) for c in e.changes]}


def event_from(d: J) -> Event:
    return Event(d["id"], int(d["tick"]), intent_from(d["intent"]), d.get("place"), Outcome(d["outcome"]),
                 d.get("reason"), tuple(change_from(c) for c in d["changes"]))


# ============================================================
#  感知
# ============================================================


def sketch_to(s: EntitySketch) -> J:
    return {"id": s.id, "kind": s.kind.value, "name": s.name, "attrs": [list(a) for a in s.attrs], "seen": s.seen}


def sketch_from(d: J) -> EntitySketch:
    return EntitySketch(d["id"], Kind(d["kind"]), d["name"], tuple((k, v) for k, v in d["attrs"]), bool(d.get("seen", True)))


def pevent_to(v: PerceivedEvent | None) -> J | None:
    if v is None:
        return None
    return {"kind": v.kind, "place": v.place, "actor": v.actor, "target": v.target, "obj": v.obj,
            "outcome": v.outcome.value if v.outcome else None, "topic": fact_to(v.topic),
            "reason": v.reason, "utterance": v.utterance, "social": v.social.value if v.social else None}


def pevent_from(d: J | None) -> PerceivedEvent | None:
    if d is None:
        return None
    return PerceivedEvent(d["kind"], d["place"], d.get("actor"), d.get("target"), d.get("obj"),
                          Outcome(d["outcome"]) if d.get("outcome") else None, fact_from(d.get("topic")),
                          d.get("reason"), d.get("utterance"), _social(d.get("social")))


def percept_to(p: Percept) -> J:
    return {"tick": p.tick, "modality": p.modality.value, "event": pevent_to(p.event),
            "facts": [fact_to(f) for f in p.facts], "scopes": list(p.scopes),
            "sketches": [sketch_to(s) for s in p.sketches], "informant": p.informant}


def percept_from(d: J) -> Percept:
    return Percept(int(d["tick"]), Modality(d["modality"]), pevent_from(d.get("event")),
                   tuple(fact_from(f) for f in d["facts"]), tuple(d["scopes"]),  # type: ignore[misc]
                   tuple(sketch_from(s) for s in d["sketches"]), d.get("informant"))


# ============================================================
#  认知
# ============================================================


def belief_to(b: Belief) -> J:
    return {"holds": b.holds, "confidence": b.confidence, "modality": b.modality.value,
            "learned_at": b.learned_at, "informant": b.informant}


def belief_from(prop: Proposition, d: J) -> Belief:
    return Belief(prop, bool(d["holds"]), float(d["confidence"]), Modality(d["modality"]),
                  int(d["learned_at"]), d.get("informant"))


def episode_to(e: Episode) -> J:
    return {"tick": e.tick, "modality": e.modality.value, "event": pevent_to(e.event), "informant": e.informant}


def episode_from(d: J) -> Episode:
    ev = pevent_from(d["event"])
    assert ev is not None
    return Episode(int(d["tick"]), Modality(d["modality"]), ev, d.get("informant"))


def obligation_to(o: Obligation) -> J:
    return {"kind": o.kind, "counterpart": o.counterpart, "topic": fact_to(o.topic), "since": o.since}


def obligation_from(d: J) -> Obligation:
    topic = fact_from(d["topic"])
    assert topic is not None
    return Obligation(d["kind"], d["counterpart"], topic, int(d["since"]))


def said_to(s: Said) -> J:
    return {"listener": s.listener, "fact": fact_to(s.fact), "tick": s.tick}


def said_from(d: J) -> Said:
    fact = fact_from(d["fact"])
    assert fact is not None
    return Said(d["listener"], fact, int(d["tick"]))


# ============================================================
#  请求进度：叙述文字不进这份 JSON，由 record_render 单独补写
# ============================================================


def envelope_to(e: TurnEnvelope) -> J:
    return {"request_id": e.request_id, "payload_hash": e.payload_hash, "intent": intent_to(e.intent),
            "planned_ticks": e.planned_ticks, "start_version": e.start_version, "start_clock": e.start_clock,
            "versions": list(e.versions), "ticks": list(e.ticks), "percepts": [percept_to(p) for p in e.percepts],
            "fresh": list(e.fresh), "done": e.done, "source": e.source,
            "followups": [intent_to(i) for i in e.followups], "reaction": e.reaction}


def envelope_from(d: J, narration: str | None = None) -> TurnEnvelope:
    return TurnEnvelope(d["request_id"], d["payload_hash"], intent_from(d["intent"]), int(d["planned_ticks"]),
                        int(d["start_version"]), int(d["start_clock"]), tuple(int(v) for v in d["versions"]),
                        tuple(int(t) for t in d["ticks"]), tuple(percept_from(p) for p in d["percepts"]),
                        tuple(d["fresh"]), bool(d["done"]), d.get("source", "rules"), narration,
                        tuple(intent_from(i) for i in d.get("followups", ())), bool(d.get("reaction", False)))
