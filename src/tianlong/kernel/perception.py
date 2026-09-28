"""
[INPUT]: 依赖 core 的 WorldState / Event / Percept / Fact 等，kernel/space 的空间查询，kernel/resolution 的 Resolution
[OUTPUT]: 对外提供 Witnessing（一次事件的目击上下文）、scene_percept()、make_percept()、change_facts()、attr_fact()、sketches_for()、audibility()
[POS]: kernel 的感知物理：决定“谁以何种方式、获得事件的哪一部分”；规则通过 Witnessing 组合自己的感知方式，从而加新行动不改本模块
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

from __future__ import annotations

import random
from collections.abc import Iterable, Iterator
from dataclasses import dataclass

from tianlong.core import (
    OBSERVABLE_ATTRS,
    AddRelation,
    Change,
    EntitySketch,
    Event,
    Fact,
    Modality,
    PerceivedEvent,
    Percept,
    Proposition,
    Rel,
    RemoveRelation,
    SetAttr,
    WorldState,
    derive_seed,
)
from tianlong.kernel import space
from tianlong.kernel.resolution import Resolution

# ============================================================
#  声音传播参数
# ============================================================

HOP_ATTENUATION = 0.3   # 每隔一道门衰减
MAX_HEARING_HOPS = 1  # 声音只穿过一道门


def audibility(loudness: float, hop_count: int, alertness: float) -> float:
    """听见概率：响度随门衰减，警觉度（0~1，0.5 为常人）加减。"""
    return max(0.0, min(1.0, loudness - HOP_ATTENUATION * hop_count + (alertness - 0.5)))


# ============================================================
#  事实与草图的通用构造
# ============================================================


def change_facts(changes: Iterable[Change]) -> tuple[Fact, ...]:
    """把世界变化翻译为目击者能获知的事实。顺序保留：先负后正，便于信念修正覆盖。"""
    out: list[Fact] = []
    for c in changes:
        if isinstance(c, AddRelation):
            out.append(Fact(Proposition.of(c.rel), True))
        elif isinstance(c, RemoveRelation):
            out.append(Fact(Proposition.of(c.rel), False))
        elif isinstance(c, SetAttr):
            out.append(attr_fact(c.entity, c.key, c.old, c.new))
    return tuple(out)


def attr_fact(entity: str, key: str, old: object, new: object) -> Fact:
    """布尔属性统一规范为 (attr=True, 极性)：“开锁”与“推门发现没锁”必须落在同一个命题上。"""
    if isinstance(new, bool) or (new is None and isinstance(old, bool)):
        return Fact(Proposition.attr(entity, key, True), bool(new))
    if new is None:
        return Fact(Proposition.attr(entity, key, old), False)  # type: ignore[arg-type]
    return Fact(Proposition.attr(entity, key, new), True)  # type: ignore[arg-type]


def sketches_for(s: WorldState, ids: Iterable[str | None]) -> tuple[EntitySketch, ...]:
    """为感知中出现的实体生成外观草图——只含肉眼可见属性，不泄露锁状态等隐藏属性。"""
    seen: dict[str, EntitySketch] = {}
    for eid in ids:
        if eid is None or eid in seen or not s.has_entity(eid):
            continue
        e = s.entity(eid)
        attrs = tuple((k, v) for k, v in e.attrs if k in OBSERVABLE_ATTRS)
        seen[eid] = EntitySketch(e.id, e.kind, e.name, attrs)
    return tuple(seen[k] for k in sorted(seen))


def _referenced(s: WorldState, facts: Iterable[Fact]) -> Iterator[str]:
    for f in facts:
        yield f.prop.subject
        if not f.prop.is_attr and isinstance(f.prop.value, str) and s.has_entity(f.prop.value):
            yield f.prop.value


def _view_ids(view: PerceivedEvent | None) -> tuple[str | None, ...]:
    if view is None:
        return ()
    topic_ids: tuple[str | None, ...] = ()
    if view.topic is not None:
        v = view.topic.prop.value
        topic_ids = (view.topic.prop.subject, v if isinstance(v, str) else None)
    return (view.place, view.actor, view.target, view.obj, *topic_ids)


def make_percept(
    s: WorldState,
    modality: Modality,
    view: PerceivedEvent | None = None,
    facts: tuple[Fact, ...] = (),
    scopes: tuple[str, ...] = (),
    informant: str | None = None,
) -> Percept:
    ids = (*_view_ids(view), *_referenced(s, facts), *scopes, informant)
    return Percept(s.clock, modality, view, facts, scopes, sketches_for(s, ids), informant)


# ============================================================
#  Witnessing：一次事件的目击上下文
#  规则调用这些积木组合出“行动者 / 在场者 / 隔壁听者”各自得到什么
# ============================================================


@dataclass(frozen=True, slots=True)
class Witnessing:
    before: WorldState
    after: WorldState
    event: Event
    resolution: Resolution
    loudness: float

    @property
    def actor(self) -> str:
        return self.event.actor

    def full_view(self, with_topic: bool = True) -> PerceivedEvent:
        it = self.event.intent
        return PerceivedEvent(
            kind=it.op.value,
            place=self.event.place or "",
            actor=it.actor,
            target=it.target,
            obj=it.obj,
            outcome=self.event.outcome,
            topic=it.topic if with_topic else None,
            reason=self.event.reason,
            utterance=it.utterance if with_topic else None,
        )

    def actor_percept(self) -> tuple[str, Percept]:
        facts = (*self.resolution.learned, *change_facts(self.resolution.changes))
        p = make_percept(self.after, Modality.SELF, self.full_view(), facts, self.resolution.scopes)
        # 感知时刻是行动发生的 tick，而非结算后的新时钟
        return self.actor, _at_tick(p, self.event.tick)

    def witnesses(self, places: Iterable[str]) -> tuple[str, ...]:
        """这些地点里（行动前或行动后）在场、且不是行动者本人的人。"""
        found: set[str] = set()
        for place in places:
            found.update(space.persons_in(self.before, place))
            found.update(space.persons_in(self.after, place))
        found.discard(self.actor)
        return tuple(sorted(found))

    def sight(self, view: PerceivedEvent | None = None, facts: tuple[Fact, ...] | None = None) -> Percept:
        v = view or self.full_view()
        fs = change_facts(self.resolution.changes) if facts is None else facts
        return _at_tick(make_percept(self.after, Modality.SIGHT, v, fs), self.event.tick)

    def speech(self, facts: tuple[Fact, ...]) -> Percept:
        p = make_percept(self.after, Modality.SPEECH, self.full_view(), facts, informant=self.actor)
        return _at_tick(p, self.event.tick)

    def sounds(self, exclude: Iterable[str]) -> Iterator[tuple[str, Percept]]:
        """隔壁（经门）的人按概率听到“某处有响动”——不知道是谁、做了什么。"""
        if self.loudness <= 0 or not self.event.place:
            return
        skip = set(exclude) | {self.actor}
        dist = space.hops(self.before, self.event.place, MAX_HEARING_HOPS)
        for place in sorted(dist):
            if dist[place] == 0:
                continue
            for listener in space.persons_in(self.before, place):
                if listener in skip:
                    continue
                alertness = float(self.before.attr(listener, "alertness", 0.5))
                p = audibility(self.loudness, dist[place], alertness)
                roll = random.Random(derive_seed(self.before.seed, self.event.id, listener, "sound")).random()
                if roll < p:
                    view = PerceivedEvent(kind="noise", place=self.event.place)
                    yield listener, _at_tick(make_percept(self.before, Modality.SOUND, view), self.event.tick)

    def standard(self, witness_places: Iterable[str]) -> Iterator[tuple[str, Percept]]:
        """默认感知组合：行动者自知 + 在场者目击 + 隔壁听声。"""
        yield self.actor_percept()
        seers = self.witnesses(witness_places)
        for w in seers:
            yield w, self.sight()
        yield from self.sounds(exclude=seers)


def _at_tick(p: Percept, tick: int) -> Percept:
    if p.tick == tick:
        return p
    return Percept(tick, p.modality, p.event, p.facts, p.scopes, p.sketches, p.informant)


# ============================================================
#  环顾：每个 tick 结束时，每个人对所在地点的被动视觉
#  scopes = 地点 + 其中台面：这些容纳者“可见内容”被完整看清，
#  于是认知层可以据此推出“原本以为在这里的东西不见了”
# ============================================================


def scene_percept(s: WorldState, observer: str) -> Percept:
    place = space.place_of(s, observer)
    if place is None:
        return make_percept(s, Modality.SCENE)
    facts: list[Fact] = [Fact(Proposition.rel(observer, Rel.AT, place))]
    for e in space.visible_in(s, place, observer):
        holder = space.holder_of(s, e)
        if holder is not None:
            facts.append(Fact(Proposition.rel(e, Rel.AT, holder)))
    # 自己身上的东西自己清楚（包括藏在身上的小物件）
    facts.extend(Fact(Proposition.rel(i, Rel.AT, observer)) for i in space.contents(s, observer))
    for door, other in space.neighbors(s, place):
        facts.append(Fact(Proposition.rel(door, Rel.CONNECTS, place)))
        facts.append(Fact(Proposition.rel(door, Rel.CONNECTS, other)))
    scopes = (place, *space.surfaces_in(s, place))
    return make_percept(s, Modality.SCENE, None, tuple(facts), scopes)
