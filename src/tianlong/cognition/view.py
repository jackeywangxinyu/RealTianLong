"""
[INPUT]: 依赖 core 的 WorldState / Kind / Rel / Proposition，cognition/beliefs 的 BeliefStore / Episode
[OUTPUT]: 对外提供 ViewNode / ViewEdge / GraphView、world_view()（全知入口）、belief_view()（角色入口）、VIEW_ATTRS、EVENT_RELS
[POS]: cognition 的统一图投影；learning 的特征构造只接受 GraphView——角色入口在结构上拿不到 WorldState，
       因此“先隔离信息、再做消息传递”由类型边界保证，而不是靠调用方自觉
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

from __future__ import annotations

from dataclasses import dataclass

from tianlong.cognition.beliefs import BeliefStore
from tianlong.core import OBSERVABLE_ATTRS, Kind, Proposition, WorldState

# 进入特征的属性：1 = 真，-1 = 假，0 = 未知/不适用
VIEW_ATTRS = ("small", "locked", "hidden", "weapon", "wounded", "poisoned", "subdued")
# 事件节点与实体之间的视图关系（不是世界关系，只存在于认知投影中）
EVENT_RELS = ("OCCURRED_AT", "BY", "ON", "WITH")
EVENT_KIND = "event"


@dataclass(frozen=True, slots=True)
class ViewNode:
    id: str
    kind: str                                  # Kind 值，或 "event"
    attrs: tuple[tuple[str, float], ...] = ()  # 数值特征
    is_self: bool = False


@dataclass(frozen=True, slots=True)
class ViewEdge:
    src: str
    rel: str
    dst: str
    holds: bool = True
    confidence: float = 1.0
    age: int = 0
    hearsay: bool = False


@dataclass(frozen=True, slots=True)
class GraphView:
    owner: str | None   # None = 全知视角（仅供环境动态学习与评价）
    now: int
    nodes: tuple[ViewNode, ...]
    edges: tuple[ViewEdge, ...]

    def node_ids(self) -> tuple[str, ...]:
        return tuple(n.id for n in self.nodes)


def _tri(value: object) -> float:
    if value is None:
        return 0.0
    return 1.0 if value else -1.0


# ============================================================
#  全知入口：实际世界 → 视图（环境动态模型的输入）
# ============================================================


def world_view(s: WorldState, self_id: str | None = None) -> GraphView:
    nodes = tuple(
        ViewNode(
            e.id,
            e.kind.value,
            tuple((a, _tri(_true_attr(s, e.id, a)) if _applies(e.kind, a) else 0.0) for a in VIEW_ATTRS),
            e.id == self_id,
        )
        for e in sorted(s.entities.values(), key=lambda e: e.id)
    )
    edges = tuple(ViewEdge(r.src, r.type.value, r.dst) for r in s.sorted_relations())
    return GraphView(None, s.clock, nodes, edges)


_APPLIES = {
    (Kind.ITEM, "small"), (Kind.ITEM, "hidden"), (Kind.ITEM, "weapon"), (Kind.DOOR, "locked"), (Kind.DOOR, "hidden"),
    (Kind.PERSON, "wounded"), (Kind.PERSON, "poisoned"), (Kind.PERSON, "subdued"),
}


def _applies(kind: Kind, attr: str) -> bool:
    return (kind, attr) in _APPLIES


def _true_attr(s: WorldState, eid: str, attr: str) -> bool:
    if attr == "subdued":
        return int(s.attr(eid, "subdued_until", 0) or 0) > s.clock
    return bool(s.attr(eid, attr, False))


# ============================================================
#  角色入口：个人认知 → 视图（角色决策与角色视角预测的输入）
#  - 节点只有“认识的实体” + 近期经历
#  - 边带着可信度、时效与极性：过时、传闻、否定都保留给模型
# ============================================================


def belief_view(store: BeliefStore, now: int) -> GraphView:
    nodes: list[ViewNode] = []
    for eid in sorted(store.entities):
        sk = store.entities[eid]
        observable = dict(sk.attrs)
        attrs = []
        for a in VIEW_ATTRS:
            if not _applies(sk.kind, a):
                attrs.append((a, 0.0))
            elif a in OBSERVABLE_ATTRS:
                attrs.append((a, _tri(bool(observable.get(a, False)))))  # 看得见的属性：没看到即为否
            else:
                b = store.believed(Proposition.attr(eid, a, True))
                attrs.append((a, (1.0 if b.holds else -1.0) if b is not None else 0.0))
        nodes.append(ViewNode(eid, sk.kind.value, tuple(attrs), eid == store.owner))

    edges: list[ViewEdge] = []
    for b in store.sorted_beliefs():
        p = b.prop
        if p.is_attr or not (store.knows(p.subject) and isinstance(p.value, str) and store.knows(p.value)):
            continue
        edges.append(ViewEdge(p.subject, p.predicate, p.value, b.holds, b.confidence, now - b.learned_at, b.hearsay))

    for i, ep in enumerate(store.episodes):
        eid = f"episode:{i}"
        feats = ((f"op:{ep.event.kind}", 1.0), (f"modality:{ep.modality.value}", 1.0))
        nodes.append(ViewNode(eid, EVENT_KIND, feats))
        age = now - ep.tick
        for rel, other in zip(EVENT_RELS, (ep.event.place, ep.event.actor, ep.event.target, ep.event.obj), strict=True):
            if other and store.knows(other):
                edges.append(ViewEdge(eid, rel, other, True, 1.0, age, ep.informant is not None))

    return GraphView(store.owner, now, tuple(nodes), tuple(edges))
