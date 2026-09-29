"""
[INPUT]: 依赖 core 的 Op / Manner / Kind / Rel / Fact / Proposition / Intent / signature_error，cognition/beliefs 的 BeliefStore
[OUTPUT]: 对外提供 Candidate（结构化候选行动）、candidates()（从个人认知生成候选集）
[POS]: cognition 的行动空间；候选对象只来自角色的认知图——按角色“以为”的世界剪枝是合理的，按真实世界剪枝则是泄密。
       策略（脚本/RL）与预测器都在这个候选集上工作
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from tianlong.cognition.beliefs import BeliefStore
from tianlong.core import Fact, Intent, Kind, Manner, Op, Proposition, Rel
from tianlong.core.grammar import signature_error

# 截断优先级：身体行动在前，组合爆炸的言语在后——人多时截断只会丢掉一部分“说什么”，不会丢掉“动手”
_PRIORITY = (Op.WAIT, Op.MOVE, Op.ATTACK, Op.USE, Op.TAKE, Op.PUT, Op.GIVE, Op.UNLOCK, Op.LOCK, Op.INSPECT, Op.STUDY,
             Op.TELL, Op.ASK)
_OP_ORDER = {op: i for i, op in enumerate(_PRIORITY)}
assert set(_OP_ORDER) == set(Op), "新增操作必须在截断优先级中登记"
_WHILE_SUBDUED = frozenset({Op.WAIT, Op.TELL, Op.ASK})


@dataclass(frozen=True, slots=True)
class Candidate:
    op: Op
    target: str | None = None
    obj: str | None = None
    manner: Manner = Manner.NORMAL
    topic: Fact | None = None

    def to_intent(self, intent_id: str, actor: str, based_on: int, utterance: str | None = None) -> Intent:
        return Intent(intent_id, actor, self.op, self.target, self.obj, self.manner, self.topic, based_on, utterance)

    def sort_key(self) -> tuple:
        topic = self.topic.sort_key() if self.topic else ()
        return (_OP_ORDER[self.op], self.target or "", self.obj or "", self.manner.value, topic)

    @staticmethod
    def of(it: Intent) -> Candidate:
        return Candidate(it.op, it.target, it.obj, it.manner, it.topic)


def candidates(
    store: BeliefStore, interests: Iterable[str] | None = None, max_count: int | None = None
) -> tuple[Candidate, ...]:
    """角色此刻“想得到”的全部行动。interests 限定言语话题涉及的物品（默认：所有认识的物品）。"""
    me = store.owner
    here = store.location_of(me)

    def kind(eid: str) -> Kind | None:
        sk = store.sketch(eid)
        return sk.kind if sk else None

    def of_kind(k: Kind) -> list[str]:
        return sorted(e for e, sk in store.entities.items() if sk.kind == k and e != me)

    def is_here(eid: str) -> bool:
        loc = store.location_of(eid)
        if loc is None or here is None:
            return False
        return loc == here or (kind(loc) == Kind.SURFACE and store.location_of(loc) == here)

    places_here = [here] if here else []
    surfaces = [s for s in of_kind(Kind.SURFACE) if store.location_of(s) == here]
    persons = [p for p in of_kind(Kind.PERSON) if store.location_of(p) == here]
    held = [i for i in of_kind(Kind.ITEM) if store.location_of(i) == me]
    doors = [d for d in of_kind(Kind.DOOR) if here and store.holds(Proposition.rel(d, Rel.CONNECTS, here))]
    # 话题 = 关心的物品 + 认识的人（“钥匙在哪”“玩家在哪”都是值得说/问的）
    topics = sorted(set(interests if interests is not None else of_kind(Kind.ITEM)) | set(of_kind(Kind.PERSON)))

    out: list[Candidate] = [Candidate(Op.WAIT)]

    # ---- 移动：目的地 + 路线。经由认为存在的门（含自己发现的暗门），去认为相邻的地点；
    #      认为锁着也照样可以去推（记忆可能过时）；确知是单向且方向不对的才剪掉 ----
    for d in doors:
        oneway = next((b.prop.value for b in store.positives(d, "attr.oneway")), None)
        for b in store.positives(d, Rel.CONNECTS.value):
            if b.prop.value != here and (oneway is None or oneway == b.prop.value):
                out.append(Candidate(Op.MOVE, target=b.prop.value, obj=d))  # type: ignore[arg-type]
                out.append(Candidate(Op.MOVE, target=b.prop.value, obj=d, manner=Manner.CAREFUL))  # type: ignore[arg-type]

    # ---- 物件：拿认为在身边的（含认为已被制住者身上的），放/给/开锁用手里的 ----
    helpless = {p for p in persons if store.holds(Proposition.attr(p, "subdued", True))}
    for item in of_kind(Kind.ITEM):
        if item not in held and (is_here(item) or store.location_of(item) in helpless):
            out += [Candidate(Op.TAKE, item), Candidate(Op.TAKE, item, manner=Manner.CAREFUL)]
    for item in held:
        for dest in (*places_here, *surfaces):
            out += [Candidate(Op.PUT, dest, item), Candidate(Op.PUT, dest, item, Manner.CAREFUL)]
        out += [Candidate(Op.GIVE, p, item) for p in persons]
        for d in doors:
            if store.holds(Proposition.rel(item, Rel.MATCHES, d)) or not store.believed(
                Proposition.rel(item, Rel.MATCHES, d)
            ):
                # 认为配、或不知道配不配：都值得一试；确知不配才剪掉
                out += [Candidate(Op.UNLOCK, d, item), Candidate(Op.LOCK, d, item)]

    # ---- 查看 ----
    out += [Candidate(Op.INSPECT, t) for t in (*places_here, *surfaces, *persons)]

    # ---- 武斗、修习、施用：对在场者动手；研读手中之物；把手中之物用在在场者或自己身上 ----
    out += [Candidate(Op.ATTACK, p) for p in persons]
    out += [Candidate(Op.STUDY, i) for i in held]
    out += [Candidate(Op.USE, p, i) for i in held for p in (*persons, me)]

    # ---- 言语：说出自己相信的下落；询问不知下落的东西或人 ----
    for p in persons:
        for subject in topics:
            if subject == p:
                continue
            best = store.best(subject, Rel.AT.value)
            if best is not None:
                out.append(Candidate(Op.TELL, p, topic=Fact(best.prop, True)))
            # 以为知道也可以问：当面质问、求证，都是合理的言语行动
            out.append(Candidate(Op.ASK, p, topic=Fact(Proposition.rel(subject, Rel.AT, None), True)))

    if store.holds(Proposition.attr(me, "subdued", True)):
        out = [c for c in out if c.op in _WHILE_SUBDUED]   # 自知穴道被制：只剩开口与等待
    valid = [c for c in out if signature_error(c.op, kind, c.target, c.obj, c.topic) is None]
    ordered = [valid[0], *sorted(set(valid[1:]), key=Candidate.sort_key)]  # WAIT 永远在首位，截断时不丢
    return tuple(ordered[:max_count] if max_count else ordered)
