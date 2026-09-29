"""
[INPUT]: 依赖 cognition 的 BeliefStore / Candidate / navigation，core 的 Fact / Kind / Manner / Modality / Op / Proposition / Rel，
         core/profiles 的 Profile，agents/predictors 的 Prediction
[OUTPUT]: 对外提供 Situation / Choice（含结构化标签 tag）/ Policy 协议、PolicyKit（规则策略共享的“在候选集中挑选”、沿自己的地图带路
          （认为锁着的门先试着开、打不开就不去撞）、凭个人勘察记录探索、信念查询积木）、WAIT_REASONS、RECENT、STALE
[POS]: agents 的决策契约与策略工具箱：策略只能在候选集中选（Choice.index），一切判断来自信念与近期经历。
       探索只凭自己的地图与勘察记录（BeliefStore.surveyed/searched），从不读真相里的最短路或藏匿处。
       ScriptedPolicy 与 MartialTactics 都建立在这些积木之上，保证脚本行为与 RL 面对的是同一套候选与同一份认知
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from tianlong.agents.predictors import Prediction
from tianlong.cognition import BeliefStore, Candidate, effective_confidence
from tianlong.cognition.navigation import believed_distance, route_to
from tianlong.core import Fact, Kind, Manner, Modality, Op, Proposition, Rel
from tianlong.core.profiles import Profile

RECENT = 5
STALE = 20          # 多久没看过的地方值得再去看一眼（分钟）
LOCK_DOUBT = 0.3    # 对“门锁着”的把握低于此（记忆已旧）就再去推一推
# 示范者等待的结构化原因：模仿学习据此区分合理等待与卡住
WAIT_REASONS = ("idle", "goal_inactive", "goal_done", "stuck_unknown", "no_candidate", "expert_no_action")


@dataclass(frozen=True)
class Situation:
    agent: str
    profile: Profile
    beliefs: BeliefStore
    now: int
    candidates: tuple[Candidate, ...]
    predictions: tuple[Prediction, ...]
    memories: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class Choice:
    index: int        # 候选集下标：策略永远只能在候选集中选
    rationale: str
    tag: str = ""     # 结构化标签：等待时为 WAIT_REASONS 之一，探索时为 "explore"


class Policy(Protocol):
    def choose(self, situation: Situation) -> Choice: ...


class PolicyKit:
    # ------------------------------------------------------------
    #  在候选集中挑选
    # ------------------------------------------------------------

    @staticmethod
    def _pick(sit: Situation, why: str, op: Op, target: str | None = None, obj: str | None = None,
              manner: Manner | None = None, topic: Fact | None = None) -> Choice | None:
        for i, c in enumerate(sit.candidates):
            if c.op == op and c.target == target and (obj is None or c.obj == obj) \
                    and (manner is None or c.manner == manner) and (topic is None or c.topic == topic):
                return Choice(i, why)
        return None

    def _go_towards(self, sit: Situation, place: str | None, why: str, manner: Manner | None = None) -> Choice | None:
        """沿自己以为的地图走一步：目的地与路线（门）都来自认知。
        认为那扇门锁着：手里有（认为配或不知配不配的）钥匙就先开锁；没有就不去撞门——除非这份记忆已经很旧。"""
        hop = route_to(sit.beliefs, place) if place else None
        if not hop:
            return None
        nxt, door = hop
        b = sit.beliefs
        locked = b.believed(Proposition.attr(door, "locked", True))
        if locked is not None and locked.holds and effective_confidence(locked, sit.now) >= LOCK_DOUBT:
            for key in self._held_items(b):
                match = b.believed(Proposition.rel(key, Rel.MATCHES, door))
                if match is None or match.holds:
                    choice = self._pick(sit, f"门锁着，用{self._name(b, key)}开锁", Op.UNLOCK, door, key)
                    if choice is not None:
                        return choice
            return None
        return self._pick(sit, why, Op.MOVE, nxt, door, manner)

    def _explore(self, sit: Situation, what: str) -> Choice | None:
        """不知道要找的东西/人在哪：先把此处仔细翻一遍（找物件时），再去最近的、没看过或很久没看过的地方。
        一切依据是自己的地图与勘察记录——不知道的地方不在候选里，全知的最短路与藏匿处从不参与。"""
        b = sit.beliefs
        here = self._here(b)
        name = self._name(b, what)
        sk = b.sketch(what)
        if here and sk is not None and sk.kind == Kind.ITEM and here not in b.searched \
                and not self._did_recently(sit, Op.INSPECT, here):
            choice = self._pick(sit, f"{name}下落不明，先把这里仔细翻一遍", Op.INSPECT, here)
            if choice is not None:
                return Choice(choice.index, choice.rationale, "explore")
        if here is None:
            return None
        options = []
        for p, psk in sorted(b.entities.items()):
            last = b.surveyed.get(p)
            if psk.kind != Kind.PLACE or p == here or (last is not None and sit.now - last <= STALE):
                continue
            d = believed_distance(b, here, p)
            if d is not None:
                options.append((d, last if last is not None else -1, p))
        for _, _, p in sorted(options):
            choice = self._go_towards(sit, p, f"{name}下落不明，去{self._name(b, p)}找找")
            if choice is not None:
                return Choice(choice.index, choice.rationale, "explore")
        # 该看的地方都看过了（小物件可能揣在谁身上）：向眼前的人打听
        q = Fact(Proposition.rel(what, Rel.AT, None), True)
        for p in self._persons_here(b):
            if p != what and not self._did_recently(sit, Op.ASK, p):
                choice = self._pick(sit, f"打听{name}的下落", Op.ASK, p, topic=q)
                if choice is not None:
                    return Choice(choice.index, choice.rationale, "explore")
        return None

    # ------------------------------------------------------------
    #  近期经历
    # ------------------------------------------------------------

    @staticmethod
    def _did_recently(sit: Situation, op: Op, target: str) -> bool:
        return any(
            ep.modality == Modality.SELF and ep.event.kind == op.value and ep.event.target == target
            and sit.now - ep.tick <= RECENT
            for ep in sit.beliefs.episodes
        )

    @staticmethod
    def _said(sit: Situation, listener: str, fact: Fact) -> bool:
        """近期经历里是否已经对此人说过这句话（说过就不必追着再说）。"""
        return any(
            ep.modality == Modality.SELF and ep.event.kind == Op.TELL.value
            and ep.event.target == listener and ep.event.topic == fact
            for ep in sit.beliefs.episodes
        )

    @staticmethod
    def _attackers_of(sit: Situation, victim: str, window: int = 3) -> list[str]:
        """近期（亲眼所见或亲身所受）对 victim 动过手的人，新近者在前。"""
        seen: list[str] = []
        for ep in reversed(sit.beliefs.episodes):
            ev = ep.event
            if ev.kind == Op.ATTACK.value and ev.target == victim and ev.actor and sit.now - ep.tick <= window \
                    and ev.actor not in seen:
                seen.append(ev.actor)
        return seen

    # ------------------------------------------------------------
    #  信念查询
    # ------------------------------------------------------------

    @staticmethod
    def _status(b: BeliefStore, person: str, status: str) -> bool:
        return b.holds(Proposition.attr(person, status, True))

    @staticmethod
    def _here(b: BeliefStore) -> str | None:
        return b.location_of(b.owner)

    def _persons_here(self, b: BeliefStore) -> list[str]:
        here = self._here(b)
        return [p for p, sk in sorted(b.entities.items())
                if sk.kind == Kind.PERSON and p != b.owner and here is not None and b.location_of(p) == here]

    @staticmethod
    def _held_items(b: BeliefStore) -> list[str]:
        return [i for i, sk in sorted(b.entities.items()) if sk.kind == Kind.ITEM and b.location_of(i) == b.owner]

    @staticmethod
    def _cures(b: BeliefStore, item: str) -> str | None:
        sk = b.sketch(item)
        return dict(sk.attrs).get("cures") if sk else None  # type: ignore[return-value]

    @staticmethod
    def _name(b: BeliefStore, eid: str) -> str:
        sk = b.sketch(eid)
        return sk.name if sk else eid
