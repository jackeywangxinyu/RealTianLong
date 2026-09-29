"""
[INPUT]: 依赖 cognition 的 BeliefStore / Candidate / navigation，core 的 Fact / Kind / Manner / Modality / Op / Proposition / Rel，
         core/profiles 的 Profile，agents/predictors 的 Prediction
[OUTPUT]: 对外提供 Situation / Choice / Policy 协议、PolicyKit（规则策略共享的“在候选集中挑选”与信念查询积木）、RECENT
[POS]: agents 的决策契约与策略工具箱：策略只能在候选集中选（Choice.index），一切判断来自信念与近期经历。
       ScriptedPolicy 与 MartialTactics 都建立在这些积木之上，保证脚本行为与 RL 面对的是同一套候选与同一份认知
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from tianlong.agents.predictors import Prediction
from tianlong.cognition import BeliefStore, Candidate
from tianlong.cognition.navigation import next_hop
from tianlong.core import Fact, Kind, Manner, Modality, Op, Proposition
from tianlong.core.profiles import Profile

RECENT = 5


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

    def _go_towards(self, sit: Situation, place: str | None, why: str) -> Choice | None:
        hop = next_hop(sit.beliefs, place) if place else None
        return self._pick(sit, why, Op.MOVE, hop) if hop else None

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
    def _cures(b: BeliefStore, item: str) -> str | None:
        sk = b.sketch(item)
        return dict(sk.attrs).get("cures") if sk else None  # type: ignore[return-value]

    @staticmethod
    def _name(b: BeliefStore, eid: str) -> str:
        sk = b.sketch(eid)
        return sk.name if sk else eid
