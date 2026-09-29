"""
[INPUT]: 依赖 cognition 的 BeliefStore / Candidate / BeliefReader / effective_confidence，core 的 Op / Rel / Proposition / Kind / Fact /
         Modality / Percept，core/goals 的 GoalRegistry，core/profiles 的 Profile
[OUTPUT]: 对外提供 Prediction（v2：成功率、有效新观察、目标进展、风险、不确定性）、PRED_FIELDS、OutcomePredictor 协议、
          HeuristicPredictor（基于信念的先验预测器）、imagine()（在自己的认知上假想一条短分支）、goal_potential()、STALE
[POS]: agents 的后果预测接口；回答“这个候选行动可能发生什么”，只看角色认知，保留不确定性。
       预期获知 ≠ 位移：确定地走到刚看过的地方几乎没有新观察，原地仔细翻查一处没翻过的地方却可能有；
       目标进展与风险来自“在自己的认知上假想行动得手后的样子”，再用同一套目标语义（core/goals + BeliefReader）求势能之差——
       假想分支只存在于这次预测里，从不写回认知，更不碰真相。预测不替代结算：它是孤立行动的主观估计（旁人同时行动不在其中）。
       learning 训练出的 GNN 预测器实现同一协议后即可替换，策略代码不动
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Protocol

from tianlong.cognition import BeliefStore, Candidate
from tianlong.cognition.goals import BeliefReader
from tianlong.core import Fact, Kind, Modality, Op, Percept, Proposition, Rel
from tianlong.core.goals import GoalRegistry
from tianlong.core.profiles import Profile

STALE = 20         # 多久没看过的地方又值得一看（分钟），与脚本策略的探索一致


@dataclass(frozen=True, slots=True)
class Prediction:
    success: float           # 行动成功的主观概率
    info_gain: float         # 预期有效新观察（0~1）：不含行动本身的直接效果（自己走到了哪、拿起了什么）
    note: str = ""
    progress: float = 0.0    # 预期的目标进展（主观势能的增加，0~1）
    risk: float = 0.0        # 预期的损失：目标势能下降或自己挨打受伤的可能（0~1）
    uncertainty: float = 0.0  # 成败的不确定性（二元熵，0~1）


# 进入策略观测的预测字段（0~1）：观测契约按它排列列，新增字段只需在这里登记
PRED_FIELDS = ("success", "info_gain", "progress", "risk", "uncertainty")


class OutcomePredictor(Protocol):
    def predict(
        self, store: BeliefStore, now: int, cands: Sequence[Candidate], interests: Iterable[str] = (),
        profile: Profile | None = None,
    ) -> list[Prediction]: ...


# ============================================================
#  假想分支：在自己的认知上折叠一条“如果得手”的自我感知，求目标势能
# ============================================================

_REGISTRY = GoalRegistry()


def imagine(store: BeliefStore, now: int, facts: Sequence[Fact]) -> BeliefStore:
    """只存在于这次预测里的假想认知：不写回、不留经历。"""
    if not facts:
        return store
    return store.revise(Percept(now, Modality.SELF, None, tuple(facts)))[0]


def goal_potential(store: BeliefStore, now: int, profile: Profile | None) -> float:
    """已激活目标的主观势能（按权重归一到 0~1）；没有目标就是 0。"""
    if profile is None:
        return 0.0
    active = [g for g in profile.goals if g.active(now)]
    total = sum(abs(g.weight) for g in active)
    if not total:
        return 0.0
    reader = BeliefReader(store)
    return sum(g.weight * _REGISTRY.potential(reader, store.owner, g, profile.allies) for g in active) / total


def _direct_effects(store: BeliefStore, c: Candidate) -> list[Fact]:
    """候选得手时的直接效果（按自己的认知写成事实）。"""
    me = store.owner
    if c.op == Op.MOVE and c.target:
        return [Fact(Proposition.rel(me, Rel.AT, c.target))]
    if c.op == Op.TAKE and c.target:
        return [Fact(Proposition.rel(c.target, Rel.AT, me))]
    if c.op in (Op.PUT, Op.GIVE) and c.target and c.obj:
        return [Fact(Proposition.rel(c.obj, Rel.AT, c.target))]
    if c.op in (Op.UNLOCK, Op.LOCK) and c.target:
        return [Fact(Proposition.attr(c.target, "locked", True), c.op == Op.LOCK)]
    if c.op == Op.ATTACK and c.target:
        hurt = store.holds(Proposition.attr(c.target, "wounded", True))
        return [Fact(Proposition.attr(c.target, "subdued" if hurt else "wounded", True))]
    if c.op == Op.USE and c.target and c.obj:
        sk = store.sketch(c.obj)
        cure = dict(sk.attrs).get("cures") if sk else None
        return [Fact(Proposition.attr(c.target, str(cure), True), False)] if cure else []
    return []


def entropy(p: float) -> float:
    p = min(max(p, 1e-6), 1 - 1e-6)
    return -(p * math.log2(p) + (1 - p) * math.log2(1 - p))


def branch_value(store: BeliefStore, now: int, profile: Profile | None, facts: Sequence[Fact]) -> float:
    """假想分支的势能差（得手时）：正为进展，负为损失。"""
    if profile is None or not facts:
        return 0.0
    return goal_potential(imagine(store, now, facts), now, profile) - goal_potential(store, now, profile)


# ============================================================
#  启发式先验
# ============================================================


class HeuristicPredictor:
    """手写先验：门锁信念决定能否通过，位置可信度决定能否拿到；没看过/没翻过的地方才有新观察；
    进展与风险来自在自己认知上的假想分支。"""

    def predict(
        self, store: BeliefStore, now: int, cands: Sequence[Candidate], interests: Iterable[str] = (),
        profile: Profile | None = None,
    ) -> list[Prediction]:
        missing = [i for i in interests if store.location_of(i) is None]
        out = []
        for c in cands:
            p, info, note = self._one(store, now, c, missing)
            delta = branch_value(store, now, profile, _direct_effects(store, c))
            risk = max(0.0, -delta) * p
            if c.op == Op.ATTACK:
                risk = max(risk, 1.0 - p)        # 落空就可能招来还手
            out.append(Prediction(p, info, note, min(1.0, max(0.0, delta) * p), min(1.0, risk), entropy(p)))
        return out

    @staticmethod
    def _fresh(store: BeliefStore, now: int, place: str | None, table: str = "surveyed") -> bool:
        if place is None:
            return False
        last = getattr(store, table).get(place)
        return last is not None and now - last <= STALE

    def _one(self, store: BeliefStore, now: int, c: Candidate, missing: list[str]) -> tuple[float, float, str]:
        me = store.owner
        if c.op == Op.MOVE:
            door = c.obj
            locked = store.believed(Proposition.attr(door, "locked", True)) if door else None
            p = 0.8 if locked is None else (0.1 if locked.holds else 0.95)
            heard = any(ep.modality == Modality.SOUND and ep.event.place == c.target and now - ep.tick <= 5
                        for ep in store.episodes)         # 只算听到的响动：自己走过那里不是“那边有动静”
            if heard:
                return p, 0.7, "那边刚有动静"
            if self._fresh(store, now, c.target):
                return p, 0.05, "刚看过那边"
            return p, 0.6 if c.target not in store.surveyed else 0.4, ""
        if c.op == Op.TAKE:
            best = store.best(c.target, Rel.AT.value) if c.target else None
            return (0.9 * best.confidence if best else 0.2), 0.1, ""
        if c.op in (Op.PUT, Op.GIVE):
            return (0.95 if store.location_of(c.obj or "") == me else 0.1), 0.0, ""
        if c.op in (Op.UNLOCK, Op.LOCK):
            match = store.believed(Proposition.rel(c.obj or "", Rel.MATCHES, c.target or ""))
            return (0.9 if match and match.holds else 0.4), 0.3, ""
        if c.op == Op.INSPECT:
            sk = store.sketch(c.target or "")
            if sk is not None and sk.kind == Kind.PERSON:
                return 1.0, (0.6 if missing else 0.1), ("也许东西在他身上" if missing else "")
            if self._fresh(store, now, c.target, "searched"):
                return 1.0, 0.05, "刚翻过"
            return 1.0, (0.6 if missing else 0.2), ""
        if c.op == Op.ATTACK:
            helpless = store.holds(Proposition.attr(c.target or "", "subdued", True))
            return (1.0 if helpless else 0.5), 0.0, ("" if helpless else "胜负难料")
        if c.op == Op.ASK:
            return 0.95, 0.4, ""
        if c.op == Op.TELL:
            return 0.95, 0.0, ""
        return 1.0, 0.0, ""
