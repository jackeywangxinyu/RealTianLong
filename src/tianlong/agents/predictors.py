"""
[INPUT]: 依赖 cognition 的 BeliefStore / Candidate，core 的 Op / Rel / Proposition / Kind
[OUTPUT]: 对外提供 Prediction、OutcomePredictor 协议、HeuristicPredictor（基于信念的先验预测器）
[POS]: agents 的后果预测接口；回答“这个候选行动可能发生什么”，只看角色认知，保留不确定性。
       learning 训练出的 GNN 预测器实现同一协议后即可替换，策略代码不动
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Protocol

from tianlong.cognition import BeliefStore, Candidate
from tianlong.core import Kind, Op, Proposition, Rel


@dataclass(frozen=True, slots=True)
class Prediction:
    success: float     # 行动成功的主观概率
    info_gain: float   # 预期获得新信息的程度（0~1）
    note: str = ""


class OutcomePredictor(Protocol):
    def predict(
        self, store: BeliefStore, now: int, cands: Sequence[Candidate], interests: Iterable[str] = ()
    ) -> list[Prediction]: ...


class HeuristicPredictor:
    """手写先验：门锁信念决定能否通过，位置可信度决定能否拿到，未知的下落让查看更有价值。"""

    def predict(
        self, store: BeliefStore, now: int, cands: Sequence[Candidate], interests: Iterable[str] = ()
    ) -> list[Prediction]:
        missing = [i for i in interests if store.location_of(i) is None]
        return [self._one(store, now, c, missing) for c in cands]

    def _one(self, store: BeliefStore, now: int, c: Candidate, missing: list[str]) -> Prediction:
        me = store.owner
        if c.op == Op.MOVE:
            door = c.obj
            locked = store.believed(Proposition.attr(door, "locked", True)) if door else None
            p = 0.8 if locked is None else (0.1 if locked.holds else 0.95)
            heard = any(ep.event.place == c.target and now - ep.tick <= 5 for ep in store.episodes)
            return Prediction(p, 0.7 if heard else 0.3, "那边刚有动静" if heard else "")
        if c.op == Op.TAKE:
            best = store.best(c.target, Rel.AT.value) if c.target else None
            return Prediction(0.9 * best.confidence if best else 0.2, 0.1)
        if c.op in (Op.PUT, Op.GIVE):
            return Prediction(0.95 if store.location_of(c.obj or "") == me else 0.1, 0.0)
        if c.op in (Op.UNLOCK, Op.LOCK):
            match = store.believed(Proposition.rel(c.obj or "", Rel.MATCHES, c.target or ""))
            return Prediction(0.9 if match and match.holds else 0.4, 0.3)
        if c.op == Op.INSPECT:
            sk = store.sketch(c.target or "")
            if sk is not None and sk.kind == Kind.PERSON:
                return Prediction(1.0, 0.6 if missing else 0.1, "也许东西在他身上" if missing else "")
            return Prediction(1.0, 0.5 if missing else 0.1)
        if c.op == Op.ASK:
            return Prediction(0.95, 0.4)
        if c.op == Op.TELL:
            return Prediction(0.95, 0.0)
        return Prediction(1.0, 0.0)
