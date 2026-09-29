"""
[INPUT]: 依赖 core/profiles 的 Goal / GoalKind
[OUTPUT]: 对外提供 GoalReader 协议、GoalMode、GoalEvaluator、GoalRegistry、DEFAULT_GOALS、UnsupportedGoal、GOALS_VERSION
[POS]: core 的目标语义——每类目标“什么算达成、进展几何、是一次性达成还是持续维持”只在这里定义一次，
       再由两种读者求值：世界读者（kernel/space.WorldReader，奖励与评测用，读真相）与信念读者（cognition/goals.BeliefReader，
       角色观测与主观预测用，读认知，不知道就是 None）。同一套语义、两种视角——不会出现“奖励算的目标”与“角色以为的目标”各说各话。
       未注册的目标类型在使用时明确报错，绝不静默套用物品递送的兜底
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

from tianlong.core.profiles import Goal, GoalKind

GOALS_VERSION = "goals-v2"


class GoalReader(Protocol):
    """目标求值所需的全部只读查询。返回 None 表示“不知道”（世界读者从不返回 None）。"""

    def holder(self, eid: str) -> str | None: ...
    def place(self, eid: str) -> str | None: ...
    def owners(self, item: str) -> tuple[str, ...]: ...
    def status(self, person: str, status: str) -> bool | None: ...
    def persons_at(self, place: str) -> tuple[str, ...]: ...
    def distance(self, a: str, b: str) -> int | None: ...


class GoalMode(StrEnum):
    ACHIEVE = "achieve"     # 达成一次即算完成（拿到、交到、了结恩怨、抵达）
    MAINTAIN = "maintain"   # 持续约束：整个观察窗口里都要守住（守护物、守地、护人）


Tri = bool | None


@dataclass(frozen=True, slots=True)
class GoalEvaluator:
    kind: GoalKind
    mode: GoalMode
    satisfied: Callable[[GoalReader, str, Goal, tuple[str, ...]], Tri]      # 此刻是否满足（None = 不知道）
    potential: Callable[[GoalReader, str, Goal, tuple[str, ...]], float]    # 0~1 的进展势能（只作塑形，不代替达成）
    requires: tuple[str, ...]                                               # 必填字段


class UnsupportedGoal(ValueError):
    """目标类型没有注册评估器，或缺少必填字段。"""


def _co_located(r: GoalReader, a: str, b: str) -> bool:
    pa, pb = r.place(a), r.place(b)
    return pa is not None and pa == pb


# ============================================================
#  七类目标
# ============================================================


def _protect_sat(r: GoalReader, me: str, g: Goal, allies: tuple[str, ...]) -> Tri:
    h = r.holder(g.item)
    return None if h is None else (h == g.home or h in r.owners(g.item))


def _protect_pot(r: GoalReader, me: str, g: Goal, allies: tuple[str, ...]) -> float:
    if _protect_sat(r, me, g, allies):
        return 1.0
    return 0.5 if r.holder(g.item) == me else 0.0


def _acquire_sat(r: GoalReader, me: str, g: Goal, allies: tuple[str, ...]) -> Tri:
    return r.holder(g.item) == me          # 自己身上有没有，自己总是知道的


def _acquire_pot(r: GoalReader, me: str, g: Goal, allies: tuple[str, ...]) -> float:
    if r.holder(g.item) == me:
        return 1.0
    return 0.3 if _co_located(r, me, g.item) else 0.0


def _deliver_sat(r: GoalReader, me: str, g: Goal, allies: tuple[str, ...]) -> Tri:
    h = r.holder(g.item)
    return None if h is None else h == g.recipient


def _deliver_pot(r: GoalReader, me: str, g: Goal, allies: tuple[str, ...]) -> float:
    h = r.holder(g.item)
    if h == g.recipient:
        return 1.0
    if h == me:
        return 0.5 + (0.25 if _co_located(r, me, g.recipient) else 0.0)
    return 0.25 if _co_located(r, me, g.item) else 0.0


def _hostile_sat(r: GoalReader, me: str, g: Goal, allies: tuple[str, ...]) -> Tri:
    reached, subdued = r.status(g.person, g.until), r.status(g.person, "subdued")
    if reached or subdued:
        return True
    return None if reached is None else False


def _hostile_pot(r: GoalReader, me: str, g: Goal, allies: tuple[str, ...]) -> float:
    if _hostile_sat(r, me, g, allies):
        return 1.0
    partial = 0.5 if g.until == "subdued" and r.status(g.person, "wounded") else 0.0
    return max(partial, 0.25 if _co_located(r, me, g.person) else 0.0)


def _unharmed(r: GoalReader, person: str) -> Tri:
    states = [r.status(person, s) for s in ("wounded", "poisoned", "subdued")]
    if any(states):
        return False
    return None if any(s is None for s in states) else True


def _defend_sat(r: GoalReader, me: str, g: Goal, allies: tuple[str, ...]) -> Tri:
    return _unharmed(r, g.person)


def _defend_pot(r: GoalReader, me: str, g: Goal, allies: tuple[str, ...]) -> float:
    return 0.5 * float(bool(_unharmed(r, g.person))) + 0.5 * float(_co_located(r, me, g.person))


def _intruders(r: GoalReader, me: str, home: str, allies: tuple[str, ...]) -> list[str]:
    return [p for p in r.persons_at(home) if p != me and p not in allies and not r.status(p, "subdued")]


def _guard_sat(r: GoalReader, me: str, g: Goal, allies: tuple[str, ...]) -> Tri:
    here = r.place(me)
    return None if here is None else (here == g.home and not _intruders(r, me, g.home, allies))


def _guard_pot(r: GoalReader, me: str, g: Goal, allies: tuple[str, ...]) -> float:
    return 0.5 * float(r.place(me) == g.home) + 0.5 * float(not _intruders(r, me, g.home, allies))


def _escape_sat(r: GoalReader, me: str, g: Goal, allies: tuple[str, ...]) -> Tri:
    here = r.place(me)
    return None if here is None else here == g.home


def _escape_pot(r: GoalReader, me: str, g: Goal, allies: tuple[str, ...]) -> float:
    here = r.place(me)
    if here == g.home:
        return 1.0
    d = r.distance(here, g.home) if here is not None else None
    return 0.0 if d is None else 1.0 / (1.0 + d)


DEFAULT_GOALS: tuple[GoalEvaluator, ...] = (
    GoalEvaluator(GoalKind.PROTECT, GoalMode.MAINTAIN, _protect_sat, _protect_pot, ("item", "home")),
    GoalEvaluator(GoalKind.ACQUIRE, GoalMode.ACHIEVE, _acquire_sat, _acquire_pot, ("item",)),
    GoalEvaluator(GoalKind.DELIVER, GoalMode.ACHIEVE, _deliver_sat, _deliver_pot, ("item", "recipient")),
    GoalEvaluator(GoalKind.HOSTILE, GoalMode.ACHIEVE, _hostile_sat, _hostile_pot, ("person",)),
    GoalEvaluator(GoalKind.DEFEND, GoalMode.MAINTAIN, _defend_sat, _defend_pot, ("person",)),
    GoalEvaluator(GoalKind.GUARD, GoalMode.MAINTAIN, _guard_sat, _guard_pot, ("home",)),
    GoalEvaluator(GoalKind.ESCAPE, GoalMode.ACHIEVE, _escape_sat, _escape_pot, ("home",)),
)


class GoalRegistry:
    """启用的目标族 → 评估器。check() 在环境初始化时对每个目标求证：未注册或缺字段即报错。"""

    def __init__(self, evaluators: Iterable[GoalEvaluator] = DEFAULT_GOALS) -> None:
        self._by_kind: Mapping[GoalKind, GoalEvaluator] = {e.kind: e for e in evaluators}

    @property
    def kinds(self) -> tuple[GoalKind, ...]:
        return tuple(k for k in GoalKind if k in self._by_kind)

    def evaluator(self, g: Goal) -> GoalEvaluator:
        ev = self._by_kind.get(g.kind)
        if ev is None:
            raise UnsupportedGoal(f"目标类型 {g.kind} 没有注册评估器：不能静默套用其他目标的奖励")
        missing = [f for f in ev.requires if getattr(g, f) is None]
        if missing:
            raise UnsupportedGoal(f"{g.kind} 目标缺少字段 {missing}")
        return ev

    def check(self, goals: Iterable[Goal]) -> None:
        for g in goals:
            self.evaluator(g)

    def satisfied(self, r: GoalReader, me: str, g: Goal, allies: tuple[str, ...] = ()) -> Tri:
        return self.evaluator(g).satisfied(r, me, g, allies)

    def potential(self, r: GoalReader, me: str, g: Goal, allies: tuple[str, ...] = ()) -> float:
        return self.evaluator(g).potential(r, me, g, allies)

    def mode(self, g: Goal) -> GoalMode:
        return self.evaluator(g).mode
