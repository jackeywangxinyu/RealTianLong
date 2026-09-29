"""
[INPUT]: 依赖 core 的 WorldState / Event / Kind / Op / Outcome / Rel，core/goals 的 GoalRegistry / GoalMode，core/profiles 的 Goal / Profile，
         kernel/space 的 WorldReader（目标语义的真相读者）
[OUTPUT]: 对外提供 RewardWeights、RewardBreakdown、GoalRecord、GoalTracker（一局里每个目标的真实状态与任务奖励）、
          step_costs()（分项成本）、REWARD_VERSION
[POS]: learning/rl 的奖励定义。奖励是环境给训练的信号，允许读真相（角色看不到它）；分三层且各自命名：
       任务奖励 = 目标状态的跃迁（ACHIEVE：激活后第一次达成 +w；MAINTAIN：守住的被破坏 −w、恢复 +w——持续型目标按观察窗口判定，
       瞬时安全不宣告永久完成）；开局即已满足的不给奖励（区分“保持初态”与“学会达成”）；未激活的目标不求值、不算失败。
       塑形 = γ·Φ(s′,t+1) − Φ(s,t)，Φ 为已激活目标势能的加权和（势能只作塑形，不代替达成判定；环境只截断不终止，
       按随时间变化的势函数塑形保持策略不变）。成本 = 步长、失败、搜身落空，各自一项。
       所有目标求值都经 core/goals 的同一套语义；未注册的目标类型在构造 GoalTracker 时即报错
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field

from tianlong.core import Event, Kind, Op, Outcome, Rel, WorldState
from tianlong.core.goals import GoalMode, GoalRegistry
from tianlong.core.profiles import Goal, Profile
from tianlong.kernel.space import WorldReader

REWARD_VERSION = "reward-v2"


@dataclass(frozen=True)
class RewardWeights:
    step_cost: float = 0.01
    fail_cost: float = 0.02
    search_penalty: float = 0.5    # 搜身却一无所获（对方身上没有自己关心的物品）
    shaping: float = 1.0           # 塑形的总权重；0 = 只有任务奖励
    gamma: float = 0.97            # 与 PPO 的折扣一致，塑形才保持策略不变


@dataclass(frozen=True)
class RewardBreakdown:
    task: float = 0.0
    shaping: float = 0.0
    step: float = 0.0
    fail: float = 0.0
    search: float = 0.0

    @property
    def total(self) -> float:
        return self.task + self.shaping + self.step + self.fail + self.search

    def as_dict(self) -> dict[str, float]:
        return {**asdict(self), "total": self.total}


# ============================================================
#  目标跟踪：同一套目标语义，读真相
# ============================================================


@dataclass
class GoalRecord:
    goal: Goal
    mode: GoalMode
    activated_at: int | None = None     # 目标激活（not_before 已到）的时刻
    initial: bool | None = None         # 激活那一刻是否已满足：开局即满足的不算“学会达成”
    achieved_at: int | None = None      # ACHIEVE：激活后第一次满足的时刻（开局即满足者等于激活时刻）
    violated_at: int | None = None      # MAINTAIN：第一次被破坏的时刻
    satisfied: bool | None = None       # 最近一次求值
    active_ticks: int = 0
    held_ticks: int = 0                 # 激活后处于满足状态的 tick 数

    @property
    def newly_achieved(self) -> bool:
        return self.mode == GoalMode.ACHIEVE and self.achieved_at is not None and not self.initial

    @property
    def maintained(self) -> bool | None:
        """MAINTAIN：整个激活窗口里从未被破坏（且激活时已满足或随后建立）；ACHIEVE 返回 None。"""
        if self.mode != GoalMode.MAINTAIN or self.activated_at is None:
            return None
        return self.violated_at is None and bool(self.satisfied)


@dataclass
class GoalTracker:
    registry: GoalRegistry
    profiles: Mapping[str, Profile]
    records: dict[str, list[GoalRecord]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for a, p in sorted(self.profiles.items()):
            self.registry.check(p.goals)          # 未注册的目标类型：此刻就报错，绝不静默套用别的奖励
            self.records[a] = [GoalRecord(g, self.registry.mode(g)) for g in p.goals]

    def _sat(self, s: WorldState, agent: str, g: Goal) -> bool:
        return bool(self.registry.satisfied(WorldReader(s), agent, g, self.profiles[agent].allies))

    def start(self, s: WorldState) -> None:
        self.update(s)

    def update(self, s: WorldState) -> dict[str, float]:
        """按新状态推进每个目标的记录，返回每个角色本 tick 的任务奖励（只有跃迁才有）。"""
        out = {}
        for agent, recs in self.records.items():
            r = 0.0
            for rec in recs:
                g = rec.goal
                if not g.active(s.clock):
                    continue
                now = self._sat(s, agent, g)
                if rec.activated_at is None:
                    rec.activated_at, rec.initial, rec.satisfied = s.clock, now, now
                    if now and rec.mode == GoalMode.ACHIEVE:
                        rec.achieved_at = s.clock
                else:
                    if rec.mode == GoalMode.ACHIEVE:
                        if now and rec.achieved_at is None:
                            rec.achieved_at = s.clock
                            r += g.weight
                    else:
                        if rec.satisfied and not now:
                            r -= g.weight
                            if rec.violated_at is None:
                                rec.violated_at = s.clock
                        elif not rec.satisfied and now:
                            r += g.weight
                    rec.satisfied = now
                rec.active_ticks += 1
                rec.held_ticks += int(now)
            out[agent] = r
        return out

    def potential(self, s: WorldState, agent: str) -> float:
        reader, allies = WorldReader(s), self.profiles[agent].allies
        return sum(rec.goal.weight * self.registry.potential(reader, agent, rec.goal, allies)
                   for rec in self.records[agent] if rec.goal.active(s.clock))

    def achieved(self, agent: str) -> bool:
        """所有已激活目标此刻都满足（ACHIEVE 以“曾达成”为准，MAINTAIN 以窗口内从未被破坏且此刻满足为准）。"""
        recs = [r for r in self.records[agent] if r.activated_at is not None]
        return bool(recs) and all((r.achieved_at is not None) if r.mode == GoalMode.ACHIEVE else bool(r.maintained)
                                  for r in recs)


# ============================================================
#  成本
# ============================================================


def step_costs(before: WorldState, agent: str, goals: Sequence[Goal], events: Sequence[Event],
               w: RewardWeights) -> tuple[float, float, float]:
    """(步长, 失败, 搜身落空)，均为非正数。"""
    fail = search = 0.0
    wanted = {g.item for g in goals if g.item}
    for e in events:
        if e.actor != agent:
            continue
        if e.outcome != Outcome.SUCCESS and e.op != Op.WAIT:
            fail -= w.fail_cost
        if e.op == Op.INSPECT and e.outcome == Outcome.SUCCESS and e.intent.target \
                and before.kind(e.intent.target) == Kind.PERSON \
                and not set(before.sources(e.intent.target, Rel.AT)) & wanted:
            search -= w.search_penalty
    return -w.step_cost, fail, search


def step_reward(tracker: GoalTracker, before: WorldState, after: WorldState, events: Sequence[Event],
                w: RewardWeights | None = None) -> dict[str, RewardBreakdown]:
    """推进跟踪器并给出每个角色本 tick 的分项奖励。"""
    w = w or RewardWeights()
    phi0 = {a: tracker.potential(before, a) for a in tracker.records}
    task = tracker.update(after)
    out = {}
    for a, prof in tracker.profiles.items():
        shaping = w.shaping * (w.gamma * tracker.potential(after, a) - phi0[a])
        step, fail, search = step_costs(before, a, prof.goals, events, w)
        out[a] = RewardBreakdown(task.get(a, 0.0), shaping, step, fail, search)
    return out
