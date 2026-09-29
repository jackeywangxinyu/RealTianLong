"""
[INPUT]: 依赖 core 的 WorldState / Rel / Kind / Event / Op / Outcome，core/profiles 的 Goal / GoalKind，kernel/space 的 place_of
[OUTPUT]: 对外提供 potential()（目标进度势函数）、step_reward()（角色条件化奖励）、goal_achieved()
[POS]: learning/rl 的奖励定义。奖励是环境给训练的信号，允许读真相（角色看不到它）；
       采用势函数差分塑形——奖励“真实进展”，而不是奖励某种行为本身：守卫搜错人要付代价，否则会学会反复指控
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

from __future__ import annotations

from collections.abc import Sequence

from tianlong.core import Event, Kind, Op, Outcome, Rel, WorldState
from tianlong.core.profiles import Goal, GoalKind
from tianlong.kernel import space

STEP_COST = 0.01
FAIL_COST = 0.02
FALSE_ACCUSATION = 0.5    # 第二轮 0.2 太轻：探索一放开，搜身的期望收益就盖过了冤枉人的代价


def potential(s: WorldState, agent: str, g: Goal) -> float:
    holder = s.target(g.item, Rel.AT)
    near = space.place_of(s, g.item) == space.place_of(s, agent)
    if g.kind == GoalKind.PROTECT:
        owners = s.sources(g.item, Rel.OWNS)
        if holder == g.home or holder in owners:
            return 1.0
        return 0.5 if holder == agent else 0.0
    if g.kind == GoalKind.ACQUIRE:
        return 1.0 if holder == agent else (0.3 if near else 0.0)
    if g.recipient is not None and holder == g.recipient:
        return 1.0
    return 0.5 if holder == agent else (0.25 if near else 0.0)


def goal_achieved(s: WorldState, agent: str, g: Goal) -> bool:
    return potential(s, agent, g) >= 1.0


def step_reward(before: WorldState, after: WorldState, agent: str, goals: Sequence[Goal], events: Sequence[Event]) -> float:
    r = -STEP_COST
    for g in goals:
        r += g.weight * (potential(after, agent, g) - potential(before, agent, g))
    for e in events:
        if e.actor != agent:
            continue
        if e.outcome != Outcome.SUCCESS and e.op != Op.WAIT:
            r -= FAIL_COST
        if e.op == Op.INSPECT and e.intent.target and before.kind(e.intent.target) == Kind.PERSON:
            # 搜身却一无所获（对方身上没有自己关心的物品）= 冤枉人
            carried = set(before.sources(e.intent.target, Rel.AT))
            if not carried & {g.item for g in goals}:
                r -= FALSE_ACCUSATION
    return r
