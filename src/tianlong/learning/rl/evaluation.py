"""
[INPUT]: 依赖 numpy / torch，learning/rl 的 env / imitation 的 stack / observation 的 ABLATIONS / stats（统计推断），
         kernel/space 的 WorldReader / place_of，core 的 Event / Op / Kind / Rel / Outcome / Proposition / GoalMode / reason_key，
         cognition 的 BeliefStore
[OUTPUT]: 对外提供 PolicyFn、NetPolicy / net_policy()（可测试期消融）、RandomPolicy / random_policy()（无状态：随机数由种子、世界种子与局内步数派生）、
          scripted_policy()、wait_policy()（策略都可 pickle）、
          event_counts()（一个 tick 的行为计数）、repeat_kind()（无效循环与随机重掷之分）、run_episode()、
          evaluate()（workers > 1 时逐局分给子进程，与顺序评测逐项相同）；转出 stats 的 EpisodeLog / METRICS / MIN_WORLDS /
          cluster_ci / ratio / mean_return / summarize / compare
[POS]: learning/rl 的评测：跑局并按事件口径数行为（推断在 stats）。行为计数只读事件、真相与行动者事前的认知，与奖励权重无关：
       搜身分“落空”（对方身上没有自己关心的东西）与“无证据”（搜之前自己并不认为东西在他身上、也没见他拿过）；
       动手分得手/落空/被拒，并按缘由分“目标所驱”（寻仇、守地、潜逃时对非盟友、为护/取/送的物品对自己认定的持有者）
       “还手护人”“无端”；误指控 = 说某人身上有某物而其实没有；无效循环 = 重复上一 tick 因确定性原因失败的同一行动
       （被招架/闪避后再出手是重掷，另计 retries_stochastic）。目标分“开局即满足”“激活后新达成”“持续目标守住”
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

from __future__ import annotations

import random
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import torch

from tianlong.cognition import BeliefStore
from tianlong.core import Event, Kind, Op, Outcome, Proposition, Rel, WorldState, derive_seed, reason_key
from tianlong.core.goals import GoalMode
from tianlong.core.profiles import GoalKind, Profile
from tianlong.kernel import space
from tianlong.kernel.space import WorldReader
from tianlong.learning.rl.env import TianlongEnv, map_episodes
from tianlong.learning.rl.imitation import Obs, stack
from tianlong.learning.rl.observation import ABLATIONS
from tianlong.learning.rl.stats import (  # noqa: F401  （评测的调用方从这里一并取统计口径）
    METRICS,
    MIN_WORLDS,
    PER_EPISODE,
    EpisodeLog,
    cluster_ci,
    compare,
    mean_return,
    ratio,
    summarize,
)

PolicyFn = Callable[[TianlongEnv, dict[str, Obs]], dict[str, int]]
RETALIATION_WINDOW = 5
STOCHASTIC_REASONS = frozenset({"parried", "evaded"})   # 同样的出手下一 tick 可能得手：重试不是无效循环


# ============================================================
#  策略
# ============================================================


# 策略都是可 pickle 的对象或模块级函数：并行评测时整份交给子进程


@dataclass
class NetPolicy:
    """网络策略；ablate 为测试期消融（observation.ABLATIONS 的名字）——训练时看得见、评测时整列置零。"""
    net: object
    ablate: tuple[str, ...] = ()

    @torch.no_grad()
    def __call__(self, env: TianlongEnv, obs: dict[str, Obs]) -> dict[str, int]:
        batch = stack([obs[a] for a in env.agents])
        for k in (k for name in self.ablate for k in ABLATIONS[name]):
            batch[k] = torch.zeros_like(batch[k])
        logits, _ = self.net(batch)
        return {a: int(i) for a, i in zip(env.agents, logits.argmax(-1), strict=True)}


def net_policy(net, ablate: Iterable[str] = ()) -> PolicyFn:
    return NetPolicy(net, tuple(ablate))


@dataclass(frozen=True)
class RandomPolicy:
    """在合法候选里均匀随机。无状态：每一步的随机数只由（种子，世界种子，局内步数）派生——
    同一局无论单独评测、连着评测、复用同一个对象还是分给子进程，都走出同一条轨迹。"""
    seed: int = 0

    def __call__(self, env: TianlongEnv, obs: dict[str, Obs]) -> dict[str, int]:
        rng = random.Random(derive_seed("random_policy", self.seed, env.world_seed, env.t))
        return {a: int(rng.choice(list(np.flatnonzero(obs[a]["action_mask"])))) for a in env.agents}


def random_policy(seed: int = 0) -> PolicyFn:
    return RandomPolicy(seed)


def scripted_policy(env: TianlongEnv, obs: dict[str, Obs]) -> dict[str, int]:
    return env.expert_actions()


def wait_policy(env: TianlongEnv, obs: dict[str, Obs]) -> dict[str, int]:
    """永远等待：保持初态的基线——开局即满足的目标在这里也算“达成”，学得的策略必须比它好才算学会了什么。"""
    return {a: 0 for a in env.agents}


# ============================================================
#  事件口径：只读事件、真相与行动者事前的认知，不读奖励
# ============================================================


def _had_evidence(store: BeliefStore, target: str, wanted: set[str], now: int) -> bool:
    """搜之前：认为某件关心的东西在他身上，或近期亲眼见他拿过东西。"""
    if any(store.holds(Proposition.rel(i, Rel.AT, target)) for i in wanted):
        return True
    return any(ep.event.kind == Op.TAKE.value and ep.event.actor == target and now - ep.tick <= 30
               for ep in store.episodes)


def _attack_cause(profiles: Mapping[str, Profile], stores: Mapping[str, BeliefStore], e: Event,
                  history: Sequence[Event], before: WorldState) -> str:
    me, foe = e.actor, e.intent.target
    prof = profiles[me]
    for g in prof.goals:
        if not g.active(before.clock):
            continue
        if g.kind == GoalKind.HOSTILE and g.person == foe:
            return "goal_driven"
        if foe in prof.allies:
            continue
        if g.kind == GoalKind.GUARD and g.home == space.place_of(before, me):
            return "goal_driven"
        if g.kind == GoalKind.ESCAPE:
            return "goal_driven"
        # 为护/取/送的物品，对自己（事前）认定的持有者动手：制住再搜是目标所驱，不是无端
        if g.item and _had_evidence(stores[me], foe, {g.item}, before.clock):
            return "goal_driven"
    friends = {me, *prof.allies, *(g.person for g in prof.goals if g.kind == GoalKind.DEFEND and g.person)}
    if any(h.op == Op.ATTACK and h.actor == foe and h.intent.target in friends
           and before.clock - h.tick <= RETALIATION_WINDOW for h in history):
        return "retaliation"
    return "unprovoked"


def event_counts(profiles: Mapping[str, Profile], before: WorldState, stores: Mapping[str, BeliefStore],
                 events: Sequence[Event], history: Sequence[Event]) -> Counter:
    """一个 tick 的行为计数：只读事件、行动前的真相与行动者事前的认知。"""
    c: Counter = Counter()
    for e in events:
        if e.actor not in profiles:
            continue
        wanted = {g.item for g in profiles[e.actor].goals if g.item}
        target = e.intent.target
        if e.op == Op.INSPECT and e.outcome == Outcome.SUCCESS and target and before.kind(target) == Kind.PERSON:
            c["searches"] += 1
            c["search_miss"] += not set(before.sources(target, Rel.AT)) & wanted
            c["search_no_evidence"] += not _had_evidence(stores[e.actor], target, wanted, before.clock)
        elif e.op == Op.ATTACK:
            c["attacks"] += 1
            c[f"attack_{'success' if e.outcome == Outcome.SUCCESS else 'rejected' if e.outcome == Outcome.REJECTED else 'failure'}"] += 1
            if e.outcome != Outcome.REJECTED and target:
                c[f"attack_{_attack_cause(profiles, stores, e, history, before)}"] += 1
        elif e.op == Op.TELL and e.intent.topic is not None:
            p = e.intent.topic.prop
            if p.predicate == Rel.AT.value and e.intent.topic.holds and isinstance(p.value, str) \
                    and before.has_entity(p.value) and before.kind(p.value) == Kind.PERSON:
                c["claims"] += 1
                c["false_claims"] += before.target(p.subject, Rel.AT) != p.value
    return c


def repeat_kind(prev: Event | None, e: Event) -> str | None:
    """同一行动者紧接着重复上一 tick 失败的同一行动：确定性失败（锁着的门、学无可学）再试是无效循环 "loops"；
    被招架/闪避后再出手是重掷 "retries_stochastic"；其余 None。"""
    if e.op == Op.WAIT or prev is None or prev.outcome == Outcome.SUCCESS:
        return None
    if (prev.op, prev.intent.target, prev.intent.obj, prev.intent.topic) != \
            (e.op, e.intent.target, e.intent.obj, e.intent.topic):
        return None
    return "retries_stochastic" if reason_key(prev.reason) in STOCHASTIC_REASONS else "loops"


def _reachable(s: WorldState, a: str, b: str | None) -> bool | None:
    if b is None or not s.has_entity(b):
        return None
    pa, pb = space.place_of(s, a), space.place_of(s, b)
    return None if pa is None or pb is None else WorldReader(s).distance(pa, pb) is not None


# ============================================================
#  一局 = 一个世界 = 一个聚类
# ============================================================


def run_episode(env: TianlongEnv, policy: PolicyFn, seed: int) -> EpisodeLog:
    obs, _ = env.reset(seed=seed)
    log = EpisodeLog(seed, list(env.agents), {a: 0.0 for a in env.agents})
    for a, prof in env.scenario.profiles.items():
        for g in prof.goals:
            target = g.item or g.person or g.recipient or g.home
            r = _reachable(env.state, a, target)
            if r is not None:
                log.counts["goals_with_target"] += 1
                log.counts["goals_unreachable"] += not r
    history: list[Event] = []
    last: dict[str, Event] = {}
    for _ in range(env.horizon):
        before, stores = env.state, dict(env.stores)
        obs, rew, _, trunc, _ = env.step(policy(env, obs))
        events = [e for e in env.last_events]
        log.counts.update(event_counts(env.scenario.profiles, before, stores, events, history))
        for e in events:
            kind = repeat_kind(last.get(e.actor), e)
            if kind:
                log.counts[kind] += 1
            last[e.actor] = e
        log.counts["agent_ticks"] += len(env.agents)
        history += events
        for a, r in rew.items():
            log.returns[a] += r
        if trunc["__all__"]:
            break
    log.reward_parts = {a: dict(c) for a, c in env.episode_rewards.items()}
    log.counts["agents"] += len(env.agents)
    log.counts["achieved"] += sum(env.achieved(a) for a in env.agents)
    for recs in env.tracker.records.values():
        for rec in recs:
            if rec.activated_at is None:
                continue
            log.counts["goals_activated"] += 1
            log.counts["goals_initial"] += bool(rec.initial)
            if rec.mode == GoalMode.ACHIEVE:
                if not rec.initial:
                    log.counts["achieve_open"] += 1
                if rec.newly_achieved:
                    log.counts["achieve_new"] += 1
                    log.counts["time_to_goal_sum"] += rec.achieved_at - rec.activated_at  # type: ignore[operator]
            else:
                log.counts["maintain_goals"] += 1
                log.counts["maintain_kept"] += bool(rec.maintained)
    return log


def _eval_episode(env: TianlongEnv, policy: PolicyFn, seed: int) -> EpisodeLog:
    return run_episode(env, policy, seed)


def evaluate(env: TianlongEnv, policy: PolicyFn, episodes: int, seed_base: int = 900_000, draws: int = 2000,
             keep_logs: bool = False, workers: int = 1) -> dict:
    """workers > 1：各局分给子进程（每局只依赖自己的种子，结果与顺序评测逐项相同）；0 = 本机全部核。"""
    logs = map_episodes(env, _eval_episode, policy, [seed_base + ep for ep in range(episodes)], workers)
    out = summarize(logs, draws)
    if keep_logs:
        out["_logs"] = logs
    return out
