"""
[INPUT]: 依赖 numpy / torch，learning/rl 的 env / module / imitation 的 stack / rewards 的 GoalRecord，
         kernel/space 的 WorldReader / place_of / neighbors / passable，core 的 Event / Op / Kind / Rel / Outcome / Proposition / GoalMode，
         cognition 的 BeliefStore
[OUTPUT]: 对外提供 PolicyFn、net_policy()、random_policy()、scripted_policy()、wait_policy()、
          event_counts()（一个 tick 的行为计数）、EpisodeLog（一局 = 一个世界 = 一个聚类）、run_episode()、summarize()、evaluate()（聚合 + 按世界聚类自助法区间）、
          compare()（同一批世界上的配对差异 + 预先声明容许差的等效判定）、cluster_ci()、METRICS（每项指标的分子/分母定义）
[POS]: learning/rl 的评测与统计口径。行为计数只读事件与真相，与奖励权重无关（改奖励不改计数）：
       搜身分“落空”（对方身上没有自己关心的东西）与“无证据”（搜之前自己并不认为东西在他身上、也没见他拿过）；
       动手分得手/落空/被拒，并按缘由分“目标所驱”“还手护人”“无端”；误指控 = 说某人身上有某物而其实没有；
       无效循环 = 重复上一 tick 刚失败的同一行动。目标分“开局即满足”“激活后新达成”“持续目标守住”，另报达成用时与可达性。
       同一局里的角色互相影响，不能当独立样本：区间一律以世界为单位重采样；策略之间只在同一批世界上配对比较；
       “没发现显著差异”不等于“等效”，等效必须事先给出容许差并要求整个区间落在其中；世界太少（< MIN_WORLDS）不下结论
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

from __future__ import annotations

import random
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
import torch

from tianlong.cognition import BeliefStore
from tianlong.core import Event, Kind, Op, Outcome, Proposition, Rel, WorldState
from tianlong.core.goals import GoalMode
from tianlong.core.profiles import GoalKind, Profile
from tianlong.kernel import space
from tianlong.kernel.space import WorldReader
from tianlong.learning.rl.env import TianlongEnv
from tianlong.learning.rl.imitation import Obs, stack

PolicyFn = Callable[[TianlongEnv, dict[str, Obs]], dict[str, int]]
RETALIATION_WINDOW = 5
MIN_WORLDS = 20        # 少于这么多个世界，自助法区间会退化（例如全是 0），不下“等效/不同”的结论

# 每项比率指标：(分子键, 分母键, 含义)。报告里一律带上分母与样本数
METRICS: dict[str, tuple[str, str, str]] = {
    "goal_rate": ("achieved", "agents", "角色的全部已激活目标在局末都达成/守住"),
    "initial_goal_rate": ("goals_initial", "goals_activated", "激活那一刻就已满足的目标占比（不是学来的）"),
    "new_goal_achievement": ("achieve_new", "achieve_open", "一次性目标里，开局未满足、后来达成的占比"),
    "maintenance_success": ("maintain_kept", "maintain_goals", "持续目标在整个激活窗口里从未被破坏且局末满足"),
    "time_to_goal": ("time_to_goal_sum", "achieve_new", "新达成的一次性目标从激活到达成的平均 tick 数"),
    "search_miss_rate": ("search_miss", "searches", "搜身落空：对方身上没有搜者关心的物品"),
    "search_no_evidence_rate": ("search_no_evidence", "searches", "无证据搜身：搜之前既不认为东西在他身上、也没见他拿过"),
    "attack_success_rate": ("attack_success", "attacks", "动手得手"),
    "unprovoked_attack_rate": ("attack_unprovoked", "attacks", "无端动手：既非目标所驱，也非还手护人"),
    "false_claim_rate": ("false_claims", "claims", "误指控：说某物在某人身上而其实不在"),
    "invalid_loop_rate": ("loops", "agent_ticks", "重复上一 tick 刚失败的同一行动"),
    "unreachable_goal_share": ("goals_unreachable", "goals_with_target", "目标对象在真实地图上根本走不到（不计锁）"),
}
PER_EPISODE = ("searches", "search_miss", "search_no_evidence", "attacks", "attack_success", "attack_failure",
               "attack_rejected", "attack_goal_driven", "attack_retaliation", "attack_unprovoked", "false_claims",
               "claims", "loops")


# ============================================================
#  策略
# ============================================================


def net_policy(net, zero_predictions: bool = False) -> PolicyFn:
    @torch.no_grad()
    def act(env: TianlongEnv, obs: dict[str, Obs]) -> dict[str, int]:
        batch = stack([obs[a] for a in env.agents])
        if zero_predictions:
            batch["cand_pred"] = torch.zeros_like(batch["cand_pred"])
        logits, _ = net(batch)
        return {a: int(i) for a, i in zip(env.agents, logits.argmax(-1), strict=True)}

    return act


def random_policy(seed: int = 0) -> PolicyFn:
    rng = random.Random(seed)
    return lambda env, obs: {a: rng.choice(list(np.flatnonzero(obs[a]["action_mask"]))) for a in env.agents}


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


def _attack_cause(profiles: Mapping[str, Profile], e: Event, history: Sequence[Event], before: WorldState) -> str:
    me, foe = e.actor, e.intent.target
    prof = profiles[me]
    for g in prof.goals:
        if not g.active(before.clock):
            continue
        if g.kind == GoalKind.HOSTILE and g.person == foe:
            return "goal_driven"
        if g.kind == GoalKind.GUARD and g.home == space.place_of(before, me) and foe not in prof.allies:
            return "goal_driven"
        if g.kind == GoalKind.ESCAPE:
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
                c[f"attack_{_attack_cause(profiles, e, history, before)}"] += 1
        elif e.op == Op.TELL and e.intent.topic is not None:
            p = e.intent.topic.prop
            if p.predicate == Rel.AT.value and e.intent.topic.holds and isinstance(p.value, str) \
                    and before.has_entity(p.value) and before.kind(p.value) == Kind.PERSON:
                c["claims"] += 1
                c["false_claims"] += before.target(p.subject, Rel.AT) != p.value
    return c


def _reachable(s: WorldState, a: str, b: str | None) -> bool | None:
    if b is None or not s.has_entity(b):
        return None
    pa, pb = space.place_of(s, a), space.place_of(s, b)
    return None if pa is None or pb is None else WorldReader(s).distance(pa, pb) is not None


# ============================================================
#  一局 = 一个世界 = 一个聚类
# ============================================================


@dataclass
class EpisodeLog:
    world: int
    agents: list[str]
    returns: dict[str, float] = field(default_factory=dict)
    reward_parts: dict[str, dict[str, float]] = field(default_factory=dict)
    counts: Counter = field(default_factory=Counter)

    def value(self, metric: str) -> tuple[float, float]:
        """(分子, 分母)：比率指标在聚类层面按“分子之和 / 分母之和”汇总。"""
        num, den, _ = METRICS[metric]
        return float(self.counts[num]), float(self.counts[den])


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
            prev = last.get(e.actor)
            if e.op != Op.WAIT and prev is not None and prev.outcome != Outcome.SUCCESS and \
                    (prev.op, prev.intent.target, prev.intent.obj, prev.intent.topic) == \
                    (e.op, e.intent.target, e.intent.obj, e.intent.topic):
                log.counts["loops"] += 1
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


# ============================================================
#  统计：以世界为单位的自助法；配对比较；等效判定
# ============================================================


def cluster_ci(logs: Sequence[EpisodeLog], stat: Callable[[Sequence[EpisodeLog]], float], draws: int = 2000,
               seed: int = 0) -> list[float]:
    rng = np.random.default_rng(seed)
    n = len(logs)
    vals = [stat([logs[i] for i in rng.integers(0, n, n)]) for _ in range(draws)]
    vals = [v for v in vals if np.isfinite(v)]
    if not vals:
        return [float("nan"), float("nan")]
    return [round(float(np.percentile(vals, 2.5)), 4), round(float(np.percentile(vals, 97.5)), 4)]


def ratio(metric: str) -> Callable[[Sequence[EpisodeLog]], float]:
    def f(logs: Sequence[EpisodeLog]) -> float:
        num = sum(lg.value(metric)[0] for lg in logs)
        den = sum(lg.value(metric)[1] for lg in logs)
        return num / den if den else float("nan")

    return f


def mean_return(logs: Sequence[EpisodeLog]) -> float:
    vals = [np.mean(list(lg.returns.values())) for lg in logs if lg.returns]
    return float(np.mean(vals)) if vals else float("nan")


def summarize(logs: Sequence[EpisodeLog], draws: int = 2000) -> dict:
    out: dict = {"worlds": len(logs), "agents": int(sum(lg.counts["agents"] for lg in logs)),
                 "mean_return": round(mean_return(logs), 4), "mean_return_ci95_world": cluster_ci(logs, mean_return, draws)}
    for name, (num, den, meaning) in METRICS.items():
        v = ratio(name)(logs)
        out[name] = {"value": round(v, 4) if np.isfinite(v) else None,
                     "ci95_world": cluster_ci(logs, ratio(name), draws),
                     "num": int(sum(lg.counts[num] for lg in logs)), "den": int(sum(lg.counts[den] for lg in logs)),
                     "definition": meaning}
    out["per_episode"] = {k: round(sum(lg.counts[k] for lg in logs) / max(len(logs), 1), 3) for k in PER_EPISODE}
    parts = Counter()
    for lg in logs:
        for p in lg.reward_parts.values():
            parts.update(p)
    out["reward_parts_per_agent"] = {k: round(v / max(out["agents"], 1), 4) for k, v in sorted(parts.items())}
    return out


def evaluate(env: TianlongEnv, policy: PolicyFn, episodes: int, seed_base: int = 900_000, draws: int = 2000,
             keep_logs: bool = False) -> dict:
    logs = [run_episode(env, policy, seed_base + ep) for ep in range(episodes)]
    out = summarize(logs, draws)
    if keep_logs:
        out["_logs"] = logs
    return out


def compare(a: Sequence[EpisodeLog], b: Sequence[EpisodeLog], metric: str = "mean_return", margin: float | None = None,
            draws: int = 2000, seed: int = 0) -> dict:
    """同一批世界上的配对差异 a − b：按世界重采样。margin 给出时，整个区间落在 ±margin 内才判“等效”。"""
    by_world = {lg.world: lg for lg in b}
    pairs = [(x, by_world[x.world]) for x in a if x.world in by_world]
    if not pairs:
        raise ValueError("两组评测没有共同的世界：配对比较必须在同一批世界上进行")
    stat = mean_return if metric == "mean_return" else ratio(metric)

    def diff(ps: Sequence[tuple[EpisodeLog, EpisodeLog]]) -> float:
        return stat([p[0] for p in ps]) - stat([p[1] for p in ps])

    rng = np.random.default_rng(seed)
    n = len(pairs)
    vals = [diff([pairs[i] for i in rng.integers(0, n, n)]) for _ in range(draws)]
    vals = [v for v in vals if np.isfinite(v)]
    lo, hi = (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))) if vals else (float("nan"),) * 2
    if n < MIN_WORLDS:
        verdict = "insufficient_worlds"
    elif margin is not None and np.isfinite(lo) and -margin <= lo and hi <= margin:
        verdict = "equivalent_within_margin"
    elif np.isfinite(lo) and (lo > 0 or hi < 0):
        verdict = "different"
    else:
        verdict = "inconclusive"             # 没发现显著差异 ≠ 等效
    return {"metric": metric, "worlds": n, "diff": round(diff(pairs), 4), "ci95_world": [round(lo, 4), round(hi, 4)],
            "margin": margin, "verdict": verdict}
