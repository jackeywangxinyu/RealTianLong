"""
[INPUT]: 依赖 numpy / torch，learning/rl 的 env / module / imitation 的 stack，core 的 Op / Kind / Rel / Outcome
[OUTPUT]: 对外提供 PolicyFn、net_policy()、random_policy()、scripted_policy()、wait_policy()、evaluate()、bootstrap_ci()
[POS]: learning/rl 的评测：留出种子上让不同策略面对同一批世界；目标达成读 GoalTracker（真相），行为计数只读事件——
       与奖励权重无关（改奖励不改计数）
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

from __future__ import annotations

import random
from collections.abc import Callable

import numpy as np
import torch

from tianlong.core import Kind, Op, Outcome, Rel
from tianlong.learning.rl.env import TianlongEnv
from tianlong.learning.rl.imitation import Obs, stack
from tianlong.learning.rl.module import GraphPolicyNet

PolicyFn = Callable[[TianlongEnv, dict[str, Obs]], dict[str, int]]


def net_policy(net: GraphPolicyNet, zero_predictions: bool = False) -> PolicyFn:
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


def evaluate(env: TianlongEnv, policy: PolicyFn, episodes: int, seed_base: int = 900_000) -> dict:
    returns, achieved, search_miss, attacks = [], [], 0, 0
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed_base + ep)
        total = {a: 0.0 for a in env.agents}
        for _ in range(env.horizon):
            before = env.state
            obs, rew, _, trunc, _ = env.step(policy(env, obs))
            for e in env.last_events:
                attacks += e.op == Op.ATTACK and e.outcome != Outcome.REJECTED
                if e.op == Op.INSPECT and e.outcome == Outcome.SUCCESS and e.intent.target \
                        and before.kind(e.intent.target) == Kind.PERSON:
                    wanted = {g.item for g in env.scenario.profiles[e.actor].goals if g.item}
                    search_miss += not set(before.sources(e.intent.target, Rel.AT)) & wanted
            for a, r in rew.items():
                total[a] += r
            if trunc["__all__"]:
                break
        returns += list(total.values())
        achieved += [env.achieved(a) for a in env.agents]
    return {"mean_return": round(float(np.mean(returns)), 4), "return_ci95": bootstrap_ci(returns),
            "goal_rate": round(float(np.mean(achieved)), 4), "goal_rate_ci95": bootstrap_ci(achieved),
            "search_misses_per_ep": round(search_miss / episodes, 3), "attacks_per_ep": round(attacks / episodes, 3)}


def bootstrap_ci(values: list, draws: int = 2000, seed: int = 0) -> list[float]:
    v = np.asarray(values, dtype=np.float64)
    means = np.random.default_rng(seed).choice(v, size=(draws, len(v)), replace=True).mean(axis=1)
    return [round(float(np.percentile(means, 2.5)), 4), round(float(np.percentile(means, 97.5)), 4)]
