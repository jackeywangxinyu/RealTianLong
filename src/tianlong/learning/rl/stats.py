"""
[INPUT]: 依赖 numpy（不依赖 torch / ray：结果 CLI 在任何装了学习层 extras 的机器上都能重做配对比较）
[OUTPUT]: 对外提供 METRICS（每项比率指标的分子/分母定义）、PER_EPISODE、MIN_WORLDS、EpisodeLog（一局 = 一个世界 = 一个聚类，
          可 JSON 往返）、cluster_ci()、ratio()、mean_return()、summarize()、compare()
[POS]: learning/rl 的统计口径（evaluation 负责跑局与数行为，这里只做推断）。同一局里的角色互相影响，不能当独立样本：
       区间一律以世界为单位重采样；策略之间只在同一批世界上配对比较（跨报告也行——每份报告带着逐世界记录）；
       “没发现显著差异”不等于“等效”，等效必须事先给出容许差并要求整个区间落在其中；世界太少（< MIN_WORLDS）不下结论。
       算不出来的数（分母为 0、区间退化）一律是 None，报告是严格 JSON，不出现 NaN
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np

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
    "unprovoked_attack_rate": ("attack_unprovoked", "attacks",
                               "无端动手：既非目标所驱（寻仇、守地、潜逃时对非盟友、为目标物品对认定的持有者），也非还手护人"),
    "false_claim_rate": ("false_claims", "claims", "误指控：说某物在某人身上而其实不在"),
    "invalid_loop_rate": ("loops", "agent_ticks", "重复上一 tick 因确定性原因失败的同一行动（招架/闪避后再出手不算）"),
    "unreachable_goal_share": ("goals_unreachable", "goals_with_target", "目标对象在真实地图上根本走不到（不计锁）"),
}
PER_EPISODE = ("searches", "search_miss", "search_no_evidence", "attacks", "attack_success", "attack_failure",
               "attack_rejected", "attack_goal_driven", "attack_retaliation", "attack_unprovoked", "false_claims",
               "claims", "loops", "retries_stochastic")


def _finite(v: float, digits: int = 4) -> float | None:
    return round(float(v), digits) if v is not None and math.isfinite(v) else None


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

    def to_dict(self) -> dict:
        return {"world": self.world, "agents": list(self.agents), "returns": dict(self.returns),
                "counts": {k: int(v) for k, v in sorted(self.counts.items()) if v}}

    @classmethod
    def from_dict(cls, d: dict) -> EpisodeLog:
        return cls(int(d["world"]), list(d["agents"]), dict(d["returns"]), {}, Counter(d["counts"]))


def cluster_ci(logs: Sequence[EpisodeLog], stat: Callable[[Sequence[EpisodeLog]], float], draws: int = 2000,
               seed: int = 0) -> list[float | None]:
    rng = np.random.default_rng(seed)
    n = len(logs)
    vals = [stat([logs[i] for i in rng.integers(0, n, n)]) for _ in range(draws)] if n else []
    vals = [v for v in vals if np.isfinite(v)]
    if not vals:
        return [None, None]
    return [_finite(np.percentile(vals, 2.5)), _finite(np.percentile(vals, 97.5))]


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
                 "mean_return": _finite(mean_return(logs)), "mean_return_ci95_world": cluster_ci(logs, mean_return, draws)}
    for name, (num, den, meaning) in METRICS.items():
        out[name] = {"value": _finite(ratio(name)(logs)), "ci95_world": cluster_ci(logs, ratio(name), draws),
                     "num": int(sum(lg.counts[num] for lg in logs)), "den": int(sum(lg.counts[den] for lg in logs)),
                     "definition": meaning}
    out["per_episode"] = {k: round(sum(lg.counts[k] for lg in logs) / max(len(logs), 1), 3) for k in PER_EPISODE}
    parts: Counter = Counter()
    for lg in logs:
        for p in lg.reward_parts.values():
            parts.update(p)
    out["reward_parts_per_agent"] = {k: round(v / max(out["agents"], 1), 4) for k, v in sorted(parts.items())}
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
    return {"metric": metric, "worlds": n, "diff": _finite(diff(pairs)), "ci95_world": [_finite(lo), _finite(hi)],
            "margin": margin, "verdict": verdict}
