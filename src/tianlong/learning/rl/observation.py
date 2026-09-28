"""
[INPUT]: 依赖 numpy / gymnasium，learning/featurize 的 featurize / 维度常量，cognition 的 BeliefStore / Candidate / belief_view，
         agents/predictors 的 Prediction，core/profiles 的 Profile / GoalKind
[OUTPUT]: 对外提供 ObsSpec、observation_space()、encode_observation()
[POS]: learning/rl 的观测契约：把“个人认知图 + 候选集 + 冻结世界模型的预测 + 自身目标”填充成定长张量。
       RL 环境与游戏内的 LearnedPolicy 共用它——训练时看到什么，上线时就看到什么。
       掩码只屏蔽“候选集之外的空位”，从不按真相屏蔽行动
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import gymnasium as gym
import numpy as np

from tianlong.agents.predictors import Prediction
from tianlong.cognition import BeliefStore, Candidate, belief_view
from tianlong.core.profiles import GoalKind, Profile
from tianlong.learning.featurize import F_EDGE, F_NODE, MANNER_INDEX, OP_INDEX, featurize

GOAL_KINDS = tuple(GoalKind)


@dataclass(frozen=True)
class ObsSpec:
    max_nodes: int = 40
    max_edges: int = 256
    max_cands: int = 48


def observation_space(spec: ObsSpec) -> gym.spaces.Dict:
    n, e, a = spec.max_nodes, spec.max_edges, spec.max_cands
    f32 = np.float32
    return gym.spaces.Dict({
        "x": gym.spaces.Box(-1.0, 1.0, (n, F_NODE), f32),
        "node_mask": gym.spaces.Box(0.0, 1.0, (n,), f32),
        "focus": gym.spaces.Box(0.0, 1.0, (n, 2), f32),                    # 目标物品 / 目标地点或收件人
        "edge_index": gym.spaces.Box(0.0, float(n - 1), (2, e), f32),
        "edge_attr": gym.spaces.Box(0.0, 1.0, (e, F_EDGE), f32),
        "edge_mask": gym.spaces.Box(0.0, 1.0, (e,), f32),
        "goal": gym.spaces.Box(0.0, 1.0, (len(GOAL_KINDS),), f32),
        "cand": gym.spaces.Box(-1.0, float(max(n, 16)), (a, 4), f32),       # 操作、方式、目标下标、对象下标
        "cand_pred": gym.spaces.Box(0.0, 1.0, (a, 2), f32),                 # 冻结世界模型的预测：成功率、预期获知
        "action_mask": gym.spaces.Box(0.0, 1.0, (a,), f32),
    })


def encode_observation(
    store: BeliefStore, now: int, profile: Profile, cands: Sequence[Candidate], preds: Sequence[Prediction],
    spec: ObsSpec,
) -> dict[str, np.ndarray]:
    g = featurize(belief_view(store, now))
    n = min(g.num_nodes, spec.max_nodes)
    keep = {eid: i for i, eid in enumerate(g.node_ids[:n])}

    x = np.zeros((spec.max_nodes, F_NODE), np.float32)
    x[:n] = g.x[:n]
    node_mask = np.zeros(spec.max_nodes, np.float32)
    node_mask[:n] = 1.0

    ok = (g.edge_index[0] < n) & (g.edge_index[1] < n)
    ei, ea = g.edge_index[:, ok][:, : spec.max_edges], g.edge_attr[ok][: spec.max_edges]
    edge_index = np.zeros((2, spec.max_edges), np.float32)
    edge_attr = np.zeros((spec.max_edges, F_EDGE), np.float32)
    edge_mask = np.zeros(spec.max_edges, np.float32)
    m = ei.shape[1]
    edge_index[:, :m], edge_attr[:m], edge_mask[:m] = ei, ea, 1.0

    focus = np.zeros((spec.max_nodes, 2), np.float32)
    goal = np.zeros(len(GOAL_KINDS), np.float32)
    for gl in profile.goals:
        goal[GOAL_KINDS.index(gl.kind)] = 1.0
        if gl.item in keep:
            focus[keep[gl.item], 0] = 1.0
        for aux in (gl.home, gl.recipient):
            if aux in keep:
                focus[keep[aux], 1] = 1.0

    cand = np.full((spec.max_cands, 4), -1.0, np.float32)
    cand_pred = np.zeros((spec.max_cands, 2), np.float32)
    mask = np.zeros(spec.max_cands, np.float32)
    for i, (c, p) in enumerate(zip(cands[: spec.max_cands], preds, strict=False)):
        cand[i] = (OP_INDEX[c.op], MANNER_INDEX[c.manner], keep.get(c.target or "", -1), keep.get(c.obj or "", -1))
        cand_pred[i] = (p.success, p.info_gain)
        mask[i] = 1.0
    return {"x": x, "node_mask": node_mask, "focus": focus, "edge_index": edge_index, "edge_attr": edge_attr,
            "edge_mask": edge_mask, "goal": goal, "cand": cand, "cand_pred": cand_pred, "action_mask": mask}
