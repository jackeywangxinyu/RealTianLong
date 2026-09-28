"""
[INPUT]: 依赖 numpy / torch / torch_geometric 的 Data，learning/featurize 的 GraphTensors / featurize / encode_action / ActionCode，
         cognition 的 world_view / belief_view / BeliefStore / Candidate，core 的 WorldState / Rel
[OUTPUT]: 对外提供 Sample、env_sample()（环境动态样本）、agent_sample()（角色视角样本）、agent_query()（推理输入）、DynData、to_data()
[POS]: learning 的监督信号定义。两类样本刻意分开构造：
       环境样本 = 真实状态 + 行动 → 真实的下一状态（位置、属性、成败）；
       角色样本 = 个人认知 + 自己的行动 → 结算后“我会相信什么”（含“仍不知道”）。标签只来自内核实际执行的结果
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch_geometric.data import Data

from tianlong.cognition import BeliefStore, Candidate, belief_view, world_view
from tianlong.cognition.view import VIEW_ATTRS
from tianlong.core import Rel, WorldState
from tianlong.learning.featurize import (
    HOLDER_KINDS,
    LOCATED_KINDS,
    NODE_KINDS,
    ActionCode,
    GraphTensors,
    encode_action,
    featurize,
)

_A0 = len(NODE_KINDS)
_A1 = _A0 + len(VIEW_ATTRS)


@dataclass(frozen=True)
class Sample:
    graph: GraphTensors
    action: ActionCode
    success: float
    located: np.ndarray      # [M] 需要预测位置的节点
    holder_now: np.ndarray   # [M] 当前（认为的）容纳者下标，-1 = 未知
    holder_next: np.ndarray  # [M] 下一刻（认为的）容纳者下标，-1 = 未知
    attr_now: np.ndarray     # [N, A] 三态 -1/0/1
    attr_next: np.ndarray    # [N, A]
    view: str                # "env" | "agent"


def _located(g: GraphTensors) -> np.ndarray:
    return np.array([i for i, k in enumerate(g.kinds) if k in LOCATED_KINDS], dtype=np.int64)


def _holder_index(g: GraphTensors, eid: str | None) -> int:
    i = g.index_of(eid)
    return i if i >= 0 and g.kinds[i] in HOLDER_KINDS else -1


def env_sample(before: WorldState, after: WorldState, actor: str, cand: Candidate, success: bool) -> Sample:
    g = featurize(world_view(before, actor))
    g_next = featurize(world_view(after, actor))
    assert g.node_ids == g_next.node_ids, "实体集合在一步之内不变"
    located = _located(g)
    now = np.array([_holder_index(g, before.target(g.node_ids[i], Rel.AT)) for i in located], dtype=np.int64)
    nxt = np.array([_holder_index(g, after.target(g.node_ids[i], Rel.AT)) for i in located], dtype=np.int64)
    return Sample(g, encode_action(g, actor, cand), float(success), located, now, nxt,
                  g.x[:, _A0:_A1].copy(), g_next.x[:, _A0:_A1].copy(), "env")


def agent_query(store: BeliefStore, now: int, actor: str, cand: Candidate) -> Sample:
    """推理用样本：只有输入（认知 + 候选行动），标签位填“保持现状”，不参与训练。"""
    g = featurize(belief_view(store, now))
    located = _located(g)
    cur = np.array([_holder_index(g, store.location_of(g.node_ids[i])) for i in located], dtype=np.int64)
    attrs = g.x[:, _A0:_A1].copy()
    return Sample(g, encode_action(g, actor, cand), 0.0, located, cur, cur, attrs, attrs, "agent")


def agent_sample(
    before: BeliefStore, after: BeliefStore, now: int, actor: str, cand: Candidate, success: bool
) -> Sample:
    g = featurize(belief_view(before, now))
    g_next = featurize(belief_view(after, now + 1))
    next_attrs = {eid: g_next.x[i, _A0:_A1] for i, eid in enumerate(g_next.node_ids)}
    located = _located(g)
    cur = np.array([_holder_index(g, before.location_of(g.node_ids[i])) for i in located], dtype=np.int64)
    nxt = np.array([_holder_index(g, after.location_of(g.node_ids[i])) for i in located], dtype=np.int64)
    attr_next = np.stack([next_attrs.get(eid, g.x[i, _A0:_A1]) for i, eid in enumerate(g.node_ids)]) \
        if g.num_nodes else np.zeros((0, len(VIEW_ATTRS)), dtype=np.float32)
    return Sample(g, encode_action(g, actor, cand), float(success), located, cur, nxt,
                  g.x[:, _A0:_A1].copy(), attr_next.astype(np.float32), "agent")


# ============================================================
#  PyG 批处理：指针类字段随批次偏移，“未知”用独立掩码表示（-1 不能参与偏移）
# ============================================================

_INC_KEYS = frozenset({"located", "holder_now_idx", "holder_next_idx", "act_target", "act_obj", "act_actor"})


class DynData(Data):
    def __inc__(self, key, value, *args, **kwargs):
        if key in _INC_KEYS:
            return self.num_nodes
        return super().__inc__(key, value, *args, **kwargs)


def _ptr(values: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    v = torch.as_tensor(values, dtype=torch.long)
    return v.clamp(min=0), v < 0


def to_data(s: Sample) -> DynData:
    g, a = s.graph, s.action
    now_idx, now_null = _ptr(s.holder_now)
    next_idx, next_null = _ptr(s.holder_next)
    tgt, no_tgt = _ptr(np.array([a.target]))
    obj, no_obj = _ptr(np.array([a.obj]))
    act, no_act = _ptr(np.array([a.actor]))
    attr_now = torch.as_tensor(s.attr_now, dtype=torch.long) + 1     # -1/0/1 → 0/1/2
    attr_next = torch.as_tensor(s.attr_next, dtype=torch.long) + 1
    return DynData(
        x=torch.as_tensor(g.x), edge_index=torch.as_tensor(g.edge_index), edge_attr=torch.as_tensor(g.edge_attr),
        is_holder=torch.tensor([k in HOLDER_KINDS for k in g.kinds], dtype=torch.bool),
        act_op=torch.tensor([a.op]), act_manner=torch.tensor([a.manner]),
        act_target=tgt, act_has_target=~no_tgt, act_obj=obj, act_has_obj=~no_obj, act_actor=act, act_has_actor=~no_act,
        success=torch.tensor([s.success], dtype=torch.float32),
        located=torch.as_tensor(s.located, dtype=torch.long),
        holder_now_idx=now_idx, holder_now_null=now_null, holder_next_idx=next_idx, holder_next_null=next_null,
        attr_now=attr_now, attr_next=attr_next, attr_mask=(attr_now != 1) | (attr_next != 1),
        num_nodes=g.num_nodes,
    )
