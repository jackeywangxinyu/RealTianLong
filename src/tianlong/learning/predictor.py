"""
[INPUT]: 依赖 torch / torch_geometric 的 Batch，learning/model 的 DynamicsModel，learning/samples 的 agent_query / to_data，
         agents/predictors 的 Prediction，cognition 的 BeliefStore / Candidate
[OUTPUT]: 对外提供 GNNPredictor（OutcomePredictor 协议的 GNN 实现）
[POS]: learning 与 agents 的接缝：训练好的角色视角动态模型以“预测器”身份接入 LangGraph 决策流程，
       替换 HeuristicPredictor 而不改策略与图。输入只有角色认知；输出保留概率——预测不是事实
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path

import torch
from torch_geometric.data import Batch

from tianlong.agents.predictors import Prediction
from tianlong.cognition import BeliefStore, Candidate
from tianlong.learning.model import DynamicsModel
from tianlong.learning.samples import agent_query, to_data


class GNNPredictor:
    def __init__(self, model: DynamicsModel) -> None:
        self.model = model.eval()

    @classmethod
    def load(cls, path: str | Path) -> GNNPredictor:
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        model = DynamicsModel(ckpt["config"]["hidden"])
        model.load_state_dict(ckpt["state_dict"])
        return cls(model)

    @torch.no_grad()
    def predict(
        self, store: BeliefStore, now: int, cands: Sequence[Candidate], interests: Iterable[str] = ()
    ) -> list[Prediction]:
        if not cands:
            return []
        samples = [agent_query(store, now, store.owner, c) for c in cands]
        batch = Batch.from_data_list([to_data(s) for s in samples])
        out = self.model(batch)
        success = torch.sigmoid(out.success)
        # 预期获知量：每个可定位节点“下一刻认为的位置不同于现在”的概率之和（关心的物品加倍），截断到 [0, 1]
        probs = torch.softmax(out.holder, dim=-1)
        p_change = 1.0 - probs.gather(1, out.holder_now.unsqueeze(1)).squeeze(1)
        focus = set(interests)
        weights = torch.tensor([2.0 if s.graph.node_ids[i] in focus else 1.0 for s in samples for i in s.located])
        owner = batch.batch[batch.located]
        gain = torch.zeros(len(cands)).index_add_(0, owner, p_change * weights).clamp(0, 1)
        return [Prediction(float(success[i]), float(gain[i]), "gnn") for i in range(len(cands))]
