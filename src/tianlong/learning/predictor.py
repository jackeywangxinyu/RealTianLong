"""
[INPUT]: 依赖 torch / torch_geometric 的 Batch，learning/model 的 DynamicsModel，learning/samples 的 agent_query / to_data，
         learning/schema 的 check_schema，agents/predictors 的 Prediction，cognition 的 BeliefStore / Candidate
[OUTPUT]: 对外提供 GNNPredictor（OutcomePredictor 协议的 GNN 实现）、GAIN_SCALE
[POS]: learning 与 agents 的接缝：训练好的角色视角动态模型以“预测器”身份接入 LangGraph 决策流程，
       替换 HeuristicPredictor 而不改策略与图。输入只有角色认知；输出保留概率——预测不是事实。
       加载时同时核对规格指纹与视角：全知（env）模型不能冒充角色的主观预测。
       预期获知来自“有效新观察数”头（扣除行动本身的直接效果），而不是位置变化概率之和——
       确定地走到已知的地方不算获知，原地仔细查看却可能有所发现
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
from tianlong.learning.schema import check_schema

GAIN_SCALE = 5.0    # 期望有效新观察数 → [0, 1] 的尺度：五条以上视为“收获很大”


class GNNPredictor:
    def __init__(self, model: DynamicsModel, temperature: float = 1.0) -> None:
        self.model = model.eval()
        self.temperature = temperature      # 校准世界上拟合的成败头温度

    @classmethod
    def load(cls, path: str | Path) -> GNNPredictor:
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        check_schema(ckpt, path, view="agent")
        model = DynamicsModel(ckpt["config"]["hidden"])
        model.load_state_dict(ckpt["state_dict"])
        return cls(model, float(ckpt.get("success_temperature", 1.0)))

    @torch.no_grad()
    def predict(
        self, store: BeliefStore, now: int, cands: Sequence[Candidate], interests: Iterable[str] = ()
    ) -> list[Prediction]:
        if not cands:
            return []
        samples = [agent_query(store, now, store.owner, c) for c in cands]
        out = self.model(Batch.from_data_list([to_data(s) for s in samples]))
        success = torch.sigmoid(out.success / self.temperature)
        gain = (out.obs_gain / GAIN_SCALE).clamp(0, 1)
        return [Prediction(float(success[i]), float(gain[i]), "gnn") for i in range(len(cands))]
