"""
[INPUT]: 依赖 torch，ray.rllib 的 TorchRLModule / ValueFunctionAPI / Columns，learning/model 的 RelationalEncoder，learning/featurize 的维度常量
[OUTPUT]: 对外提供 CandidateScoringModule（RLlib 新 API 栈的策略/价值网络）、GraphPolicyNet（纯 torch 核心，供模仿学习与游戏内推理复用）
[POS]: learning/rl 的策略网络：定长观测 → 还原为稀疏批图 → 与动态模型同构的关系编码器 → 逐候选打分（指针式策略）。
       策略只在候选集上选，掩码外的空位 logit = -inf；价值头只看认知池化与自身目标
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

from __future__ import annotations

from typing import Any

import torch
from ray.rllib.core.columns import Columns
from ray.rllib.core.rl_module.apis.value_function_api import ValueFunctionAPI
from ray.rllib.core.rl_module.torch import TorchRLModule
from torch import Tensor, nn

from tianlong.learning.featurize import N_MANNERS, N_OPS
from tianlong.learning.model import RelationalEncoder
from tianlong.learning.rl.observation import GOAL_KINDS

NEG_INF = -1e9


class GraphPolicyNet(nn.Module):
    def __init__(self, hidden: int = 64, layers: int = 2) -> None:
        super().__init__()
        d = hidden
        self.encoder = RelationalEncoder(d, layers)
        self.focus = nn.Linear(2, d)                     # 目标物品/目标地点的标记注入节点表示
        self.op_emb = nn.Embedding(N_OPS, 16)
        self.manner_emb = nn.Embedding(N_MANNERS, 8)
        g = len(GOAL_KINDS)
        self.score = nn.Sequential(nn.Linear(16 + 8 + 2 * d + 2 + 2 * d + g, d), nn.GELU(), nn.Linear(d, 1))
        self.value = nn.Sequential(nn.Linear(2 * d + g, d), nn.GELU(), nn.Linear(d, 1))

    def encode(self, obs: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        """定长观测 → [B, N, d] 节点表示与 [B, 2d] 图池化。填充节点不参与消息传递与池化。"""
        x, node_mask = obs["x"], obs["node_mask"] > 0.5
        b, n, _ = x.shape
        ei = obs["edge_index"].long()                               # [B, 2, E]
        emask = obs["edge_mask"] > 0.5                              # [B, E]
        offset = (torch.arange(b, device=x.device) * n).view(b, 1, 1)
        flat_ei = (ei + offset).permute(1, 0, 2).reshape(2, -1)[:, emask.reshape(-1)]
        flat_ea = obs["edge_attr"].reshape(b * obs["edge_attr"].shape[1], -1)[emask.reshape(-1)]
        h = self.encoder(x.reshape(b * n, -1), flat_ei, flat_ea).view(b, n, -1)
        h = (h + self.focus(obs["focus"])) * node_mask.unsqueeze(-1)
        denom = node_mask.sum(1, keepdim=True).clamp(min=1)
        mean = h.sum(1) / denom
        maxed = h.masked_fill(~node_mask.unsqueeze(-1), NEG_INF).max(1).values
        maxed = torch.where(node_mask.any(1, keepdim=True), maxed, torch.zeros_like(maxed))
        return h, torch.cat([mean, maxed], -1)

    def forward(self, obs: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        h, pooled = self.encode(obs)
        b, a = obs["cand"].shape[:2]
        cand = obs["cand"].long()
        op, manner = cand[..., 0].clamp(min=0), cand[..., 1].clamp(min=0)

        def gather(col: Tensor) -> Tensor:
            idx = col.clamp(min=0)
            got = torch.gather(h, 1, idx.unsqueeze(-1).expand(-1, -1, h.size(-1)))
            return got * (col >= 0).unsqueeze(-1).float()

        feats = torch.cat([
            self.op_emb(op), self.manner_emb(manner), gather(cand[..., 2]), gather(cand[..., 3]),
            obs["cand_pred"], pooled.unsqueeze(1).expand(-1, a, -1), obs["goal"].unsqueeze(1).expand(-1, a, -1),
        ], -1)
        logits = self.score(feats).squeeze(-1)
        logits = logits.masked_fill(obs["action_mask"] < 0.5, NEG_INF)
        value = self.value(torch.cat([pooled, obs["goal"]], -1)).squeeze(-1)
        return logits, value


class CandidateScoringModule(TorchRLModule, ValueFunctionAPI):
    """model_config: {"hidden": 64, "layers": 2}"""

    def setup(self) -> None:
        cfg = self.model_config or {}
        self.net = GraphPolicyNet(int(cfg.get("hidden", 64)), int(cfg.get("layers", 2)))

    def _forward(self, batch: dict[str, Any], **kwargs) -> dict[str, Any]:
        logits, _ = self.net(batch[Columns.OBS])
        return {Columns.ACTION_DIST_INPUTS: logits}

    def compute_values(self, batch: dict[str, Any], embeddings: Any = None) -> Tensor:
        _, value = self.net(batch[Columns.OBS])
        return value
