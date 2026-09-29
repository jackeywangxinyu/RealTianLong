"""
[INPUT]: 依赖 torch，learning/rl 的 GraphPolicyNet / ObsSpec / encode_observation，learning/featurize 的 check_vocab，agents/policies 的 Choice / Situation
[OUTPUT]: 对外提供 LearnedPolicy（Policy 协议的神经网络实现）
[POS]: learning/rl 与 agents 的接缝：训练好的策略以“策略”身份接入 LangGraph 决策图，替换 ScriptedPolicy 而不改图。
       游玩时运行的是训练好的网络，不在每次玩家输入后临时重新训练
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from tianlong.agents.policies import Choice, Situation
from tianlong.learning.featurize import check_vocab
from tianlong.learning.rl.module import GraphPolicyNet
from tianlong.learning.rl.observation import ObsSpec, encode_observation


class LearnedPolicy:
    def __init__(self, net: GraphPolicyNet, spec: ObsSpec | None = None) -> None:
        self.net = net.eval()
        self.spec = spec or ObsSpec()

    @classmethod
    def load(cls, path: str | Path) -> LearnedPolicy:
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        check_vocab(ckpt, path)
        net = GraphPolicyNet(ckpt["config"]["hidden"])
        net.load_state_dict(ckpt["state_dict"])
        return cls(net)

    @torch.no_grad()
    def choose(self, sit: Situation) -> Choice:
        obs = encode_observation(sit.beliefs, sit.now, sit.profile, sit.candidates, sit.predictions, self.spec)
        batch = {k: torch.as_tensor(np.expand_dims(v, 0)) for k, v in obs.items()}
        logits, _ = self.net(batch)
        probs = torch.softmax(logits[0], -1)
        idx = int(probs.argmax())
        if idx >= len(sit.candidates):
            return Choice(0, "策略网络无有效选择，等待")
        return Choice(idx, f"策略网络选择（置信 {float(probs[idx]):.2f}）")
