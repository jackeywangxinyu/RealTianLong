"""
[INPUT]: 依赖 torch，learning/rl 的 GraphPolicyNet / ObsSpec / build_observation，learning/schema 的 check_schema，agents/policies 的 Choice / Situation
[OUTPUT]: 对外提供 LearnedPolicy（Policy 协议的神经网络实现）
[POS]: learning/rl 与 agents 的接缝：训练好的策略以“策略”身份接入 LangGraph 决策图，替换 ScriptedPolicy 而不改图。
       游玩时运行的是训练好的网络，不在每次玩家输入后临时重新训练；加载时核对规格指纹，并按训练时的 ObsSpec 看世界
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from tianlong.agents.policies import Choice, Situation
from tianlong.learning.rl.module import GraphPolicyNet
from tianlong.learning.rl.observation import ObsSpec, build_observation
from tianlong.learning.schema import check_schema


class LearnedPolicy:
    def __init__(self, net: GraphPolicyNet, spec: ObsSpec | None = None) -> None:
        self.net = net.eval()
        self.spec = spec or ObsSpec()

    @classmethod
    def load(cls, path: str | Path) -> LearnedPolicy:
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        check_schema(ckpt, path, view="policy")
        net = GraphPolicyNet(ckpt["config"]["hidden"])
        net.load_state_dict(ckpt["state_dict"])
        return cls(net, ObsSpec(**ckpt["obs_spec"]) if "obs_spec" in ckpt else None)

    @torch.no_grad()
    def choose(self, sit: Situation) -> Choice:
        ob = build_observation(sit.beliefs, sit.now, sit.profile, sit.candidates, sit.predictions, self.spec, sit.memory)
        batch = {k: torch.as_tensor(np.expand_dims(v, 0)) for k, v in ob.obs.items()}
        logits, _ = self.net(batch)
        probs = torch.softmax(logits[0], -1)
        idx = int(probs.argmax())
        if idx >= len(ob.candidates):
            return Choice(0, "策略网络无有效选择，等待")
        # 观测里的动作编号对应裁剪后保留的候选，映射回决策图给的候选集
        return Choice(sit.candidates.index(ob.candidates[idx]), f"策略网络选择（置信 {float(probs[idx]):.2f}）")
