"""
[INPUT]: 依赖 torch，learning/rl 的 env / module，core 的 derive_seed
[OUTPUT]: 对外提供 Demo（一条示范：观测、动作、示范者标签、世界）、collect_demos()、demo_seed() / DEMO_SEED_FLOOR、bc_weights()、
          weighted_batch_loss()、
          behavior_clone()（返回逐轮报告）、
          holdout_metrics()（留出世界上的分动作指标）
[POS]: learning/rl 的模仿学习。示范者绝大多数时刻在等待：不加权会学成“永远等待”而准确率照样很高。
       加权方案是“每条样本一个固定权重 + 固定归一化常数”：权重使等待样本在全体里的总份额恰为声明的 wait_share，
       批损失除以名义批大小而不是批内权重之和——于是一个 epoch 的总梯度贡献与批的类别组成无关，全等待的批不会被分母抵消，
       也没有 0/0。wait_share 只接受 [0, 1]；某一类不存在时份额全归另一类并如实报告。
       示范带结构化标签（合理等待/目标未到时辰/自以为已达成/不知道而卡住/没有可行候选/找不到动作/探索），逐轮报告各类样本数与损失份额；
       评估看留出世界的分动作精确率与召回率，而不是训练准确率
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

from __future__ import annotations

import math
import random
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from tianlong.core import derive_seed
from tianlong.learning.rl.env import TianlongEnv
from tianlong.learning.rl.module import GraphPolicyNet

Obs = dict[str, np.ndarray]
BATCH = 128


@dataclass(frozen=True)
class Demo:
    obs: Obs
    action: int
    tag: str          # 示范者的结构化标签：等待原因、"explore" 或 ""（按目标行动）
    world: int        # 所属世界（示范局序号）：留出评测按世界切分


def stack(obs_list: Sequence[Obs]) -> dict[str, torch.Tensor]:
    return {k: torch.as_tensor(np.stack([o[k] for o in obs_list])) for k in obs_list[0]}


DEMO_SEED_FLOOR = 10_000_000     # 示范与留出示范的世界种子都 ≥ 此值；评测种子必须落在它之下——三者不相交


def demo_seed(stream: str, seed: int, ep: int) -> int:
    return DEMO_SEED_FLOOR + derive_seed(stream, seed, ep) % 900_000_000


def collect_demos(env: TianlongEnv, episodes: int, seed: int, stream: str = "demo") -> list[Demo]:
    """stream 区分示范（"demo"）与留出示范（"bc_holdout"）：两段种子由各自的随机流派生，互不重叠，也不碰评测种子。"""
    demos = []
    for ep in range(episodes):
        obs, _ = env.reset(seed=demo_seed(stream, seed, ep))
        for _ in range(env.horizon):
            choices = env.expert_choices()
            demos += [Demo(obs[a], choices[a].index, choices[a].tag, ep) for a in env.agents]
            obs, _, _, trunc, _ = env.step({a: c.index for a, c in choices.items()})
            if trunc["__all__"]:
                break
    return demos


# ============================================================
#  加权：固定的逐样本权重，全体平均权重为 1
# ============================================================


def bc_weights(actions: Sequence[int], wait_share: float | None) -> tuple[float, float, float]:
    """返回 (等待样本权重, 非等待样本权重, 实际等待份额)。WAIT 永居候选首位（动作 0）。"""
    if wait_share is not None and (not math.isfinite(wait_share) or not 0.0 <= wait_share <= 1.0):
        raise ValueError(f"wait_share 必须在 [0, 1]，收到 {wait_share!r}")
    n = len(actions)
    n_wait = sum(a == 0 for a in actions)
    n_act = n - n_wait
    if n == 0:
        return 1.0, 1.0, 0.0
    if wait_share is None or n_wait == 0 or n_act == 0:
        return 1.0, 1.0, n_wait / n            # 按原比例；或只有一类时份额全归它
    return wait_share * n / n_wait, (1 - wait_share) * n / n_act, wait_share


def weighted_batch_loss(per: torch.Tensor, is_wait: torch.Tensor, w_wait: float, w_act: float,
                        batch: int) -> torch.Tensor:
    """Σ 逐样本损失 × 固定权重 ÷ 名义批大小：一个 epoch 的总和只取决于样本集合，与怎样分批无关。"""
    return (per * torch.where(is_wait, w_wait, w_act)).sum() / batch


def behavior_clone(net: GraphPolicyNet, demos: list[Demo], epochs: int, lr: float = 1e-3, seed: int = 0,
                   smoothing: float = 0.0, wait_share: float | None = None, batch: int = BATCH,
                   log=print) -> list[dict]:
    """交叉熵（可只在合法候选上标签平滑）× 固定逐样本权重 ÷ 名义批大小。返回逐轮报告。"""
    w_wait, w_act, share = bc_weights([d.action for d in demos], wait_share)
    opt = torch.optim.AdamW(net.parameters(), lr=lr)
    rng = random.Random(seed)
    order = list(demos)
    reports = []
    for epoch in range(epochs):
        rng.shuffle(order)
        c: Counter = Counter()
        for i in range(0, len(order), batch):
            chunk = order[i:i + batch]
            obs = stack([d.obs for d in chunk])
            target = torch.tensor([d.action for d in chunk])
            logits, _ = net(obs)
            if smoothing > 0:
                valid = obs["action_mask"]
                soft = valid * (smoothing / valid.sum(-1, keepdim=True).clamp(min=1))
                soft = soft.scatter_add(1, target.unsqueeze(1), torch.full_like(target, 1 - smoothing,
                                                                                 dtype=soft.dtype).unsqueeze(1))
                per = -(soft * torch.log_softmax(logits, -1).clamp(min=-1e4)).sum(-1)
            else:
                per = F.cross_entropy(logits, target, reduction="none")
            is_wait = target == 0
            w = torch.where(is_wait, w_wait, w_act)
            loss = weighted_batch_loss(per, is_wait, w_wait, w_act, batch)   # 固定归一化常数：与批内类别组成无关
            opt.zero_grad()
            loss.backward()
            opt.step()
            hit = logits.argmax(-1) == target
            c["loss"] += float(loss.detach()) * batch
            c["wait_loss"] += float((per * w)[is_wait].sum().detach())
            c["act_loss"] += float((per * w)[~is_wait].sum().detach())
            c["n"] += len(chunk)
            c["correct"] += int(hit.sum())
            c["act_n"] += int((~is_wait).sum())
            c["act_ok"] += int(hit[~is_wait].sum())
            c["all_wait_batches"] += int(is_wait.all())
            c["batches"] += 1
        weighted = c["wait_loss"] + c["act_loss"]
        rep = {"epoch": epoch + 1, "loss": c["loss"] / max(c["n"], 1), "acc": c["correct"] / max(c["n"], 1),
               "act_acc": c["act_ok"] / c["act_n"] if c["act_n"] else float("nan"), "act_n": c["act_n"],
               "declared_wait_share": share, "wait_loss_share": c["wait_loss"] / weighted if weighted else float("nan"),
               "all_wait_batches": c["all_wait_batches"], "batches": c["batches"]}
        reports.append(rep)
        log(f"[bc {epoch + 1}] loss={rep['loss']:.4f} acc={rep['acc']:.3f} act_acc={rep['act_acc']:.3f} "
            f"(非等待 {c['act_n']}) 等待权重份额={share:.2f} 等待损失份额={rep['wait_loss_share']:.2f}")
    return reports


# ============================================================
#  留出世界上的分动作指标：不以训练准确率论成败
# ============================================================


@torch.no_grad()
def holdout_metrics(net: GraphPolicyNet, demos: Sequence[Demo]) -> dict:
    if not demos:
        return {}
    c: Counter = Counter()
    for i in range(0, len(demos), 256):
        chunk = demos[i:i + 256]
        logits, _ = net(stack([d.obs for d in chunk]))
        pred = logits.argmax(-1).tolist()
        for d, p in zip(chunk, pred, strict=True):
            t_wait, p_wait = d.action == 0, p == 0
            c["n"] += 1
            c["wait_true"] += t_wait
            c["wait_pred"] += p_wait
            c["wait_tp"] += t_wait and p_wait
            c["act_true"] += not t_wait
            c["act_pred"] += not p_wait
            c["act_tp"] += (not t_wait) and (not p_wait)
            c["act_exact"] += (not t_wait) and p == d.action
            c[f"tag:{d.tag or 'goal'}"] += 1
            c[f"tag_ok:{d.tag or 'goal'}"] += p == d.action

    def r(a: str, b: str) -> float:
        return round(c[a] / c[b], 4) if c[b] else float("nan")

    tags = sorted(k[4:] for k in c if k.startswith("tag:"))
    return {"samples": c["n"], "wait_precision": r("wait_tp", "wait_pred"), "wait_recall": r("wait_tp", "wait_true"),
            "act_precision": r("act_tp", "act_pred"), "act_recall": r("act_tp", "act_true"),
            "act_exact_acc": r("act_exact", "act_true"), "non_wait_share": r("act_true", "n"),
            "by_expert_tag": {t: f"{c[f'tag_ok:{t}'] / c[f'tag:{t}']:.3f} (n={c[f'tag:{t}']})" for t in tags}}
