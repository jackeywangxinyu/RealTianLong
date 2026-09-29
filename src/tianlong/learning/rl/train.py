"""
[INPUT]: 依赖 torch，ray.rllib 的 PPOConfig / RLModuleSpec / MultiRLModuleSpec，learning/rl 的 env / module / observation
[OUTPUT]: 对外提供 RLConfig、collect_demos()、behavior_clone()、evaluate()、train_ppo()、main()（python -m tianlong.learning.rl.train）
[POS]: learning/rl 的训练与验收流水线：模仿学习初始化（脚本策略示范）→ PPO（同一策略网络被所有角色共享参数，但各自观测各自的认知）→
       留出种子上对照 随机 / 脚本 / 模仿 / PPO，并做“去掉世界模型预测特征”的消融，回答“每个组件究竟增加了什么”；
       评测同时数“冤枉人”与“动手”两种手段；env_runners/gpus 让同一 CLI 在 Colab 上并行采样、GPU 学习
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

from __future__ import annotations

import argparse
import json
import random
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from tianlong.core import Op
from tianlong.learning.featurize import VOCAB
from tianlong.learning.rl.env import TianlongEnv
from tianlong.learning.rl.module import CandidateScoringModule, GraphPolicyNet
from tianlong.learning.rl.rewards import FALSE_ACCUSATION

Obs = dict[str, np.ndarray]
PolicyFn = Callable[[TianlongEnv, dict[str, Obs]], dict[str, int]]


@dataclass(frozen=True)
class RLConfig:
    horizon: int = 30
    demo_episodes: int = 120
    bc_epochs: int = 8
    ppo_iterations: int = 20
    train_batch: int = 1500
    lr: float = 3e-4
    hidden: int = 64
    eval_episodes: int = 40
    seed: int = 0
    predictor_path: str = ""
    entropy: float = 0.01          # PPO 熵正则：模仿学习后的策略很尖锐，探索不足时调高
    bc_smoothing: float = 0.0      # 模仿学习的标签平滑：避免初始策略过度确定、PPO 无从探索
    env_runners: int = 0           # 并行采样进程数；0 = 在驱动进程里采样（小机器），Colab 上可设 2~8
    gpus: float = 0.0              # 学习器 GPU 数（Colab 设 1）


def _stack(obs_list: list[Obs]) -> dict[str, torch.Tensor]:
    return {k: torch.as_tensor(np.stack([o[k] for o in obs_list])) for k in obs_list[0]}


# ============================================================
#  模仿学习：脚本策略的示范 → 候选集上的交叉熵
# ============================================================


def collect_demos(env: TianlongEnv, episodes: int, seed: int) -> list[tuple[Obs, int]]:
    demos = []
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed * 100_000 + ep)
        for _ in range(env.horizon):
            act = env.expert_actions()
            demos += [(obs[a], act[a]) for a in env.agents]
            obs, _, _, trunc, _ = env.step(act)
            if trunc["__all__"]:
                break
    return demos


def behavior_clone(net: GraphPolicyNet, demos: list[tuple[Obs, int]], epochs: int, lr: float = 1e-3,
                   seed: int = 0, smoothing: float = 0.0, log=print) -> None:
    opt = torch.optim.AdamW(net.parameters(), lr=lr)
    rng = random.Random(seed)
    for epoch in range(epochs):
        rng.shuffle(demos)
        total, correct, n = 0.0, 0, 0
        for i in range(0, len(demos), 128):
            chunk = demos[i:i + 128]
            batch = _stack([o for o, _ in chunk])
            target = torch.tensor([a for _, a in chunk])
            logits, _ = net(batch)
            if smoothing > 0:
                # 只在合法候选上平滑：掩码外的空位不分概率
                valid = batch["action_mask"]
                soft = valid * (smoothing / valid.sum(-1, keepdim=True).clamp(min=1))
                soft = soft.scatter_add(1, target.unsqueeze(1), torch.full_like(target, 1 - smoothing,
                                                                                 dtype=soft.dtype).unsqueeze(1))
                loss = -(soft * torch.log_softmax(logits, -1).clamp(min=-1e4)).sum(-1).mean()
            else:
                loss = F.cross_entropy(logits, target)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss) * len(chunk)
            correct += int((logits.argmax(-1) == target).sum())
            n += len(chunk)
        log(f"[bc {epoch + 1}] loss={total / n:.4f} acc={correct / n:.3f}")


# ============================================================
#  评测：留出种子，同一批世界上比较不同策略
# ============================================================


def net_policy(net: GraphPolicyNet, zero_predictions: bool = False) -> PolicyFn:
    @torch.no_grad()
    def act(env: TianlongEnv, obs: dict[str, Obs]) -> dict[str, int]:
        batch = _stack([obs[a] for a in env.agents])
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


def evaluate(env: TianlongEnv, policy: PolicyFn, episodes: int, seed_base: int = 900_000) -> dict[str, float]:
    """除了回报与达成率，还数两种“手段”：冤枉人（搜身落空）与动手——奖励里没有禁止动手，
    制住对方再搜走东西在规则上是可行的捷径，策略学没学会这条捷径必须如实报告。"""
    returns, achieved, accusations, attacks = [], [], 0, 0
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed_base + ep)
        total = {a: 0.0 for a in env.agents}
        for _ in range(env.horizon):
            obs, rew, _, trunc, _ = env.step(policy(env, obs))
            attacks += sum(e.op == Op.ATTACK for e in env.last_events)
            for a, r in rew.items():
                total[a] += r
                accusations += int(r <= -FALSE_ACCUSATION)
            if trunc["__all__"]:
                break
        returns += list(total.values())
        achieved += [env.achieved(a) for a in env.agents]
    return {"mean_return": round(float(np.mean(returns)), 4), "return_ci95": _bootstrap_ci(returns),
            "goal_rate": round(float(np.mean(achieved)), 4), "goal_rate_ci95": _bootstrap_ci(achieved),
            "false_accusations_per_ep": round(accusations / episodes, 3),
            "attacks_per_ep": round(attacks / episodes, 3)}


def _bootstrap_ci(values: list, draws: int = 2000, seed: int = 0) -> list[float]:
    """角色局均值的 95% 自助法区间：几十局的差异常常落在噪声里，报告必须带上区间才能下结论。"""
    v = np.asarray(values, dtype=np.float64)
    means = np.random.default_rng(seed).choice(v, size=(draws, len(v)), replace=True).mean(axis=1)
    return [round(float(np.percentile(means, 2.5)), 4), round(float(np.percentile(means, 97.5)), 4)]


# ============================================================
#  PPO（RLlib 新 API 栈）
# ============================================================


def train_ppo(cfg: RLConfig, init: GraphPolicyNet | None, log=print) -> GraphPolicyNet:
    import ray
    from ray.rllib.algorithms.ppo import PPOConfig
    from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec
    from ray.rllib.core.rl_module.rl_module import RLModuleSpec

    ray.init(ignore_reinit_error=True, include_dashboard=False, num_cpus=max(2, cfg.env_runners + 1),
             num_gpus=int(cfg.gpus > 0), log_to_driver=False)
    env_config = {"horizon": cfg.horizon, "seed": cfg.seed, "predictor_path": cfg.predictor_path or None}
    model_config = {"hidden": cfg.hidden, "layers": 2}
    config = (
        PPOConfig()
        .environment(TianlongEnv, env_config=env_config)
        .multi_agent(policies={"npc"}, policy_mapping_fn=lambda agent_id, episode, **kw: "npc")
        .rl_module(rl_module_spec=MultiRLModuleSpec(rl_module_specs={
            "npc": RLModuleSpec(module_class=CandidateScoringModule, model_config=model_config)}))
        .env_runners(num_env_runners=cfg.env_runners)
        .learners(num_learners=0, num_gpus_per_learner=cfg.gpus)
        .training(lr=cfg.lr, train_batch_size_per_learner=cfg.train_batch, minibatch_size=250, num_epochs=4,
                  gamma=0.97, lambda_=0.95, entropy_coeff=cfg.entropy, vf_loss_coeff=0.5, clip_param=0.2)
        .debugging(seed=cfg.seed)
    )
    algo = config.build_algo()
    if init is not None:
        weights = {"npc": {f"net.{k}": v.detach().numpy() for k, v in init.state_dict().items()}}
        algo.learner_group.set_weights(weights)
        algo.env_runner_group.sync_weights(from_worker_or_learner_group=algo.learner_group, inference_only=True)
    for it in range(cfg.ppo_iterations):
        res = algo.train()
        er = res.get("env_runners", {})
        log(f"[ppo {it + 1}] episode_return_mean={er.get('episode_return_mean', float('nan')):.3f} "
            f"episodes={er.get('num_episodes', 0)}")
    state = algo.learner_group.get_weights()["npc"]
    net = GraphPolicyNet(cfg.hidden)
    net.load_state_dict({k[len("net."):]: torch.as_tensor(v) for k, v in state.items()})
    algo.stop()
    ray.shutdown()
    return net


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="tianlong.learning.rl.train", description="模仿学习 + PPO 训练角色策略")
    for f, default in asdict(RLConfig()).items():
        ap.add_argument(f"--{f.replace('_', '-')}", type=type(default), default=default)
    ap.add_argument("--out", default="artifacts")
    args = vars(ap.parse_args(argv))
    out_dir = Path(args.pop("out"))
    cfg = RLConfig(**args)
    torch.manual_seed(cfg.seed)
    t0 = time.time()
    env = TianlongEnv({"horizon": cfg.horizon, "seed": cfg.seed, "predictor_path": cfg.predictor_path or None})

    report: dict[str, dict] = {"random": evaluate(env, random_policy(cfg.seed), cfg.eval_episodes),
                               "scripted": evaluate(env, scripted_policy, cfg.eval_episodes)}
    print("[eval] random", report["random"], "\n[eval] scripted", report["scripted"])

    demos = collect_demos(env, cfg.demo_episodes, cfg.seed)
    print(f"[bc] demos={len(demos)}")
    bc = GraphPolicyNet(cfg.hidden)
    behavior_clone(bc, demos, cfg.bc_epochs, seed=cfg.seed, smoothing=cfg.bc_smoothing)
    report["bc"] = evaluate(env, net_policy(bc), cfg.eval_episodes)
    print("[eval] bc", report["bc"])

    ppo = train_ppo(cfg, bc)
    report["ppo"] = evaluate(env, net_policy(ppo), cfg.eval_episodes)
    report["ppo_without_world_model_features"] = evaluate(env, net_policy(ppo, zero_predictions=True),
                                                          cfg.eval_episodes)
    print("[eval] ppo", report["ppo"], "\n[eval] ppo(无世界模型特征)", report["ppo_without_world_model_features"])
    report["config"] = asdict(cfg)
    report["seconds"] = round(time.time() - t0, 1)

    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": ppo.state_dict(), "config": asdict(cfg), "vocab": VOCAB}, out_dir / "policy_ppo.pt")
    (out_dir / "policy_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
