"""
[INPUT]: 依赖 torch，ray.rllib 的 PPOConfig / RLModuleSpec / MultiRLModuleSpec，learning/rl 的 env / module / imitation / evaluation，
         learning/task 的 TaskConfig，learning/schema 的 SCHEMA
[OUTPUT]: 对外提供 RLConfig（含展平的 TaskConfig 字段）、train_ppo()、main()（python -m tianlong.learning.rl.train）
[POS]: learning/rl 的训练流水线：模仿学习初始化（脚本策略示范，见 imitation）→ PPO（同一策略网络被所有角色共享参数，
       但各自观测各自的认知）→ 留出种子上对照 随机 / 永远等待 / 脚本 / 模仿 / PPO，并做“去掉世界模型预测特征”的消融（见 evaluation）。
       TaskConfig 一份解析、处处同用：示范、PPO 的每个 env runner、评测与检查点读的是同一个任务分布；
       env_runners/gpus 让同一 CLI 在 Colab 上并行采样、GPU 学习；策略检查点带规格指纹、视角 policy、训练时的 ObsSpec 与 TaskConfig
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from tianlong.learning.rl.env import TianlongEnv
from tianlong.learning.rl.evaluation import evaluate, net_policy, random_policy, scripted_policy, wait_policy
from tianlong.learning.rl.imitation import behavior_clone, collect_demos, holdout_metrics
from tianlong.learning.rl.module import CandidateScoringModule, GraphPolicyNet
from tianlong.learning.rl.rewards import REWARD_VERSION
from tianlong.learning.schema import FEATURES_VERSION, SCHEMA
from tianlong.learning.task import ALL_GOALS, TaskConfig


@dataclass(frozen=True)
class RLConfig:
    demo_episodes: int = 120
    bc_epochs: int = 8
    ppo_iterations: int = 20
    train_batch: int = 1500
    lr: float = 3e-4
    gamma: float = 0.97            # PPO 折扣；塑形用同一个 γ 才保持策略不变
    hidden: int = 64
    eval_episodes: int = 40
    seed: int = 0
    predictor_path: str = ""
    entropy: float = 0.01          # PPO 熵正则：模仿学习后的策略很尖锐，探索不足时调高
    bc_smoothing: float = 0.0      # 模仿学习的标签平滑：避免初始策略过度确定、PPO 无从探索
    bc_wait_share: float = 0.5     # 模仿学习里“等待”样本占的总权重（[0, 1]；见 imitation.bc_weights）
    env_runners: int = 0           # 并行采样进程数；0 = 在驱动进程里采样（小机器），Colab 上可设 2~8
    gpus: float = 0.0              # 学习器 GPU 数（Colab 设 1）
    # ---- 任务分布（TaskConfig 展平，CLI 可直接传）----
    jianghu: float = 0.5
    max_places: int = 5
    max_items: int = 4
    max_persons: int = 3
    scroll_rate: float = 0.5
    scroll_held: float = 0.0
    goals: str = ",".join(ALL_GOALS)
    horizon: int = 30

    def task(self) -> TaskConfig:
        return TaskConfig(self.jianghu, self.max_places, self.max_items, self.max_persons, self.scroll_rate,
                          self.scroll_held, tuple(g for g in self.goals.split(",") if g), self.horizon)

    def env_config(self) -> dict:
        return {"task": self.task().to_dict(), "seed": self.seed, "predictor_path": self.predictor_path or None,
                "reward": {"gamma": self.gamma}}


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
    env_config = cfg.env_config()
    model_config = {"hidden": cfg.hidden, "layers": 2}
    config = (
        PPOConfig()
        .environment(TianlongEnv, env_config=env_config)
        .multi_agent(policies={"npc"}, policy_mapping_fn=lambda agent_id, episode, **kw: "npc")
        .rl_module(rl_module_spec=MultiRLModuleSpec(rl_module_specs={
            "npc": RLModuleSpec(module_class=CandidateScoringModule, model_config=model_config)}))
        .env_runners(num_env_runners=cfg.env_runners)
        .learners(num_learners=0, num_gpus_per_learner=cfg.gpus)
        .training(lr=cfg.lr, train_batch_size_per_learner=cfg.train_batch,
                  minibatch_size=min(250, cfg.train_batch), num_epochs=4,
                  gamma=cfg.gamma, lambda_=0.95, entropy_coeff=cfg.entropy, vf_loss_coeff=0.5, clip_param=0.2)
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
    env = TianlongEnv(cfg.env_config())

    report: dict[str, dict] = {"random": evaluate(env, random_policy(cfg.seed), cfg.eval_episodes),
                               "wait_only": evaluate(env, wait_policy, cfg.eval_episodes),
                               "scripted": evaluate(env, scripted_policy, cfg.eval_episodes)}
    for k in ("random", "wait_only", "scripted"):
        print(f"[eval] {k}", report[k])

    demos = collect_demos(env, cfg.demo_episodes, cfg.seed)
    held = collect_demos(env, max(10, cfg.demo_episodes // 5), cfg.seed + 7_919)      # 留出世界：另一段种子
    print(f"[bc] demos={len(demos)} holdout={len(held)}")
    bc = GraphPolicyNet(cfg.hidden)
    report["bc_epochs"] = behavior_clone(bc, demos, cfg.bc_epochs, seed=cfg.seed, smoothing=cfg.bc_smoothing,
                                         wait_share=cfg.bc_wait_share)       # type: ignore[assignment]
    report["bc_holdout"] = holdout_metrics(bc, held)
    report["bc"] = evaluate(env, net_policy(bc), cfg.eval_episodes)
    print("[eval] bc", report["bc"], "\n[bc] holdout", report["bc_holdout"])

    ppo = train_ppo(cfg, bc)
    report["ppo"] = evaluate(env, net_policy(ppo), cfg.eval_episodes)
    report["ppo_without_world_model_features"] = evaluate(env, net_policy(ppo, zero_predictions=True),
                                                          cfg.eval_episodes)
    print("[eval] ppo", report["ppo"], "\n[eval] ppo(无世界模型特征)", report["ppo_without_world_model_features"])
    report["config"] = asdict(cfg)
    report["task"] = cfg.task().to_dict()
    report["coverage"] = dict(env.coverage)
    report["seconds"] = round(time.time() - t0, 1)

    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": ppo.state_dict(), "config": asdict(cfg), "schema": SCHEMA,
                "features_version": FEATURES_VERSION, "reward_version": REWARD_VERSION, "view": "policy",
                "obs_spec": asdict(env.obs_spec), "task": cfg.task().to_dict()}, out_dir / "policy_ppo.pt")
    (out_dir / "policy_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
