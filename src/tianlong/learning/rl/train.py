"""
[INPUT]: 依赖 torch，ray.rllib 的 PPOConfig / RLModuleSpec / MultiRLModuleSpec，learning/rl 的 env / module / imitation / evaluation，
         learning/task 的 TaskConfig / arg_type，learning/schema 的 SCHEMA，learning/bundle 的 file_sha256，learning/provenance 的 run_manifest
[OUTPUT]: 对外提供 RLConfig（含展平的 TaskConfig 字段；ablations() / tag() / env_config()）、train_ppo()（可从断点接续）、
          main()（python -m tianlong.learning.rl.train [--resume true]）
[POS]: learning/rl 的训练流水线：模仿学习初始化（脚本策略示范，见 imitation）→ PPO（同一策略网络被所有角色共享参数，
       但各自观测各自的认知）→ 留出种子上对照 随机 / 永远等待 / 脚本 / 模仿 / PPO，并做“去掉世界模型预测特征”的消融（见 evaluation）。
       TaskConfig 一份解析、处处同用：示范、PPO 的每个 env runner、评测与检查点读的是同一个任务分布；
       env_runners/gpus 让同一 CLI 在 Colab 上并行采样、GPU 学习；策略检查点带规格指纹、视角 policy、训练时的 ObsSpec、TaskConfig、
       训练期消融与预测器文件哈希（部署包据此配对）。训练期消融（预测/记忆列整列置零）是另一次运行，报告带逐世界记录，
       由 results --pair 在同一批留出世界上与主实验配对比较。断点续训：模仿学习权重与 PPO 最近权重（原子写入）带着配置，
       --resume 时配置不符即拒绝；续训与不中断是同一次实验，--resume 不进 run_id
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from tianlong.learning.bundle import file_sha256
from tianlong.learning.provenance import run_manifest
from tianlong.learning.rl.env import TianlongEnv
from tianlong.learning.rl.evaluation import (
    compare,
    evaluate,
    net_policy,
    random_policy,
    scripted_policy,
    wait_policy,
)
from tianlong.learning.rl.imitation import DEMO_SEED_FLOOR, behavior_clone, collect_demos, holdout_metrics
from tianlong.learning.rl.module import CandidateScoringModule, GraphPolicyNet
from tianlong.learning.rl.rewards import REWARD_VERSION
from tianlong.learning.schema import FEATURES_VERSION, SCHEMA
from tianlong.learning.task import ALL_GOALS, TaskConfig, arg_type


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
    hide_goal_items: float = 0.0   # E04 探查任务：目标物品藏起来的概率
    roles: float = 0.3             # 江湖世界里“守地 + 约时潜逃”角色出现的概率
    goals: str = ",".join(ALL_GOALS)
    horizon: int = 30
    # ---- 评测与消融 ----
    eval_seed: int = 900_000       # 留出世界的种子起点（与示范、PPO 采样的种子段不相交）
    ablate_predictions: bool = False   # 训练期消融：PPO 与模仿学习全程看不到世界模型预测（与测试期置零分开解释）
    ablate_memory: bool = False        # 训练期消融：全程看不到长期记忆摘要列（M01 的对照）
    margin: float = 0.05           # 等效判定的容许差（目标达成率）；区间整个落在 ±margin 内才说“等效”

    def task(self) -> TaskConfig:
        return TaskConfig(jianghu=self.jianghu, max_places=self.max_places, max_items=self.max_items,
                          max_persons=self.max_persons, scroll_rate=self.scroll_rate, scroll_held=self.scroll_held,
                          hide_goal_items=self.hide_goal_items, roles=self.roles,
                          goals=tuple(g for g in self.goals.split(",") if g), horizon=self.horizon)

    def ablations(self) -> tuple[str, ...]:
        """训练期消融的名字（observation.ABLATIONS）：进 env_config、检查点与报告，上线时 LearnedPolicy 照样置零。"""
        return tuple(n for n, on in (("predictions", self.ablate_predictions), ("memory", self.ablate_memory)) if on)

    def tag(self) -> str:
        suffix = {"predictions": "_noPred", "memory": "_noMem"}
        return f"policy_ppo{''.join(suffix[n] for n in self.ablations())}_s{self.seed}"

    def env_config(self) -> dict:
        return {"task": self.task().to_dict(), "seed": self.seed, "predictor_path": self.predictor_path or None,
                "reward": {"gamma": self.gamma}, "ablate": list(self.ablations())}


# ============================================================
#  PPO（RLlib 新 API 栈）
# ============================================================


def _net_from(state: dict, hidden: int) -> GraphPolicyNet:
    net = GraphPolicyNet(hidden)
    net.load_state_dict({k[len("net."):]: torch.as_tensor(v) for k, v in state.items()})
    return net


# 续训时允许不同的字段：加长 PPO 轮数、换评测规模、换并行度与设备（都不改变已训练部分的含义）
_RESUMABLE_DIFF = frozenset({"ppo_iterations", "eval_episodes", "env_runners", "gpus"})


def _resume_state(path: Path, cfg: RLConfig) -> dict:
    """读断点并核对配置：换了影响训练含义的配置却想“续训”，明确拒绝。"""
    st = torch.load(path, map_location="cpu", weights_only=True)
    diff = sorted(k for k, v in asdict(cfg).items()
                  if k not in _RESUMABLE_DIFF and (st.get("config") or {}).get(k) != v)
    if diff:
        raise ValueError(f"断点 {path} 的训练配置与本次不同 {diff}：换配置请换输出目录或删掉断点")
    return st


def train_ppo(cfg: RLConfig, init: GraphPolicyNet | None, log=print, latest: Path | None = None,
              resume: bool = False) -> GraphPolicyNet:
    """latest：断点文件。每隔若干轮保存一次策略权重、已完成轮数与配置；存在且 resume 时从那里接着训练
    （优化器状态不保存——接续后的前几轮更新幅度与不中断时不同，报告里如实标记 resumed_from）。"""
    start = 0
    if latest is not None and resume and latest.exists():
        ck = _resume_state(latest, cfg)
        init, start = GraphPolicyNet(cfg.hidden), int(ck["iteration"])
        init.load_state_dict(ck["state_dict"])
        log(f"[ppo] 从断点第 {start} 轮接着训练")
    if start >= cfg.ppo_iterations and init is not None:
        return init
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
    every = max(1, cfg.ppo_iterations // 10)
    for it in range(start, cfg.ppo_iterations):
        res = algo.train()
        er = res.get("env_runners", {})
        log(f"[ppo {it + 1}] episode_return_mean={er.get('episode_return_mean', float('nan')):.3f} "
            f"episodes={er.get('num_episodes', 0)}")
        if latest is not None and ((it + 1) % every == 0 or it + 1 == cfg.ppo_iterations):
            net = _net_from(algo.learner_group.get_weights()["npc"], cfg.hidden)
            tmp = latest.with_suffix(".tmp")
            torch.save({"state_dict": net.state_dict(), "iteration": it + 1, "config": asdict(cfg)}, tmp)
            tmp.replace(latest)                     # 原子替换：断在写一半时不留坏断点
    net = _net_from(algo.learner_group.get_weights()["npc"], cfg.hidden)
    algo.stop()
    ray.shutdown()
    return net


def _public(r: dict) -> dict:
    return {k: v for k, v in r.items() if not k.startswith("_")}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="tianlong.learning.rl.train", description="模仿学习 + PPO 训练角色策略")
    for f, default in asdict(RLConfig()).items():
        ap.add_argument(f"--{f.replace('_', '-')}", type=arg_type(default), default=default)
    ap.add_argument("--out", default="artifacts")
    ap.add_argument("--resume", type=arg_type(False), default=False,
                    help="输出目录里有同配置的模仿学习权重与 PPO 断点就接着用（不进 run_id）")
    args = vars(ap.parse_args(argv))
    out_dir, resume = Path(args.pop("out")), args.pop("resume")
    cfg = RLConfig(**args)
    torch.manual_seed(cfg.seed)
    t0 = time.time()
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = cfg.tag()
    predictor_sha = file_sha256(cfg.predictor_path) if cfg.predictor_path else None
    env = TianlongEnv(cfg.env_config())
    n = cfg.eval_episodes
    timing: dict[str, float] = {}

    def ev(policy) -> dict:
        t = time.time()
        r = evaluate(env, policy, n, seed_base=cfg.eval_seed, keep_logs=True)
        timing["eval"] = round(timing.get("eval", 0.0) + time.time() - t, 1)
        return r

    runs: dict[str, dict] = {"random": ev(random_policy(cfg.seed)), "wait_only": ev(wait_policy),
                             "scripted": ev(scripted_policy)}
    for k in runs:
        print(f"[eval] {k}", _brief(runs[k]))

    if cfg.eval_seed + n > DEMO_SEED_FLOOR:
        raise ValueError(f"评测种子 [{cfg.eval_seed}, {cfg.eval_seed + n}) 会与示范种子（≥{DEMO_SEED_FLOOR}）重叠")
    demos = collect_demos(env, cfg.demo_episodes, cfg.seed, "demo")
    held = collect_demos(env, max(10, cfg.demo_episodes // 5), cfg.seed, "bc_holdout")   # 留出世界：另一条种子流
    print(f"[bc] demos={len(demos)} holdout={len(held)}")
    bc = GraphPolicyNet(cfg.hidden)
    bc_file = out_dir / f"{tag}.bc.pt"
    t = time.time()
    if resume and bc_file.exists():
        saved = _resume_state(bc_file, cfg)
        bc.load_state_dict(saved["state_dict"])
        bc_epochs = saved["epochs"]
        print("[bc] 从断点载入模仿学习权重")
    else:
        bc_epochs = behavior_clone(bc, demos, cfg.bc_epochs, seed=cfg.seed, smoothing=cfg.bc_smoothing,
                                   wait_share=cfg.bc_wait_share)
        torch.save({"state_dict": bc.state_dict(), "epochs": bc_epochs, "config": asdict(cfg)}, bc_file)
    timing["bc"] = round(time.time() - t, 1)
    runs["bc"] = ev(net_policy(bc))
    print("[eval] bc", _brief(runs["bc"]))

    t = time.time()
    latest = out_dir / f"{tag}.ppo_latest.pt"
    resumed_from = int(_resume_state(latest, cfg)["iteration"]) if resume and latest.exists() else None
    ppo = train_ppo(cfg, bc, latest=latest, resume=resume)
    timing["ppo"] = round(time.time() - t, 1)
    runs["ppo"] = ev(net_policy(ppo))
    print("[eval] ppo", _brief(runs["ppo"]))
    if "predictions" not in cfg.ablations():          # 训练期就没见过预测的策略，测试期再置零只是重复同一评测
        runs["ppo_test_time_no_predictions"] = ev(net_policy(ppo, ablate=("predictions",)))   # 测试期消融
        print("[eval] ppo(测试期置零预测)", _brief(runs["ppo_test_time_no_predictions"]))

    logs = {k: v["_logs"] for k, v in runs.items()}
    comparisons = {
        f"ppo_vs_{b}:{m}": compare(logs["ppo"], logs[b], m, cfg.margin if m == "goal_rate" else None)
        for b in ("wait_only", "scripted", "bc", "ppo_test_time_no_predictions") if b in logs
        for m in ("mean_return", "goal_rate")
    }
    comparisons["scripted_vs_wait_only:goal_rate"] = compare(logs["scripted"], logs["wait_only"], "goal_rate",
                                                             cfg.margin)
    report = {
        "manifest": run_manifest("policy", asdict(cfg), cfg.task(), seeds={
            "train": cfg.seed, "demo": f"demo_seed('demo', {cfg.seed}, 0..{cfg.demo_episodes})",
            "bc_holdout": f"demo_seed('bc_holdout', {cfg.seed}, ...)", "eval": [cfg.eval_seed, cfg.eval_seed + n]},
            extra={"training_time_ablation": list(cfg.ablations()),
                   "predictor_sha256": predictor_sha, "resumed_from_ppo_iteration": resumed_from,
                   "device": {"learner_gpus": cfg.gpus, "env_runners": cfg.env_runners,
                              "cuda_available": torch.cuda.is_available(),
                              "note": "GPU 只用于 PPO 学习器；示范、评测与环境采样都在 CPU 上"},
                   "timing_seconds": timing}),
        "policies": {k: _public(v) for k, v in runs.items()},
        # 逐世界记录：跨报告（如训练期有/无预测两次运行）也能在同一批世界上配对比较（results --pair）
        "worlds": {k: [lg.to_dict() for lg in v["_logs"]] for k, v in runs.items()},
        "paired_comparisons": comparisons,
        "bc_epochs": bc_epochs,
        "bc_holdout": holdout_metrics(bc, held),
        # 驱动进程里的环境只跑了示范与评测；PPO 各 env runner 用同一份 env_config（同一 TaskConfig）取样
        "coverage_driver_demo_and_eval": dict(env.coverage),
        "seconds": round(time.time() - t0, 1),
    }
    torch.save({"state_dict": ppo.state_dict(), "config": asdict(cfg), "schema": SCHEMA, "predictor_sha256": predictor_sha,
                "features_version": FEATURES_VERSION, "reward_version": REWARD_VERSION, "view": "policy",
                "obs_spec": asdict(env.obs_spec), "task": cfg.task().to_dict(), "ablate": list(cfg.ablations()),
                "manifest": report["manifest"]},
               out_dir / f"{tag}.pt")
    (out_dir / f"{tag}.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False))
    print(json.dumps({k: report[k] for k in ("manifest", "paired_comparisons")}, indent=2, ensure_ascii=False))
    return 0


def _brief(r: dict) -> str:
    return (f"return={r['mean_return']} goal={r['goal_rate']['value']} new={r['new_goal_achievement']['value']} "
            f"kept={r['maintenance_success']['value']} search_miss={r['per_episode']['search_miss']} "
            f"unprovoked={r['per_episode']['attack_unprovoked']}")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
