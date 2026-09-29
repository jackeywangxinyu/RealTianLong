"""
[INPUT]: 依赖 torch，torch_geometric 的 DataLoader，learning 的 datagen / samples / model / schema / task
[OUTPUT]: 对外提供 TrainConfig、split_by_world()、split_worlds()、split3()、fit_baselines()、coverage()、loss_fn()、fit_temperature()、evaluate()、
          mean_loss()、train_dynamics()（按校准损失选轮、可断点续训）、save_checkpoint()、main()（python -m tianlong.learning.train）
[POS]: learning 的训练与验收：按世界切分（检验对没见过的布局的泛化），指标按 schema.TARGETS 声明的覆盖范围逐项报告——
       位置召回只叫“位置召回”（holder_*），不冒充“全部事实”；动态布尔属性逐属性、数值属性（进度、内力、点穴余时）给 MAE；
       成败按操作分项给 Brier，对照的常数基线取自**训练集**（测试集最优常数只作诊断，标明 test_const）；
       角色视角另报发现新实体、GONE 与有效新观察数；coverage 报告每类机制在数据里出现了多少次。
       世界三分：训练 / 校准（成败头的温度只在这里拟合）/ 测试（只做最终报告，报原始与校准后两种 Brier）；
       报告与检查点都带 manifest（提交、版本、配置、种子、设备）。
       模型选择：每轮在校准世界上算一次损失，最终用校准损失最低的一轮（测试世界从不参与选择）。
       断点续训：每轮结束把 模型/优化器/调度器/随机数状态/当前最优 原子写入断点文件；--resume true 时从那里接着训练，
       配置不一致即拒绝（不会拿别的配置的断点冒充续训）。数据按种子确定性重生成，不缓存。
       workers 按世界并行生成数据（与顺序收集逐项相同，不进 run_id）；device=auto 时有 GPU 即用 GPU，同一 CLI 可直接在 Colab 上放大跑
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

from __future__ import annotations

import argparse
import json
import random
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch_geometric.loader import DataLoader

from tianlong.learning.datagen import RolloutConfig, collect
from tianlong.learning.model import DynamicsModel, DynamicsOutput, loss_terms
from tianlong.learning.provenance import EXACT_RESOURCE_KEYS, run_manifest
from tianlong.learning.samples import GONE, NEW, Sample, to_data
from tianlong.learning.schema import DYN_BOOL, DYN_NUM, FEATURES_VERSION, OBS_GAIN_CAP, OPS, SCHEMA
from tianlong.learning.task import TaskConfig, arg_type


@dataclass(frozen=True)
class TrainConfig:
    view: str = "env"          # env：环境动态；agent：角色视角
    worlds: int = 400
    steps: int = 24
    epochs: int = 20
    batch: int = 64
    lr: float = 2e-3
    hidden: int = 64
    seed: int = 0
    val_frac: float = 0.2      # 留出测试世界比例：只用于最终报告
    calib_frac: float = 0.1    # 从训练世界里再留出的校准世界比例：温度缩放只在这里拟合
    jianghu: float = 0.5       # 江湖化世界比例（见 learning/task）
    scroll_rate: float = 0.5   # 江湖世界里有秘籍的概率
    scroll_held: float = 0.0   # 秘籍开局就在某人手上的概率（修习机制的数据覆盖）
    hide_goal_items: float = 0.0   # 目标物品藏起来的概率（让“查看才有发现”在数据里足够常见）
    roles: float = 0.3             # 江湖世界里“守地 + 约时潜逃”角色出现的概率
    max_places: int = 5
    max_items: int = 4
    max_persons: int = 3
    device: str = "auto"       # auto：有 GPU 用 GPU（Colab），否则 CPU
    workers: int = 1           # 数据生成的并行进程数（0 = 本机全部核）：只改执行方式，样本逐项相同

    def task(self) -> TaskConfig:
        return TaskConfig(jianghu=self.jianghu, max_places=self.max_places, max_items=self.max_items,
                          max_persons=self.max_persons, scroll_rate=self.scroll_rate, scroll_held=self.scroll_held,
                          hide_goal_items=self.hide_goal_items, roles=self.roles)


def _device(name: str) -> torch.device:
    return torch.device("cuda" if name == "auto" and torch.cuda.is_available() else ("cpu" if name == "auto" else name))


def split_by_world(samples: list[Sample], world_of: list[int], val_frac: float, seed: int):
    worlds = sorted(set(world_of))
    random.Random(seed).shuffle(worlds)
    held = set(worlds[: max(1, int(len(worlds) * val_frac))])
    train = [s for s, w in zip(samples, world_of, strict=True) if w not in held]
    test = [s for s, w in zip(samples, world_of, strict=True) if w in held]
    return train, test


def split_worlds(world_of: list[int], test_frac: float, calib_frac: float, seed: int) -> dict[str, set[int]]:
    """按世界三分（一次洗牌，同一个种子）：测试世界取前段，校准世界紧随其后，其余训练。"""
    worlds = sorted(set(world_of))
    random.Random(seed).shuffle(worlds)
    n_test = max(1, int(len(worlds) * test_frac))
    n_calib = max(1, int(len(worlds) * calib_frac)) if calib_frac > 0 else 0
    return {"test": set(worlds[:n_test]), "calib": set(worlds[n_test:n_test + n_calib]),
            "train": set(worlds[n_test + n_calib:])}


def split3(samples: list[Sample], world_of: list[int], test_frac: float, calib_frac: float, seed: int):
    """按世界三分：训练 / 校准（选轮与拟合温度）/ 测试（只做最终报告）。"""
    parts = split_worlds(world_of, test_frac, calib_frac, seed)
    test_w, calib_w = parts["test"], parts["calib"]
    pick = [(s, "test" if w in test_w else "calib" if w in calib_w else "train")
            for s, w in zip(samples, world_of, strict=True)]
    return tuple([s for s, part in pick if part == name] for name in ("train", "calib", "test"))


# ============================================================
#  基线与覆盖：常数基线只从训练集拟合；覆盖报告回答“这个机制在数据里出现过几次”
# ============================================================


def fit_baselines(train: list[Sample]) -> dict:
    by_op: dict[str, list[float]] = {}
    for s in train:
        by_op.setdefault(OPS[s.action.op], []).append(s.success)
    agent = [s for s in train if s.view == "agent"]
    return {
        "success_rate_by_op": {k: float(np.mean(v)) for k, v in sorted(by_op.items())},
        "success_rate": float(np.mean([s.success for s in train])) if train else 0.5,
        "obs_gain_mean": float(np.mean([s.obs_gain for s in agent])) if agent else 0.0,
        "discover_rate": float(np.mean([s.discover for s in agent])) if agent else 0.0,
    }


def coverage(samples: list[Sample]) -> dict:
    c: Counter = Counter()
    for s in samples:
        c[f"op:{OPS[s.action.op]}"] += 1
        c["holder_changed"] += int((s.holder_now != s.holder_next).sum())
        c["holder_gone"] += int((s.holder_next == GONE).sum())
        c["holder_new"] += int((s.holder_next == NEW).sum())
        for j, k in enumerate(DYN_BOOL):
            c[f"attr:{k}"] += int((s.bool_mask[:, j] & (s.bool_now[:, j] != s.bool_next[:, j])).sum())
        for j, k in enumerate(DYN_NUM):
            chg = s.num_mask[:, j] & s.num_next_known[:, j] & (np.abs(s.num_now[:, j] - s.num_next[:, j]) > 1e-6)
            c[f"num:{k}"] += int(chg.sum())
        c["discover"] += int(s.discover > 0)
        c["obs_gain_positive"] += int(s.obs_gain > 0)
    return {"samples": len(samples), **dict(sorted(c.items()))}


def loss_fn(out: DynamicsOutput, data) -> tuple[torch.Tensor, dict[str, float]]:
    terms = loss_terms(out, data)
    total = sum(terms.values())
    return total, {k: float(v.detach()) for k, v in terms.items()}  # type: ignore[return-value]


# ============================================================
#  评测
# ============================================================


@torch.no_grad()
def fit_temperature(model: DynamicsModel, loader: DataLoader, device: torch.device | None = None) -> float:
    """在校准世界上为成败头拟合温度（对数网格上最小化交叉熵，确定性）；测试世界不参与。"""
    model.eval()
    logits, ys = [], []
    for data in loader:
        data = data.to(device) if device is not None else data
        logits.append(model(data).success.cpu())
        ys.append(data.success.cpu())
    if not logits:
        return 1.0
    z, y = torch.cat(logits), torch.cat(ys)
    grid = torch.exp(torch.linspace(-2.0, 2.0, 81))
    losses = torch.stack([torch.nn.functional.binary_cross_entropy_with_logits(z / t, y) for t in grid])
    return round(float(grid[int(losses.argmin())]), 4)


@torch.no_grad()
def evaluate(model: DynamicsModel, loader: DataLoader, device: torch.device | None = None,
             baselines: dict | None = None, temperature: float = 1.0) -> dict:
    model.eval()
    base = baselines or {}
    const = base.get("success_rate_by_op", {})
    c: Counter = Counter()
    by_op: Counter = Counter()
    by_attr: Counter = Counter()
    by_num: Counter = Counter()
    for data in loader:
        data = data.to(device) if device is not None else data
        out = model(data)
        y = data.success
        p = torch.sigmoid(out.success / temperature)
        ok = (out.success > 0) == (y > 0.5)
        c["succ_n"] += int(y.numel())
        c["succ_ok"] += int(ok.sum())
        c["succ_pos"] += int((y > 0.5).sum())
        c["brier"] += float(((p - y) ** 2).sum())
        for op_i, hit, pi, yi in zip(data.act_op.tolist(), ok.tolist(), p.tolist(), y.tolist(), strict=True):
            op = OPS[op_i]
            q = const.get(op, base.get("success_rate", 0.5))
            for key, v in (("n", 1), ("ok", int(hit)), ("se", (pi - yi) ** 2), ("pos", yi), ("const_se", (q - yi) ** 2)):
                by_op[(op, key)] += v

        # ---- 位置 ----
        pred = out.holder.argmax(-1)
        nmax = out.holder.size(1) - 3
        changed = out.holder_next != out.holder_now
        c["keep_ok"] += int((pred[~changed] == out.holder_next[~changed]).sum())
        c["keep_n"] += int((~changed).sum())
        c["chg_ok"] += int((pred[changed] == out.holder_next[changed]).sum())
        c["chg_n"] += int(changed.sum())
        moved = pred != out.holder_now
        c["pred_chg"] += int(moved.sum())
        c["pred_chg_ok"] += int((pred[moved] == out.holder_next[moved]).sum())
        unknown = out.holder_next == nmax
        c["null_n"] += int(unknown.sum())
        c["null_wrong"] += int(((pred[unknown] < nmax) | (pred[unknown] == nmax + 2)).sum())   # 指到了某处或 NEW
        gone = out.holder_next == nmax + 1
        c["gone_n"] += int(gone.sum())
        c["gone_ok"] += int((pred[gone] == nmax + 1).sum())
        new = out.holder_next == nmax + 2
        c["new_n"] += int(new.sum())
        c["new_ok"] += int((pred[new] == nmax + 2).sum())

        # ---- 布尔属性 ----
        m = data.bool_mask
        a_pred = out.attr.argmax(-1)
        a_chg = m & (data.bool_next != data.bool_now)
        a_keep = m & (data.bool_next == data.bool_now) & (data.bool_now != 1)
        hit = a_pred == data.bool_next
        c["attr_chg_ok"] += int(hit[a_chg].sum())
        c["attr_chg_n"] += int(a_chg.sum())
        c["attr_keep_ok"] += int(hit[a_keep].sum())
        c["attr_keep_n"] += int(a_keep.sum())
        for j, name in enumerate(DYN_BOOL):
            by_attr[(name, "n")] += int(a_chg[:, j].sum())
            by_attr[(name, "ok")] += int((hit[:, j] & a_chg[:, j]).sum())

        # ---- 数值属性：变化了的条目上的 MAE，对照“不变”基线 ----
        vm = data.num_mask & data.num_next_known & data.num_now_known
        n_chg = vm & ((data.num_next - data.num_now).abs() > 1e-6)
        err = (out.num - data.num_next).abs()
        stay = (data.num_now - data.num_next).abs()
        for j, name in enumerate(DYN_NUM):
            sel = n_chg[:, j]
            by_num[(name, "n")] += int(sel.sum())
            by_num[(name, "mae")] += float(err[:, j][sel].sum())
            by_num[(name, "base")] += float(stay[:, j][sel].sum())

        # ---- 角色视角专有：发现、有效新观察数 ----
        ag = data.is_agent
        if ag.any():
            d_pred, d_true = out.discover[ag] > 0, data.discover[ag] > 0.5
            c["disc_tp"] += int((d_pred & d_true).sum())
            c["disc_pred"] += int(d_pred.sum())
            c["disc_true"] += int(d_true.sum())
            c["gain_n"] += int(ag.sum())
            c["gain_ae"] += float((out.obs_gain[ag] - data.obs_gain[ag]).abs().sum())
            c["gain_base_ae"] += float((base.get("obs_gain_mean", 0.0) - data.obs_gain[ag]).abs().sum())

    def r(a: str, b: str) -> float | None:
        return round(c[a] / c[b], 4) if c[b] else None          # 分母为 0：算不出来就是 null（报告是严格 JSON）

    def split(cnt: Counter) -> dict[str, str]:
        keys = sorted({k for k, _ in cnt})
        return {k: f"{cnt[(k, 'ok')] / cnt[(k, 'n')]:.3f} (n={cnt[(k, 'n')]})" for k in keys if cnt[(k, "n")]}

    def calibration() -> dict[str, str]:
        """带随机的行动（动手）准确率有天花板；拿 Brier 分与训练集常数相比才看得出学到了多少；test_const 仅作诊断。"""
        res = {}
        for k in sorted({k for k, _ in by_op}):
            n = by_op[(k, "n")]
            q = by_op[(k, "pos")] / n
            res[k] = (f"acc {by_op[(k, 'ok')] / n:.3f} · brier {by_op[(k, 'se')] / n:.3f}"
                      f"（训练集常数 {by_op[(k, 'const_se')] / n:.3f}；test_const {q * (1 - q):.3f}）· n={n}")
        return res

    def num_mae() -> dict[str, str]:
        return {k: f"mae {by_num[(k, 'mae')] / by_num[(k, 'n')]:.3f}（不变基线 {by_num[(k, 'base')] / by_num[(k, 'n')]:.3f}）"
                   f" · n={by_num[(k, 'n')]}" for k in DYN_NUM if by_num[(k, "n")]}

    rate = base.get("success_rate", 0.5)
    majority = 1.0 if rate >= 0.5 else 0.0
    return {
        "success_acc": r("succ_ok", "succ_n"),
        "success_train_majority_baseline": round(
            (c["succ_pos"] if majority else c["succ_n"] - c["succ_pos"]) / max(c["succ_n"], 1), 4),
        "success_brier": round(c["brier"] / max(c["succ_n"], 1), 4),
        "success_by_op": calibration(),
        "holder_changed_recall": r("chg_ok", "chg_n"), "holder_changed_count": c["chg_n"],
        "holder_unchanged_kept": r("keep_ok", "keep_n"),
        "holder_predicted_change_precision": r("pred_chg_ok", "pred_chg"),
        "holder_unknown_wrongly_determined": r("null_wrong", "null_n"), "holder_unknown_count": c["null_n"],
        "holder_gone_recall": r("gone_ok", "gone_n"), "holder_gone_count": c["gone_n"],
        "holder_new_recall": r("new_ok", "new_n"), "holder_new_count": c["new_n"],
        "attr_changed_acc": r("attr_chg_ok", "attr_chg_n"), "attr_changed_count": c["attr_chg_n"],
        "attr_changed_by_attr": split(by_attr), "attr_unchanged_kept": r("attr_keep_ok", "attr_keep_n"),
        "num_changed_by_attr": num_mae(),
        "discover_precision": r("disc_tp", "disc_pred"), "discover_recall": r("disc_tp", "disc_true"),
        "discover_count": c["disc_true"],
        "obs_gain_mae": r("gain_ae", "gain_n"), "obs_gain_mae_train_mean_baseline": r("gain_base_ae", "gain_n"),
        # 什么都不变基线：未变保持 = 1，变化召回 = 0
        "baseline_changed_recall": 0.0, "baseline_unchanged_kept": 1.0,
        "targets_cap": {"obs_gain": OBS_GAIN_CAP},
    }


@torch.no_grad()
def mean_loss(model: DynamicsModel, loader: DataLoader, device: torch.device) -> float | None:
    """整个 loader 上的平均损失（按批平均）；空 loader 返回 None。"""
    model.eval()
    total, n = 0.0, 0
    for data in loader:
        data = data.to(device)
        total += float(loss_fn(model(data), data)[0])
        n += 1
    return total / n if n else None


def _save_state(path: Path, state: dict) -> None:
    tmp = path.with_suffix(".tmp")
    torch.save(state, tmp)
    tmp.replace(path)                     # 原子替换：断在写一半时不留坏断点


def _load_state(path: Path, cfg: TrainConfig) -> dict:
    st = torch.load(path, map_location="cpu", weights_only=True)
    diff = sorted(k for k, v in asdict(cfg).items()
                  if k not in EXACT_RESOURCE_KEYS and (st.get("config") or {}).get(k) != v)
    if diff:
        raise ValueError(f"断点 {path} 的训练配置与本次不同 {diff}：换配置请换输出目录或删掉断点")
    return st


def train_dynamics(cfg: TrainConfig, log=print, state_path: Path | None = None,
                   resume: bool = False) -> tuple[DynamicsModel, dict]:
    """state_path：断点文件（每轮原子写入）；resume 且断点存在时从那里接着训练。"""
    torch.manual_seed(cfg.seed)
    t0 = time.time()
    device = _device(cfg.device)
    rollouts = collect(RolloutConfig(worlds=cfg.worlds, steps=cfg.steps, seed=cfg.seed, task=cfg.task()), cfg.workers)
    samples = rollouts.env if cfg.view == "env" else rollouts.agent
    train, calib, test = split3(samples, rollouts.world_of, cfg.val_frac, cfg.calib_frac, cfg.seed)
    t_data = time.time() - t0
    log(f"[data] {cfg.view}: train={len(train)} calib={len(calib)} test={len(test)} device={device} ({t_data:.1f}s)")
    baselines = fit_baselines(train)
    tr = DataLoader([to_data(s) for s in train], batch_size=cfg.batch, shuffle=True)
    ca = DataLoader([to_data(s) for s in calib], batch_size=256)
    te = DataLoader([to_data(s) for s in test], batch_size=256)
    model = DynamicsModel(cfg.hidden).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)
    start, best, history = 0, {"epoch": None, "calib_loss": None, "state_dict": None}, []
    if resume and state_path is not None and state_path.exists():
        st = _load_state(state_path, cfg)
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        sched.load_state_dict(st["sched"])
        torch.set_rng_state(st["rng"])
        start, best, history = int(st["epoch"]), st["best"], list(st["history"])
        log(f"[resume] 从第 {start} 轮之后接着训练（最优轮 {best['epoch']}）")
    for epoch in range(start, cfg.epochs):
        model.train()
        total = 0.0
        for data in tr:
            data = data.to(device)
            opt.zero_grad()
            loss, _ = loss_fn(model(data), data)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item()
        sched.step()
        calib_loss = mean_loss(model, ca, device)
        history.append({"epoch": epoch + 1, "train_loss": round(total / max(len(tr), 1), 5),
                        "calib_loss": None if calib_loss is None else round(calib_loss, 5)})
        if calib_loss is None or best["calib_loss"] is None or calib_loss < best["calib_loss"]:
            best = {"epoch": epoch + 1, "calib_loss": calib_loss,
                    "state_dict": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}
        if epoch % 5 == 4 or epoch == cfg.epochs - 1:
            m = evaluate(model, te, device, baselines)
            log(f"[epoch {epoch + 1}] loss={total / max(len(tr), 1):.4f} calib={calib_loss} succ={m['success_acc']} "
                f"holder_recall={m['holder_changed_recall']} kept={m['holder_unchanged_kept']} "
                f"attr={m['attr_changed_acc']}")
        if state_path is not None:
            _save_state(state_path, {"config": asdict(cfg), "epoch": epoch + 1, "model": model.state_dict(),
                                     "opt": opt.state_dict(), "sched": sched.state_dict(),
                                     "rng": torch.get_rng_state(), "best": best, "history": history})
    if best["state_dict"] is not None:
        model.load_state_dict(best["state_dict"])       # 校准损失最低的一轮；测试世界不参与选择
    temperature = fit_temperature(model, ca, device)
    metrics = evaluate(model, te, device, baselines)
    calibrated = evaluate(model, te, device, baselines, temperature)
    metrics["success_temperature_fit_on_calib"] = temperature
    metrics["success_brier_calibrated"] = calibrated["success_brier"]
    metrics["success_by_op_calibrated"] = calibrated["success_by_op"]
    metrics["baselines_fit_on_train"] = baselines
    metrics["coverage_train"] = coverage(train)
    metrics["coverage_test"] = coverage(test)
    metrics["timing_seconds"] = {"data": round(t_data, 1), "total": round(time.time() - t0, 1), "device": str(device)}
    worlds = split_worlds(rollouts.world_of, cfg.val_frac, cfg.calib_frac, cfg.seed)
    metrics["split"] = {"worlds": {k: len(v) for k, v in worlds.items()},
                        "samples": {"train": len(train), "calib": len(calib), "test": len(test)},
                        "unit_of_split": "world"}
    metrics["model_selection"] = {"criterion": "min calib_loss (calib worlds; test never used)",
                                  "best_epoch": best["epoch"], "best_calib_loss": best["calib_loss"],
                                  "resumed_from_epoch": start or None, "history": history}
    metrics["manifest"] = run_manifest(f"dynamics_{cfg.view}", asdict(cfg), cfg.task(),
                                       seeds={"data": cfg.seed, "split_test_and_calib": cfg.seed},
                                       extra={"device": str(device), "model_scope": "isolated_action"})
    return model.cpu(), metrics


def save_checkpoint(model: DynamicsModel, cfg: TrainConfig, metrics: dict, path: Path) -> None:
    torch.save({"state_dict": model.state_dict(), "config": asdict(cfg), "metrics": metrics, "schema": SCHEMA,
                "features_version": FEATURES_VERSION, "view": cfg.view, "task": cfg.task().to_dict(),
                "success_temperature": metrics.get("success_temperature_fit_on_calib", 1.0),
                "manifest": metrics.get("manifest")}, path)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="tianlong.learning.train", description="训练 GNN 动态模型")
    for f, default in asdict(TrainConfig()).items():
        ap.add_argument(f"--{f.replace('_', '-')}", type=arg_type(default), default=default)
    ap.add_argument("--out", default="artifacts")
    ap.add_argument("--resume", type=arg_type(False), default=False,
                    help="输出目录里有同配置的断点就接着训练（不进 run_id：续训与不中断是同一次实验）")
    args = vars(ap.parse_args(argv))
    out_dir, resume = Path(args.pop("out")), args.pop("resume")
    cfg = TrainConfig(**args)
    out_dir.mkdir(parents=True, exist_ok=True)
    model, metrics = train_dynamics(cfg, state_path=out_dir / f"dynamics_{cfg.view}.resume.pt", resume=resume)
    save_checkpoint(model, cfg, metrics, out_dir / f"dynamics_{cfg.view}.pt")
    (out_dir / f"dynamics_{cfg.view}.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False, allow_nan=False))
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
