"""
[INPUT]: 依赖 torch，torch_geometric 的 DataLoader，learning 的 datagen / samples / model
[OUTPUT]: 对外提供 TrainConfig、split_by_world()、loss_fn()、evaluate()、train_dynamics()、main()（python -m tianlong.learning.train）
[POS]: learning 的训练与验收：按世界切分（检验对没见过的布局的泛化），指标对照“什么都不变”基线——
       动态模型的价值不在总体准确率（不变的事实占绝大多数），而在：变化召回、未变保持、以及不把未知错误确定化
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

from tianlong.learning.datagen import RolloutConfig, collect
from tianlong.learning.model import DynamicsModel, DynamicsOutput
from tianlong.learning.samples import Sample, to_data


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
    val_frac: float = 0.2


def split_by_world(samples: list[Sample], world_of: list[int], val_frac: float, seed: int):
    worlds = sorted(set(world_of))
    random.Random(seed).shuffle(worlds)
    held = set(worlds[: max(1, int(len(worlds) * val_frac))])
    train = [s for s, w in zip(samples, world_of, strict=True) if w not in held]
    test = [s for s, w in zip(samples, world_of, strict=True) if w in held]
    return train, test


def loss_fn(out: DynamicsOutput, data) -> tuple[torch.Tensor, dict[str, float]]:
    l_succ = F.binary_cross_entropy_with_logits(out.success, data.success)
    l_holder = F.cross_entropy(out.holder, out.holder_next) if out.holder.numel() else out.success.sum() * 0
    mask = data.attr_mask
    l_attr = F.cross_entropy(out.attr[mask], data.attr_next[mask]) if mask.any() else out.success.sum() * 0
    total = l_succ + l_holder + l_attr
    return total, {"success": float(l_succ), "holder": float(l_holder), "attr": float(l_attr)}


@torch.no_grad()
def evaluate(model: DynamicsModel, loader: DataLoader) -> dict[str, float]:
    model.eval()
    c = dict(succ_ok=0, succ_n=0, succ_pos=0, keep_ok=0, keep_n=0, chg_ok=0, chg_n=0, pred_chg=0, pred_chg_ok=0,
             null_n=0, null_wrong=0, attr_chg_ok=0, attr_chg_n=0, attr_keep_ok=0, attr_keep_n=0)
    for data in loader:
        out = model(data)
        pred_s = out.success > 0
        c["succ_ok"] += int((pred_s == (data.success > 0.5)).sum())
        c["succ_n"] += int(data.success.numel())
        c["succ_pos"] += int((data.success > 0.5).sum())
        pred = out.holder.argmax(-1)
        nmax = out.holder.size(1) - 1
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
        c["null_wrong"] += int((pred[unknown] != nmax).sum())
        m = data.attr_mask
        a_pred = out.attr.argmax(-1)
        a_chg = m & (data.attr_next != data.attr_now)
        a_keep = m & (data.attr_next == data.attr_now)
        c["attr_chg_ok"] += int((a_pred[a_chg] == data.attr_next[a_chg]).sum())
        c["attr_chg_n"] += int(a_chg.sum())
        c["attr_keep_ok"] += int((a_pred[a_keep] == data.attr_next[a_keep]).sum())
        c["attr_keep_n"] += int(a_keep.sum())

    def r(a: str, b: str) -> float:
        return round(c[a] / c[b], 4) if c[b] else float("nan")

    base_succ = max(c["succ_pos"], c["succ_n"] - c["succ_pos"]) / max(c["succ_n"], 1)
    return {
        "success_acc": r("succ_ok", "succ_n"), "success_majority_baseline": round(base_succ, 4),
        "changed_recall": r("chg_ok", "chg_n"), "changed_count": c["chg_n"],
        "unchanged_kept": r("keep_ok", "keep_n"),
        "predicted_change_precision": r("pred_chg_ok", "pred_chg"),
        "unknown_wrongly_determined": r("null_wrong", "null_n"), "unknown_count": c["null_n"],
        "attr_changed_acc": r("attr_chg_ok", "attr_chg_n"), "attr_changed_count": c["attr_chg_n"],
        "attr_unchanged_kept": r("attr_keep_ok", "attr_keep_n"),
        # 什么都不变基线：未变保持 = 1，变化召回 = 0
        "baseline_changed_recall": 0.0, "baseline_unchanged_kept": 1.0,
    }


def train_dynamics(cfg: TrainConfig, log=print) -> tuple[DynamicsModel, dict]:
    torch.manual_seed(cfg.seed)
    t0 = time.time()
    rollouts = collect(RolloutConfig(worlds=cfg.worlds, steps=cfg.steps, seed=cfg.seed))
    samples = rollouts.env if cfg.view == "env" else rollouts.agent
    train, test = split_by_world(samples, rollouts.world_of, cfg.val_frac, cfg.seed)
    log(f"[data] {cfg.view}: train={len(train)} test={len(test)} ({time.time() - t0:.1f}s)")
    tr = DataLoader([to_data(s) for s in train], batch_size=cfg.batch, shuffle=True)
    te = DataLoader([to_data(s) for s in test], batch_size=256)
    model = DynamicsModel(cfg.hidden)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)
    for epoch in range(cfg.epochs):
        model.train()
        total, parts_sum = 0.0, {"success": 0.0, "holder": 0.0, "attr": 0.0}
        for data in tr:
            opt.zero_grad()
            loss, parts = loss_fn(model(data), data)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += float(loss)
            for k in parts_sum:
                parts_sum[k] += parts[k]
        sched.step()
        if epoch % 5 == 4 or epoch == cfg.epochs - 1:
            m = evaluate(model, te)
            log(f"[epoch {epoch + 1}] loss={total / len(tr):.4f} succ={m['success_acc']} "
                f"chg_recall={m['changed_recall']} kept={m['unchanged_kept']} unk_wrong={m['unknown_wrongly_determined']}")
    metrics = evaluate(model, te)
    metrics["train_seconds"] = round(time.time() - t0, 1)
    return model, metrics


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="tianlong.learning.train", description="训练 GNN 动态模型")
    for f, default in asdict(TrainConfig()).items():
        ap.add_argument(f"--{f.replace('_', '-')}", type=type(default), default=default)
    ap.add_argument("--out", default="artifacts")
    args = vars(ap.parse_args(argv))
    out_dir = Path(args.pop("out"))
    cfg = TrainConfig(**args)
    model, metrics = train_dynamics(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "config": asdict(cfg), "metrics": metrics},
               out_dir / f"dynamics_{cfg.view}.pt")
    (out_dir / f"dynamics_{cfg.view}.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
