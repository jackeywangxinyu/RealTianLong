"""
[INPUT]: 依赖标准库 json / statistics / argparse
[OUTPUT]: 对外提供 load_reports()、seed_summary()（跨训练种子的均值与标准差）、markdown()（带 run_id 与提交号的结果表）、
          main()（python -m tianlong.learning.results 报告.json ... [--out 结果.md]）
[POS]: learning 的结果出口：README 的表格由它从机器可读报告生成，每张表头写明 run_id、提交号、任务指纹与种子——
       手抄数字、混用不同版本的实验在这里没有入口。策略报告按训练种子汇总（同一任务分布、不同种子的差异另报），
       动态模型报告按视角列出；比率指标一律带分子分母
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

POLICY_ROWS = ("random", "wait_only", "scripted", "bc", "ppo", "ppo_test_time_no_predictions")
POLICY_COLS = ("goal_rate", "initial_goal_rate", "new_goal_achievement", "maintenance_success", "search_miss_rate",
               "search_no_evidence_rate", "unprovoked_attack_rate", "false_claim_rate", "invalid_loop_rate")


def load_reports(paths: list[str | Path]) -> list[dict]:
    out = []
    for p in paths:
        d = json.loads(Path(p).read_text())
        d["_path"] = str(p)
        out.append(d)
    return out


def _kind(r: dict) -> str:
    return (r.get("manifest") or {}).get("kind") or ("policy" if "policies" in r else "dynamics")


def _val(cell) -> float | None:
    return cell.get("value") if isinstance(cell, dict) else cell


def seed_summary(reports: list[dict]) -> dict:
    """同一任务指纹下、不同训练种子的策略报告：每个策略每个指标的均值、标准差与种子数。"""
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in reports:
        if _kind(r) != "policy":
            continue
        m = r["manifest"]
        groups[(m.get("task_fingerprint"), (m.get("config") or {}).get("ablate_predictions"))].append(r)
    out = {}
    for (fp, ablate), rs in groups.items():
        table = {}
        for pol in POLICY_ROWS:
            row = {}
            for col in ("mean_return", *POLICY_COLS):
                vals = [_val(r["policies"][pol][col]) for r in rs if pol in r.get("policies", {})]
                vals = [v for v in vals if v is not None]
                if vals:
                    row[col] = {"mean": round(statistics.fmean(vals), 4),
                                "sd": round(statistics.stdev(vals), 4) if len(vals) > 1 else None, "seeds": len(vals)}
            if row:
                table[pol] = row
        out[f"{fp}{'·训练期无预测' if ablate else ''}"] = {
            "runs": [r["manifest"]["run_id"] for r in rs], "commits": sorted({r["manifest"].get("git_sha") for r in rs}),
            "train_seeds": [r["manifest"]["seeds"].get("train") for r in rs], "table": table}
    return out


def _fmt(cell) -> str:
    if isinstance(cell, dict) and "value" in cell:
        v = cell["value"]
        return "—" if v is None else f"{v:.3f} ({cell['num']}/{cell['den']})"
    return "—" if cell is None else f"{cell:.3f}"


def markdown(reports: list[dict]) -> str:
    lines: list[str] = []
    for r in reports:
        m = r.get("manifest") or {}
        head = (f"run `{m.get('run_id')}` · commit `{(m.get('git_sha') or 'None')[:10]}`"
                f"{' (工作区有未提交改动)' if m.get('git_dirty') else ''} · task `{m.get('task_fingerprint')}` "
                f"· seeds {json.dumps(m.get('seeds'), ensure_ascii=False)}")
        if _kind(r) == "policy":
            lines += [f"#### 策略评测（{r['policies']['ppo']['worlds']} 个留出世界）", "", head, "",
                      "| 策略 | 平均回报 [世界聚类 95%] | " + " | ".join(POLICY_COLS) + " |",
                      "|---|---|" + "---|" * len(POLICY_COLS)]
            for pol in POLICY_ROWS:
                p = r["policies"].get(pol)
                if p:
                    ci = p["mean_return_ci95_world"]
                    lines.append(f"| {pol} | {p['mean_return']:.3f} [{ci[0]}, {ci[1]}] | "
                                 + " | ".join(_fmt(p[c]) for c in POLICY_COLS) + " |")
            lines += ["", "| 配对比较（同一批世界） | 差 | 95% 区间 | 判定 |", "|---|---|---|---|"]
            for k, c in r.get("paired_comparisons", {}).items():
                lines.append(f"| {k} | {c['diff']} | {c['ci95_world']} | {c['verdict']} |")
        else:
            view = (m.get("config") or {}).get("view", "?")
            lines += [f"#### 动态模型（{view} 视角）", "", head, "", "| 指标 | 值 |", "|---|---|"]
            for k in ("success_acc", "success_brier", "success_brier_calibrated", "holder_changed_recall",
                      "holder_unchanged_kept", "holder_unknown_wrongly_determined", "holder_gone_recall",
                      "attr_changed_acc", "discover_recall", "obs_gain_mae", "obs_gain_mae_train_mean_baseline"):
                if k in r:
                    lines.append(f"| {k} | {_fmt(r[k])} |")
        lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="tianlong.learning.results", description="从报告生成可溯源的结果表")
    ap.add_argument("reports", nargs="+")
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    reports = load_reports(a.reports)
    text = markdown(reports)
    summary = seed_summary(reports)
    if summary:
        text += "\n#### 跨训练种子\n\n```json\n" + json.dumps(summary, indent=1, ensure_ascii=False) + "\n```\n"
    if a.out:
        Path(a.out).write_text(text)
    print(text)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
