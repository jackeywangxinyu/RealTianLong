"""
[INPUT]: 依赖 torch，learning/schema 的 StaleModel，learning/provenance 的 git_state / versions / compat_signature（全部语义版本的唯一定义）；
         按需加载 learning/predictor、learning/rl/policy、agents/predictors
[OUTPUT]: 对外提供 BUNDLE_VERSION、BUNDLE_FILE、file_sha256()、write_bundle()（把动态模型 + 策略 + 报告打成一个带清单的部署包）、
          load_bundle()（逐项核对后加载）、Loaded、main()（python -m tianlong.learning.bundle 目录 --predictor --policy --reports）
[POS]: learning 的部署边界。词表一致不等于模型兼容：规则语义、目标规格、奖励、候选规则、观测布局（目标槽位、预测列、记忆列）
       任何一项变了，旧模型都可能在形状对得上的情况下悄悄做错事。部署包记下的是**训练时**的全部语义版本（取自检查点 manifest，
       不是打包时的代码）与文件哈希：打包时预测器与策略的版本必须一致、且与当前代码一致；加载时再逐项比对、
       列出所有不一致并拒绝（StaleModel）——本仓库不做隐式迁移，要迁移就重训。
       策略训练时用的是哪个预测器（启发式或某个 GNN 文件的哈希），上线就必须用同一个：候选的预测特征是策略输入的一部分
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import torch

from tianlong.learning.provenance import compat_signature, git_state, versions
from tianlong.learning.schema import StaleModel

BUNDLE_VERSION = "bundle-v1"
BUNDLE_FILE = "bundle.json"


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _ckpt(path: Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=True)


def _rel(path: str | Path, root: Path) -> str:
    p = Path(path).resolve()
    try:
        return str(p.relative_to(root.resolve()))
    except ValueError:
        raise ValueError(f"{path} 不在部署包目录 {root} 之内：部署包只记录相对路径，文件必须随目录一起搬") from None


def _trained_compat(ck: dict) -> dict:
    """检查点训练时的语义版本（manifest 里记下的）；缺失的键为 None——旧检查点因此一定对不上。"""
    v = (ck.get("manifest") or {}).get("versions") or {}
    return {k: v.get(k) for k in compat_signature()}


def _diff(a: dict, b: dict) -> list[str]:
    return sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))


def write_bundle(out_dir: str | Path, predictor: str | Path | None = None, policy: str | Path | None = None,
                 reports: tuple[str | Path, ...] | list[str | Path] = ()) -> Path:
    """在 out_dir 写 bundle.json（文件以相对 out_dir 的路径记录）。策略若用 GNN 预测器训练，部署包必须带着同一个文件。"""
    out = Path(out_dir)
    files: dict[str, dict] = {}
    trained: dict[str, dict] = {}
    if predictor is not None:
        ck = _ckpt(Path(predictor))
        trained["predictor"] = _trained_compat(ck)
        if ck.get("view") != "agent":
            raise StaleModel(f"{predictor} 不是角色视角模型（view={ck.get('view')!r}）：部署包里的预测器只能是 agent 视角")
        files["predictor"] = {"path": _rel(predictor, out), "sha256": file_sha256(predictor),
                              "hidden": ck["config"]["hidden"], "temperature": ck.get("success_temperature", 1.0),
                              "run_id": (ck.get("manifest") or {}).get("run_id")}
    if policy is not None:
        ck = _ckpt(Path(policy))
        trained["policy"] = _trained_compat(ck)
        if ck.get("view") != "policy":
            raise StaleModel(f"{policy} 不是策略检查点（view={ck.get('view')!r}）")
        trained_with = ck.get("predictor_sha256")        # None = 训练时用的是启发式预测器
        if trained_with and (not files.get("predictor") or files["predictor"]["sha256"] != trained_with):
            raise StaleModel("策略训练时用的 GNN 预测器不在部署包里（或不是同一个文件）：候选的预测特征会对不上")
        files["policy"] = {"path": _rel(policy, out), "sha256": file_sha256(policy), "hidden": ck["config"]["hidden"],
                           "obs_spec": ck.get("obs_spec"), "task": ck.get("task"),
                           "predictor": "gnn" if trained_with else "heuristic", "ablate": ck.get("ablate", []),
                           "run_id": (ck.get("manifest") or {}).get("run_id")}
    compat = next(iter(trained.values()), compat_signature())
    if len(trained) == 2 and (d := _diff(trained["predictor"], trained["policy"])):
        raise StaleModel(f"预测器与策略是在不同的语义版本下训练的：{d}")
    if d := _diff(compat, compat_signature()):
        raise StaleModel(f"这些检查点训练时的语义版本与当前代码不一致：{d}——请重训后再打包")
    bundle = {"bundle_version": BUNDLE_VERSION, **git_state(), "compat": compat, "versions": versions(),
              "files": files, "reports": [_rel(r, out) for r in reports]}
    path = out / BUNDLE_FILE
    path.write_text(json.dumps(bundle, indent=2, ensure_ascii=False))
    return path


@dataclass
class Loaded:
    predictor: object | None      # GNNPredictor / HeuristicPredictor / None
    policy: object | None         # LearnedPolicy / None
    bundle: dict


def load_bundle(dir_: str | Path, want_predictor: bool = True, want_policy: bool = True) -> Loaded:
    """逐项核对部署包：包格式、全部语义版本、文件哈希；策略配套的预测器随包决定。任何不一致都列出来并拒绝。
    want_policy 且包里有策略：预测器取策略训练时用的那个（GNN 或启发式）；只要预测器：取包里的 GNN 文件。"""
    root = Path(dir_)
    path = root / BUNDLE_FILE
    if not path.exists():
        raise FileNotFoundError(f"{root} 下没有 {BUNDLE_FILE}：请用 python -m tianlong.learning.bundle 打包训练产物")
    b = json.loads(path.read_text())
    if b.get("bundle_version") != BUNDLE_VERSION:
        raise StaleModel(f"部署包格式 {b.get('bundle_version')!r} ≠ {BUNDLE_VERSION!r}")
    diff = _diff(b.get("compat") or {}, compat_signature())
    if diff:
        raise StaleModel(f"部署包与当前代码的语义版本不一致：{diff}——请按 README“训练与结果”重训（本仓库不做隐式迁移）")
    files = b.get("files") or {}
    for name, f in files.items():
        if file_sha256(root / f["path"]) != f["sha256"]:
            raise StaleModel(f"部署包里的 {name} 文件已被改动（哈希不符）：{f['path']}")
    pol = files.get("policy")
    if pol and pol.get("predictor") == "gnn" and "predictor" not in files:
        raise StaleModel("部署包声明策略配套 GNN 预测器，却没有带上预测器文件")
    predictor = policy = None
    if want_policy and pol:
        from tianlong.learning.rl.policy import LearnedPolicy
        policy = LearnedPolicy.load(root / pol["path"])
    use_gnn = (pol["predictor"] == "gnn") if policy is not None else (want_predictor and "predictor" in files)
    if use_gnn:
        from tianlong.learning.predictor import GNNPredictor
        predictor = GNNPredictor.load(root / files["predictor"]["path"])
    elif policy is not None:
        from tianlong.agents.predictors import HeuristicPredictor
        predictor = HeuristicPredictor()          # 策略是用启发式预测特征训练的：上线也必须用它
    return Loaded(predictor, policy, b)


def main(argv: list[str] | None = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(prog="tianlong.learning.bundle", description="把训练产物打成带清单的部署包")
    ap.add_argument("dir")
    ap.add_argument("--predictor")
    ap.add_argument("--policy")
    ap.add_argument("--reports", nargs="*", default=[])
    a = ap.parse_args(argv)
    print(write_bundle(a.dir, a.predictor, a.policy, a.reports))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
