"""
[INPUT]: 依赖标准库 subprocess / platform / importlib.metadata / datetime，core 的 digest / ATTRS_VERSION，core/goals 的 GOALS_VERSION，
         kernel 的 KERNEL_VERSION，cognition/candidates 的 CANDIDATES_VERSION，learning/schema 的 FEATURES_VERSION / SCHEMA，
         learning/rl/rewards 的 REWARD_VERSION，learning/rl/observation 的 OBS_VERSION（均不需要 ray），learning/task 的 TaskConfig
[OUTPUT]: 对外提供 git_state()、compat_signature()（全部语义版本：唯一定义）、versions()（语义版本 + 运行环境）、run_manifest()
[POS]: learning 的溯源：每份结果（动态模型报告、策略报告、检查点）都带一份 manifest——哪个提交（含工作区是否有未提交改动）、
       全部语义版本（特征规格、属性、目标、奖励、候选规则、规则内核、观测布局——与部署包比对的是同一份 compat_signature）、
       任务指纹、完整配置、训练与评测种子、依赖版本、运行设备。
       run_id = 种类 + 提交 + 工作区状态（含未提交改动的哈希）+ 配置 + 种子派生：同一提交上改了代码再跑，run_id 也不同。
       README 的每一个数字都应能按 run_id 找回这份机器可读记录；拿不到提交号时如实写 None，不猜
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

from __future__ import annotations

import hashlib
import os
import platform
import subprocess
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path

from tianlong.cognition.candidates import CANDIDATES_VERSION
from tianlong.core import ATTRS_VERSION, digest
from tianlong.core.goals import GOALS_VERSION
from tianlong.kernel import KERNEL_VERSION
from tianlong.learning.rl.observation import OBS_VERSION
from tianlong.learning.rl.rewards import REWARD_VERSION
from tianlong.learning.schema import FEATURES_VERSION, SCHEMA
from tianlong.learning.task import TaskConfig

_DEPS = ("torch", "torch-geometric", "ray", "numpy", "gymnasium", "langgraph", "neo4j", "qdrant-client")


def git_state(repo: Path | None = None) -> dict:
    """提交号、工作区是否干净、以及未提交改动的哈希（同一提交上改了代码再跑，run_id 也不同）；
    不在 git 仓库里时读 TIANLONG_GIT_SHA（从固定提交安装时由调用方设置）。"""
    cwd = repo or Path(__file__).resolve().parents[3]

    def git(*args: str) -> str:
        return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=True,
                              timeout=10).stdout

    try:
        sha = git("rev-parse", "HEAD").strip()
        dirty = bool(git("status", "--porcelain", "--untracked-files=no").strip())
        diff = hashlib.sha256(git("diff", "HEAD").encode()).hexdigest()[:16] if dirty else None
        return {"git_sha": sha, "git_dirty": dirty, "git_diff_sha256": diff}
    except (OSError, subprocess.SubprocessError):
        return {"git_sha": os.environ.get("TIANLONG_GIT_SHA"), "git_dirty": None, "git_diff_sha256": None}


def compat_signature() -> dict:
    """当前代码的全部语义版本：模型在它们全都一致时才能直接用（部署包逐项比对），manifest 也原样记下。"""
    return {"schema": SCHEMA, "features": FEATURES_VERSION, "attributes": ATTRS_VERSION, "goals": GOALS_VERSION,
            "reward": REWARD_VERSION, "candidates": CANDIDATES_VERSION, "kernel": KERNEL_VERSION,
            "observation": OBS_VERSION}


def versions() -> dict:
    deps = {}
    for d in _DEPS:
        try:
            deps[d] = metadata.version(d)
        except metadata.PackageNotFoundError:
            deps[d] = None
    return {**compat_signature(), "python": platform.python_version(), "deps": deps}


def run_manifest(kind: str, config: dict, task: TaskConfig | None = None, seeds: dict | None = None,
                 extra: dict | None = None) -> dict:
    git = git_state()
    seeds = dict(seeds or {})
    return {
        "run_id": digest(kind, git["git_sha"], git["git_dirty"], git["git_diff_sha256"], sorted(config.items()),
                         sorted(seeds.items()))[:12],
        "kind": kind,
        **git,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "versions": versions(),
        "task": task.to_dict() if task else None,
        "task_fingerprint": task.fingerprint() if task else None,
        "config": config,
        "seeds": seeds,
        **(extra or {}),
    }
