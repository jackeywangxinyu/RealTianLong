"""
[INPUT]: 依赖标准库 subprocess / platform / importlib.metadata / datetime，core 的 digest / ATTRS_VERSION，core/goals 的 GOALS_VERSION，
         learning/schema 的 FEATURES_VERSION / SCHEMA，learning/task 的 TaskConfig
[OUTPUT]: 对外提供 git_state()、versions()、run_manifest()
[POS]: learning 的溯源：每份结果（动态模型报告、策略报告、检查点）都带一份 manifest——哪个提交（含工作区是否有未提交改动）、
       特征/属性/目标/奖励/任务的版本与指纹、完整配置、训练与评测种子、依赖版本、运行设备。
       README 的每一个数字都应能按 run_id 找回这份机器可读记录；拿不到提交号时如实写 None，不猜
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

from __future__ import annotations

import os
import platform
import subprocess
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path

from tianlong.core import ATTRS_VERSION, digest
from tianlong.core.goals import GOALS_VERSION
from tianlong.learning.schema import FEATURES_VERSION, SCHEMA
from tianlong.learning.task import TaskConfig

_DEPS = ("torch", "torch-geometric", "ray", "numpy", "gymnasium", "langgraph", "neo4j", "qdrant-client")


def git_state(repo: Path | None = None) -> dict:
    """提交号与工作区是否干净；不在 git 仓库里时读 TIANLONG_GIT_SHA（Colab 从固定提交安装时由笔记本设置）。"""
    cwd = repo or Path(__file__).resolve().parents[3]
    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=cwd, capture_output=True, text=True, check=True,
                             timeout=10).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=cwd,
                                    capture_output=True, text=True, check=True, timeout=10).stdout.strip())
        return {"git_sha": sha, "git_dirty": dirty}
    except (OSError, subprocess.SubprocessError):
        return {"git_sha": os.environ.get("TIANLONG_GIT_SHA"), "git_dirty": None}


def versions() -> dict:
    deps = {}
    for d in _DEPS:
        try:
            deps[d] = metadata.version(d)
        except metadata.PackageNotFoundError:
            deps[d] = None
    try:
        from tianlong.learning.rl.rewards import REWARD_VERSION
    except ImportError:          # 没装 ray 时奖励模块不可用：如实缺省
        REWARD_VERSION = None    # noqa: N806
    return {"features": FEATURES_VERSION, "schema": SCHEMA, "attributes": ATTRS_VERSION, "goals": GOALS_VERSION,
            "reward": REWARD_VERSION, "python": platform.python_version(), "deps": deps}


def run_manifest(kind: str, config: dict, task: TaskConfig | None = None, seeds: dict | None = None,
                 extra: dict | None = None) -> dict:
    git = git_state()
    seeds = dict(seeds or {})
    return {
        "run_id": digest(kind, git["git_sha"], sorted(config.items()), sorted(seeds.items()))[:12],
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
