"""
[INPUT]: 依赖标准库 json / argparse / pathlib
[OUTPUT]: 对外提供 cells()（笔记本单元格：(类型, 源码) 列表）、build()（组装 nbformat 4 文档）、main()
          （python scripts/make_colab_notebook.py [--commit SHA] [--out 路径]）
[POS]: scripts 的笔记本生成器，notebooks/train_colab.ipynb 的唯一源头：单元格在这里以 Python 字符串维护（可审阅、可测试），
       仓库里的 .ipynb 是它不带提交号的产物（tests/test_notebook.py 核对两者一致）；交给 Colab 跑的那一份用 --commit 填入固定提交。
       笔记本只是外壳：检出固定提交 → 安装（按该提交的 constraints.txt 锁定版本）→ 全量测试（失败即停）→ 剖析 → 调用仓库里同一套训练 CLI → 结果写进 Drive
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

PLACEHOLDER = "在此填入要复现的 40 位提交号"

# ============================================================
#  单元格
# ============================================================

_INTRO = """
# RealTianLong · Colab 训练与评测

笔记本只是外壳：**检出固定提交 → 安装（版本锁定）→ 全量测试（失败即停）→ 剖析 → 调用仓库里同一套训练 CLI → 结果写进 Google Drive**。
训练逻辑只在 `src/tianlong/learning` 里一处。

1. 菜单 **代码执行程序 → 更改运行时类型 → GPU**，优先选 **A100**。决定速度的主要是 **CPU 核数**：世界模拟、认知折叠、GNN 预测、
   评测与示范都在 CPU 上按世界/按局并行，GPU 只加速学习器与 GNN 训练。A100 运行时（约 12 核）全流程约 1.5–2 小时，
   T4 标准运行时（2 核）约 8–10 小时。下面剖析格会打印本机核数与各阶段耗时。
2. 核对第一个代码格里的 `COMMIT`（固定提交号）与 `SCALE`（`smoke` 约十五分钟冒烟 / `full` 正式运行）。
3. **代码执行程序 → 全部运行**。任何一步失败都会抛异常，后续单元格不再执行。
4. 产物（日志、报告 JSON、检查点、依赖锁定、剖析、结果表、部署包）都写在 Drive 的 `MyDrive/RealTianLong/runs/<运行名>/`。
   断线后重新“全部运行”：已完成的阶段跳过，GNN 按轮、PPO 按最近一次保存的权重接着训练。
5. 跑完后把运行名告诉 Claude（或把 `RESULTS.md` 贴回来）。结果表只从报告生成，README 只引用它。

代码来源：优先用 Drive 上的 `MyDrive/RealTianLong/code/realtianlong.bundle`（完整 git 历史，不需要 GitHub 权限）；
没有就从 GitHub 克隆。私有仓库可在左侧 🔑 Secrets 放 `GH_TOKEN`：令牌只经环境变量注入一次性请求头，
不进命令行、不进 `.git/config`、不进日志（最后一格会扫描运行目录确认）。

> 训练产物只是**意图的来源**：GNN 预测与 RL 选择都必须经内核结算才成为事实。
"""

_CONFIG = """
# ==== 运行配置 ====
REPO_URL = "https://github.com/nekoduck/RealTianLong.git"
COMMIT = "{commit}"   # 固定提交：可复现
SCALE = "full"                                                 # "smoke"：冒烟；"full"：正式运行

BUNDLE = "/content/drive/MyDrive/RealTianLong/code/realtianlong.bundle"   # 完整 git 历史（可选）
WORKDIR = "/content/RealTianLong"                              # 绝对路径：从不嵌套克隆
DRIVE_ROOT = "/content/drive/MyDrive/RealTianLong/runs"
RUN_NAME = f"{{COMMIT[:10]}}-{{SCALE}}"
assert len(COMMIT) == 40 and all(c in "0123456789abcdef" for c in COMMIT), "COMMIT 必须是完整的 40 位提交号"
assert SCALE in ("smoke", "full"), SCALE
"""

_MOUNT = """
# ==== 挂载 Google Drive ====
from google.colab import drive
drive.mount("/content/drive")
"""

_HELPERS = """
# ==== 工具函数：sh() 非零即抛异常（“全部运行”就此停下），输出同时写进运行目录的日志，已知秘密一律打码 ====
import base64, json, os, pathlib, shlex, subprocess, sys, time

RUN = pathlib.Path(DRIVE_ROOT) / RUN_NAME
(RUN / "logs").mkdir(parents=True, exist_ok=True)
SECRETS = []            # 令牌及其编码：任何输出与日志里出现都替换成 ***


def redact(text):
    for s in SECRETS:
        text = text.replace(s, "***")
    return text


def sh(cmd, log, cwd=None, env=None, check=True):
    args = [str(a) for a in (cmd if isinstance(cmd, list) else shlex.split(cmd))]
    line = redact("$ " + " ".join(args))
    print(line, flush=True)
    t0 = time.time()
    with open(RUN / "logs" / f"{log}.log", "a") as f:
        f.write(line + "\\n")
        with subprocess.Popen(args, cwd=cwd, env={**os.environ, **(env or {})}, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True, bufsize=1) as p:
            for raw in p.stdout:
                text = redact(raw)
                print(text, end="")
                f.write(text)
    if check and p.returncode != 0:
        raise RuntimeError(f"命令失败（退出码 {p.returncode}，{time.time() - t0:.0f}s，日志 logs/{log}.log）：{line}")
    return p.returncode


def sh_all(jobs):
    \"\"\"并行跑多条互不依赖的命令（各写各的日志）；全部结束后，任何一条非零都抛异常。\"\"\"
    procs = []
    for cmd, log in jobs:
        args = [str(a) for a in cmd]
        line = redact("$ " + " ".join(args))
        print(line, flush=True)
        f = open(RUN / "logs" / f"{log}.log", "a")
        f.write(line + "\\n")
        f.flush()
        procs.append((subprocess.Popen(args, stdout=f, stderr=subprocess.STDOUT), f, log))
    failed = []
    for p, f, log in procs:
        p.wait()
        f.close()
        path = RUN / "logs" / f"{log}.log"
        if SECRETS:
            path.write_text(redact(path.read_text()))
        if p.returncode != 0:
            failed.append(f"{log}（退出码 {p.returncode}）")
            print(f"---- logs/{log}.log 末尾 ----\\n" + "\\n".join(path.read_text().splitlines()[-30:]))
    if failed:
        raise RuntimeError("并行命令失败：" + "；".join(failed))


def out(cmd, cwd=None):
    r = subprocess.run(shlex.split(cmd), cwd=cwd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"命令失败：{cmd}\\n{redact(r.stderr)}")
    return r.stdout.strip()


print("运行目录：", RUN)
"""

_CHECKOUT = """
# ==== 检出固定提交：已有目录必须是干净的 git 仓库；从不嵌套克隆、从不静默用旧提交；remote 与 git 配置里不留凭据 ====
def github_auth_env():
    \"\"\"私有仓库：令牌只经环境变量注入一次性的 HTTP 头——不进命令行、不进 .git/config、不进日志。\"\"\"
    try:
        from google.colab import userdata
        token = userdata.get("GH_TOKEN")
    except Exception:            # 没有这个 Secret 或未授权本笔记本：按公开仓库处理
        return {}
    if not token:
        return {}
    basic = base64.b64encode(f"x-access-token:{token}".encode()).decode()
    SECRETS.extend([token, basic])
    return {"GIT_CONFIG_COUNT": "1", "GIT_CONFIG_KEY_0": "http.https://github.com/.extraheader",
            "GIT_CONFIG_VALUE_0": f"AUTHORIZATION: basic {basic}"}


GIT_ENV = {"GIT_TERMINAL_PROMPT": "0", **github_auth_env()}


def have_commit(sha):
    return subprocess.run(["git", "cat-file", "-e", f"{sha}^{{commit}}"], cwd=WORKDIR, capture_output=True).returncode == 0


if os.path.exists(WORKDIR):
    if not os.path.isdir(os.path.join(WORKDIR, ".git")):
        raise RuntimeError(f"{WORKDIR} 已存在但不是 git 仓库：请删除或改 WORKDIR，笔记本不会覆盖它")
    if out("git status --porcelain --untracked-files=no", cwd=WORKDIR):
        raise RuntimeError(f"{WORKDIR} 有未提交的改动：请先处理，笔记本不会覆盖它们")
elif os.path.exists(BUNDLE):
    sh(["git", "clone", "--quiet", "--no-checkout", BUNDLE, WORKDIR], log="git")
    sh(["git", "remote", "set-url", "origin", REPO_URL], cwd=WORKDIR, log="git")
else:
    sh(["git", "clone", "--quiet", "--no-checkout", REPO_URL, WORKDIR], env=GIT_ENV, log="git")

if not have_commit(COMMIT):
    sh(["git", "fetch", "--quiet", "origin", COMMIT], cwd=WORKDIR, env=GIT_ENV, log="git", check=False)
if not have_commit(COMMIT) and os.path.exists(BUNDLE):
    sh(["git", "fetch", "--quiet", BUNDLE, "+refs/heads/*:refs/remotes/bundle/*"], cwd=WORKDIR, log="git")
if not have_commit(COMMIT):
    raise RuntimeError(f"拿不到提交 {COMMIT}：GitHub 上没有，Drive 上的 bundle 里也没有")
sh(["git", "checkout", "--quiet", "--detach", COMMIT], cwd=WORKDIR, log="git")
head = out("git rev-parse HEAD", cwd=WORKDIR)
assert head == COMMIT, (head, COMMIT)
remote = out("git remote get-url origin", cwd=WORKDIR)
git_config = pathlib.Path(WORKDIR, ".git", "config").read_text()
assert "@" not in remote and "extraheader" not in git_config.lower() and not any(s in git_config for s in SECRETS), \\
    "remote 或 git 配置里残留了凭据"
os.chdir(WORKDIR)
print("检出", head, "| remote", remote)
"""

_INSTALL_MD = """
## 安装与测试
依赖版本按检出提交里的 `constraints.txt` 锁定（与本地、CI 验证过的一致；torch 用 Colab 自带的 CUDA 版），实际安装结果写进运行目录；随后跑全量测试（需要 Neo4j 的用例在没有数据库时自动跳过）。**测试失败即停**，不带着坏代码去训练。
"""

_INSTALL = """
# ==== 安装（版本锁定）与全量测试：失败即停 ====
sh([sys.executable, "-m", "pip", "install", "-q", "-c", "constraints.txt", "-e", ".[all,dev]"], log="pip")
import torch, torch_geometric, ray
print("torch", torch.__version__, "| cuda", torch.cuda.is_available(), "| pyg", torch_geometric.__version__,
      "| ray", ray.__version__)
(RUN / "requirements.lock").write_text(out(f"{sys.executable} -m pip freeze"))
CPUS = os.cpu_count() or 2
print("CPU 核数", CPUS, "| GPU", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
(RUN / "env.json").write_text(json.dumps({"commit": COMMIT, "scale": SCALE, "python": sys.version, "cpus": CPUS,
    "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}, indent=1))
sh([sys.executable, "-m", "pytest", "-q", "-x", "-p", "no:cacheprovider"], log="pytest")
"""

_PROFILE = """
# ==== 剖析：先看清时间花在哪一段（设备写进报告）====
if not (RUN / "profile.json").exists():
    sh([sys.executable, "-m", "tianlong.learning.profile", "--worlds", "10" if SCALE == "smoke" else "30",
        "--out", RUN / "profile.json"], log="profile")
for k, v in json.loads((RUN / "profile.json").read_text())["stages"].items():
    print(f"{k:28s} {v['mean_ms']:9.3f} ms × {v['calls']}")
"""

_GNN_MD = """
## 阶段 B：GNN 动态模型（环境视角 / 角色视角）
按世界三分 训练 / 校准 / 测试：每轮在校准世界上算损失、取最低的一轮，温度也只在校准世界拟合；测试世界只做最终报告。
`--scroll-held` 与 `--hide-goal-items` 提高修习与“查看才有发现”在数据里的覆盖（报告的 coverage 里可见）。
内存 ≥ 24 GB 时两个视角同时训练（GPU 共用，数据生成各占一半 CPU 核并按世界并行），否则逐个训练；输出在 `logs/gnn_env.log`、`logs/gnn_agent.log`。
断线后重跑本格：已完成的视角跳过，未完成的从 `dynamics_<视角>.resume.pt` 按轮接着训练。
"""

_GNN = """
GNN = RUN / "gnn"
GNN.mkdir(exist_ok=True)
B = dict(smoke=dict(worlds=60, epochs=3, hidden=32), full=dict(worlds=3000, epochs=40, hidden=128))[SCALE]
todo = [v for v in ("env", "agent") if not (GNN / f"dynamics_{v}.json").exists()]
print("已完成，跳过：", [v for v in ("env", "agent") if v not in todo])
RAM_GB = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30
together = len(todo) > 1 and RAM_GB >= 24          # 正式规模每个视角的数据约 3 GB：内存够才两个视角同时训练
jobs = [([sys.executable, "-m", "tianlong.learning.train", "--view", v, "--worlds", B["worlds"], "--epochs", B["epochs"],
          "--hidden", B["hidden"], "--jianghu", "0.5", "--scroll-held", "0.3", "--hide-goal-items", "0.3", "--seed", "0",
          "--workers", max(1, CPUS // (len(todo) if together else 1)), "--resume", "true", "--out", GNN], f"gnn_{v}")
        for v in todo]
print(f"内存 {RAM_GB:.0f} GB：", "两个视角同时训练" if together else "逐个视角训练")
for group in ([jobs] if together else [[j] for j in jobs]):
    sh_all(group)
for view in ("env", "agent"):
    m = json.loads((GNN / f"dynamics_{view}.json").read_text())
    print(view, {k: m.get(k) for k in ("holder_changed_recall", "holder_unchanged_kept", "success_brier",
                                       "success_brier_calibrated", "attr_changed_acc", "obs_gain_mae")},
          "best_epoch", m["model_selection"]["best_epoch"])
"""

_RL_MD = """
## 阶段 C：角色策略（模仿学习 → PPO → 留出世界评测）
* 主实验：三个训练种子，以角色视角 GNN 的预测为候选特征。
* 训练期消融：同一设置，但策略从头到尾看不到预测 / 看不到长期记忆列（与测试期置零分开解释）。
* E04 探查任务：目标物品一律藏起（`--hide-goal-items 1`），有 / 无预测对照。

所有运行评测同一批留出世界，报告带逐世界记录：消融与主实验在结果表里按世界配对比较（`results --pair`）。

每个运行独占全部 CPU 核：PPO 用 核数−1 个采样进程，评测与示范按局分给全部核（与顺序执行逐项相同）；训练期消融了预测的运行不算预测，快得多。
每个运行写 `policy_ppo*_s{种子}.json/.pt`；已完成的跳过，PPO 断线后从最近一次保存的权重接着训练。
"""

_RL = """
RL = RUN / "rl"
RL.mkdir(exist_ok=True)
C = dict(smoke=dict(demo=6, bc=1, it=1, batch=400, ev=20, hidden=32),
         full=dict(demo=300, bc=4, it=30, batch=3000, ev=100, hidden=64))[SCALE]
PRED = GNN / "dynamics_agent.pt"
RUNS = [dict(tag=f"main_s{s}", seed=s, extra=[]) for s in (0, 1, 2)] + [
    dict(tag="noPred_s0", seed=0, extra=["--ablate-predictions", "true"]),
    dict(tag="noMem_s0", seed=0, extra=["--ablate-memory", "true"]),
    dict(tag="probe_s0", seed=0, extra=["--hide-goal-items", "1.0"]),
    dict(tag="probe_noPred_s0", seed=0, extra=["--hide-goal-items", "1.0", "--ablate-predictions", "true"]),
]
if SCALE == "smoke":
    RUNS = [r for r in RUNS if r["tag"] in ("main_s0", "noPred_s0")]
RUNNERS = max(1, CPUS - 1)
for r in RUNS:
    out_dir = RL / r["tag"]
    if list(out_dir.glob("policy_ppo*.json")):
        print(r["tag"], "已完成，跳过")
        continue
    out_dir.mkdir(exist_ok=True)
    sh([sys.executable, "-m", "tianlong.learning.rl.train", "--seed", r["seed"], "--jianghu", "0.5",
        "--demo-episodes", C["demo"], "--bc-epochs", C["bc"], "--bc-smoothing", "0.1", "--entropy", "0.03",
        "--ppo-iterations", C["it"], "--train-batch", C["batch"], "--eval-episodes", C["ev"], "--hidden", C["hidden"],
        "--env-runners", RUNNERS, "--workers", CPUS, "--gpus", "1" if torch.cuda.is_available() else "0",
        "--predictor-path", PRED,
        "--resume", "true", "--out", out_dir, *r["extra"]], log=f"rl_{r['tag']}")
"""

_RESULTS_MD = """
## 结果与部署包
结果表只从报告生成（每张表写明 run_id、提交号、任务指纹与种子；跨训练种子另给均值与标准差）。
部署包 `bundle.json` 记下训练时的全部语义版本与文件哈希；本地把整个运行目录拷下来，
`python -m tianlong --predictor gnn --policy learned --artifacts <运行目录>` 加载前逐项核对。
"""

_RESULTS = """
reports = sorted([*GNN.glob("dynamics_*.json"), *RL.glob("*/policy_ppo*.json")])
PAIRS = [("main_s0", "noPred_s0"), ("main_s0", "noMem_s0"), ("probe_s0", "probe_noPred_s0")]
pair_args = []
for a, b in PAIRS:
    ra, rb = sorted((RL / a).glob("policy_ppo*.json")), sorted((RL / b).glob("policy_ppo*.json"))
    if ra and rb:
        pair_args += ["--pair", ra[0], rb[0]]
sh([sys.executable, "-m", "tianlong.learning.results", *reports, *pair_args, "--out", RUN / "RESULTS.md"],
   log="results")
main = sorted(RL.glob("main_s0/policy_ppo_s0.pt"))
if main:
    sh([sys.executable, "-m", "tianlong.learning.bundle", RUN, "--predictor", PRED, "--policy", main[0],
        "--reports", *reports], log="bundle")
# 交付物里不得出现凭据
leaks = [p for p in RUN.rglob("*") if p.is_file() and p.suffix in (".log", ".json", ".md", ".lock")
         and any(s in p.read_text(errors="ignore") for s in SECRETS)]
assert not leaks, f"这些文件里出现了凭据：{leaks}"
print((RUN / "RESULTS.md").read_text()[:4000])
print("全部完成。把运行名告诉 Claude：", RUN_NAME)
"""


def cells(commit: str = PLACEHOLDER) -> list[tuple[str, str]]:
    return [
        ("markdown", _INTRO), ("code", _CONFIG.format(commit=commit)), ("code", _MOUNT), ("code", _HELPERS),
        ("code", _CHECKOUT), ("markdown", _INSTALL_MD), ("code", _INSTALL),
        ("code", _PROFILE), ("markdown", _GNN_MD), ("code", _GNN), ("markdown", _RL_MD), ("code", _RL),
        ("markdown", _RESULTS_MD), ("code", _RESULTS),
    ]


def build(commit: str = PLACEHOLDER) -> dict:
    out = []
    for kind, src in cells(commit):
        lines = src.strip("\n").split("\n")
        cell = {"cell_type": kind, "metadata": {}, "source": [ln + "\n" for ln in lines[:-1]] + lines[-1:]}
        if kind == "code":
            cell.update(execution_count=None, outputs=[])
        out.append(cell)
    return {"cells": out, "metadata": {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "T4"},
                                       "kernelspec": {"display_name": "Python 3", "name": "python3"},
                                       "language_info": {"name": "python"}},
            "nbformat": 4, "nbformat_minor": 0}


def render(commit: str = PLACEHOLDER) -> str:
    return json.dumps(build(commit), ensure_ascii=False, indent=1) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="生成 Colab 训练笔记本")
    ap.add_argument("--commit", default=PLACEHOLDER, help="填入固定提交（交给 Colab 的那一份）；仓库里的版本不填")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "notebooks" / "train_colab.ipynb"))
    a = ap.parse_args(argv)
    Path(a.out).write_text(render(a.commit))
    print("wrote", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
