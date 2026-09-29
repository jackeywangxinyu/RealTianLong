"""
[INPUT]: 依赖 scripts/make_colab_notebook 的 cells() / render() / PLACEHOLDER，notebooks/train_colab.ipynb，本机 git，
         各训练 CLI 的 main(["--help"])
[OUTPUT]: Colab 笔记本验收：仓库里的 .ipynb 与生成器一致；C03（重复运行检出单元格不嵌套克隆、明确检出指定 SHA、
          已有目录不是干净仓库即停下）；C04（sh() 遇非零退出即抛异常，“全部运行”就此停下）；C05（令牌经 Secrets 注入时
          remote、.git/config 与日志里都不含令牌）；笔记本用到的每个 CLI 参数都真实存在（防止笔记本与 CLI 漂移）
[POS]: tests 的笔记本层：把“笔记本单元格”当代码执行（在本地临时目录里模拟 Drive 与仓库），而不是只做字符串检查
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import re
import subprocess
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("make_colab_notebook", ROOT / "scripts" / "make_colab_notebook.py")
gen = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gen)


def _cell(marker: str, commit: str = gen.PLACEHOLDER) -> str:
    hits = [src for kind, src in gen.cells(commit) if kind == "code" and marker in src]
    assert len(hits) == 1, marker
    return hits[0]


def _git(*args, cwd):
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()


@pytest.fixture
def source_repo(tmp_path):
    """一个带两次提交的上游仓库，以及它的完整 git bundle（相当于 Drive 上的 realtianlong.bundle）。"""
    src = tmp_path / "upstream"
    src.mkdir()
    _git("init", "-q", "-b", "main", cwd=src)
    _git("config", "user.email", "t@example.com", cwd=src)
    _git("config", "user.name", "t", cwd=src)
    for i in range(2):
        (src / "f.txt").write_text(str(i))
        _git("add", ".", cwd=src)
        _git("commit", "-q", "-m", f"c{i}", cwd=src)
    shas = _git("rev-list", "--reverse", "HEAD", cwd=src).split()
    bundle = tmp_path / "realtianlong.bundle"
    _git("bundle", "create", str(bundle), "main", cwd=src)
    return src, bundle, shas


def _run_cells(ns: dict, *markers: str, commit: str) -> dict:
    for m in markers:
        exec(compile(_cell(m, commit), m, "exec"), ns)
    return ns


def _ns(tmp_path, src, bundle, commit, **over) -> dict:
    ns = {"REPO_URL": str(src), "COMMIT": commit, "SCALE": "smoke", "BUNDLE": str(bundle),
          "WORKDIR": str(tmp_path / "content" / "RealTianLong"), "DRIVE_ROOT": str(tmp_path / "drive" / "runs"),
          "RUN_NAME": f"{commit[:10]}-smoke"}
    ns.update(over)
    return ns


# ============================================================
#  生成器与仓库里的笔记本一致
# ============================================================


def test_committed_notebook_is_the_generator_output():
    assert (ROOT / "notebooks" / "train_colab.ipynb").read_text() == gen.render(), \
        "notebooks/train_colab.ipynb 与生成器不一致：请运行 python scripts/make_colab_notebook.py"


def test_every_code_cell_compiles_and_has_no_shell_magics():
    nb = json.loads(gen.render("0" * 40))
    for c in nb["cells"]:
        if c["cell_type"] != "code":
            continue
        src = "".join(c["source"])
        compile(src, "cell", "exec")
        # `!cmd` / `%cd` 的失败不会中断“全部运行”，也会在相对路径下嵌套克隆：一律不用
        assert not any(ln.lstrip().startswith(("!", "%")) for ln in src.splitlines()), src[:80]


def test_config_cell_refuses_a_missing_commit():
    with pytest.raises(AssertionError, match="40 位"):
        exec(_cell("==== 运行配置"), {})
    ns: dict = {}
    exec(_cell("==== 运行配置", "a" * 40), ns)
    assert ns["RUN_NAME"] == "aaaaaaaaaa-full" and ns["WORKDIR"].startswith("/")


# ============================================================
#  C04：失败即停
# ============================================================


def test_c04_sh_raises_on_nonzero_exit_and_logs(tmp_path):
    ns = _run_cells({"DRIVE_ROOT": str(tmp_path), "RUN_NAME": "r"}, "==== 工具函数", commit=gen.PLACEHOLDER)
    with contextlib.redirect_stdout(io.StringIO()), pytest.raises(RuntimeError, match="退出码 3"):
        ns["sh"]([sys.executable, "-c", "print('boom'); raise SystemExit(3)"], log="step")
    assert "boom" in (tmp_path / "r" / "logs" / "step.log").read_text()
    with contextlib.redirect_stdout(io.StringIO()):
        assert ns["sh"]([sys.executable, "-c", "raise SystemExit(0)"], log="ok") == 0
    with pytest.raises(RuntimeError):
        ns["out"]("git definitely-not-a-command")


def test_c04_install_and_test_steps_go_through_the_failing_sh():
    install = _cell("==== 安装")
    assert 'sh([sys.executable, "-m", "pip", "install"' in install
    assert 'sh([sys.executable, "-m", "pytest"' in install and "check=False" not in install


# ============================================================
#  C03：可重跑、明确的提交、不嵌套克隆
# ============================================================


def test_c03_checkout_uses_the_pinned_sha_and_is_rerunnable(tmp_path, source_repo, monkeypatch):
    src, bundle, (first, second) = source_repo
    monkeypatch.chdir(tmp_path)
    ns = _ns(tmp_path, tmp_path / "no-such-remote", bundle, first)
    with contextlib.redirect_stdout(io.StringIO()):
        _run_cells(ns, "==== 工具函数", "==== 检出固定提交", commit=first)
    work = Path(ns["WORKDIR"])
    assert _git("rev-parse", "HEAD", cwd=work) == first            # 指定的旧提交，而不是分支最新的 second
    assert Path.cwd() == work
    monkeypatch.chdir(tmp_path)
    with contextlib.redirect_stdout(io.StringIO()):                 # 再跑一遍：复用已有仓库，不嵌套
        _run_cells(ns, "==== 检出固定提交", commit=first)
    assert not (work / "RealTianLong").exists()
    assert _git("rev-parse", "HEAD", cwd=work) == first
    ns["COMMIT"] = second                                            # 换提交重跑：切到新提交
    monkeypatch.chdir(tmp_path)
    with contextlib.redirect_stdout(io.StringIO()):
        _run_cells(ns, "==== 检出固定提交", commit=second)
    assert _git("rev-parse", "HEAD", cwd=work) == second


def test_c03_unknown_commit_stops_instead_of_using_whatever_is_there(tmp_path, source_repo, monkeypatch):
    src, bundle, _ = source_repo
    monkeypatch.chdir(tmp_path)
    ns = _ns(tmp_path, src, bundle, "f" * 40)
    with contextlib.redirect_stdout(io.StringIO()), pytest.raises(RuntimeError, match="拿不到提交"):
        _run_cells(ns, "==== 工具函数", "==== 检出固定提交", commit="f" * 40)


def test_c03_existing_dirty_or_foreign_dir_is_never_overwritten(tmp_path, source_repo, monkeypatch):
    src, bundle, (first, _) = source_repo
    monkeypatch.chdir(tmp_path)
    ns = _ns(tmp_path, src, bundle, first)
    work = Path(ns["WORKDIR"])
    work.mkdir(parents=True)
    (work / "notes.txt").write_text("mine")
    with contextlib.redirect_stdout(io.StringIO()), pytest.raises(RuntimeError, match="不是 git 仓库"):
        _run_cells(ns, "==== 工具函数", "==== 检出固定提交", commit=first)
    (work / "notes.txt").unlink()
    work.rmdir()
    with contextlib.redirect_stdout(io.StringIO()):
        _run_cells(ns, "==== 检出固定提交", commit=first)
    (work / "f.txt").write_text("local edit")
    monkeypatch.chdir(tmp_path)
    with contextlib.redirect_stdout(io.StringIO()), pytest.raises(RuntimeError, match="未提交的改动"):
        _run_cells(ns, "==== 检出固定提交", commit=first)
    assert (work / "f.txt").read_text() == "local edit"


# ============================================================
#  C05：凭据不残留
# ============================================================


def test_c05_token_never_reaches_remote_git_config_or_logs(tmp_path, source_repo, monkeypatch):
    src, bundle, (_, second) = source_repo
    token = "ghp_TESTTOKEN_should_never_leak_123"
    colab = types.ModuleType("google.colab")
    colab.userdata = types.SimpleNamespace(get=lambda name: token if name == "GH_TOKEN" else None)
    google = types.ModuleType("google")
    google.colab = colab
    monkeypatch.setitem(sys.modules, "google", google)
    monkeypatch.setitem(sys.modules, "google.colab", colab)
    monkeypatch.chdir(tmp_path)
    ns = _ns(tmp_path, src, tmp_path / "no-bundle", second)         # 没有 bundle：走“克隆远端”的路径
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        _run_cells(ns, "==== 工具函数", "==== 检出固定提交", commit=second)
    work = Path(ns["WORKDIR"])
    assert _git("rev-parse", "HEAD", cwd=work) == second
    assert token in ns["SECRETS"] and "GIT_CONFIG_VALUE_0" in ns["GIT_ENV"]      # 令牌确实被用上了
    assert token not in _git("remote", "get-url", "origin", cwd=work)
    assert token not in (work / ".git" / "config").read_text()
    for f in (Path(ns["DRIVE_ROOT"]) / ns["RUN_NAME"] / "logs").iterdir():
        assert token not in f.read_text()
    assert token not in buf.getvalue()
    # 日志与屏幕输出里偶然出现的令牌会被打码
    with contextlib.redirect_stdout(io.StringIO()) as echo:
        ns["sh"]([sys.executable, "-c", f"print('leak {token}')"], log="echo")
    assert token not in echo.getvalue() and "***" in echo.getvalue()
    assert token not in (Path(ns["DRIVE_ROOT"]) / ns["RUN_NAME"] / "logs" / "echo.log").read_text()


def test_c05_results_cell_scans_deliverables_for_secrets():
    assert "assert not leaks" in _cell("tianlong.learning.results")


# ============================================================
#  笔记本与 CLI 不漂移
# ============================================================


def _flags(main) -> set[str]:
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), pytest.raises(SystemExit):
        main(["--help"])
    return set(re.findall(r"--[a-z][a-z0-9-]*", buf.getvalue()))


def test_every_cli_flag_the_notebook_passes_exists():
    pytest.importorskip("torch_geometric")
    pytest.importorskip("ray.rllib")
    from tianlong.learning import bundle, profile, results, train
    from tianlong.learning.rl import train as rl_train
    known = {"tianlong.learning.train": _flags(train.main), "tianlong.learning.rl.train": _flags(rl_train.main),
             "tianlong.learning.profile": _flags(profile.main), "tianlong.learning.results": _flags(results.main),
             "tianlong.learning.bundle": _flags(bundle.main)}
    for kind, src in gen.cells("0" * 40):
        if kind != "code":
            continue
        for mod, flags in known.items():
            for call in re.findall(rf'"-m", "{re.escape(mod)}"(.*?)log=', src, flags=re.S):
                used = set(re.findall(r'"(--[a-z][a-z0-9-]*)"', call))
                assert used <= flags, (mod, used - flags)
    rl_cell = _cell("tianlong.learning.rl.train")
    for extra in re.findall(r'"(--[a-z][a-z0-9-]*)"', rl_cell.split("RUNS = ")[1].split("]\nif")[0]):
        assert extra in known["tianlong.learning.rl.train"], extra
    for extra in re.findall(r'"(--[a-z][a-z0-9-]*)"', _cell("tianlong.learning.results").split("sh(")[0]):
        assert extra in known["tianlong.learning.results"], extra          # 先拼好再传的参数（如 --pair）
