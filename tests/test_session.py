"""
[INPUT]: 依赖 tianlong.runtime 的 GameSession / cli.main，tianlong.scenarios 的 build_warehouse
[OUTPUT]: 会话层测试：解析失败不推进时间、完整回合装配（NPC 扇出 + 结算 + 索引 + 叙述）、命令行可跑通
[POS]: tests 的装配层；验证各组件经由明确接口连接成可玩的一回合
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

from __future__ import annotations

import pytest

pytest.importorskip("langgraph")
pytest.importorskip("qdrant_client")

from tianlong.core import Op, Rel  # noqa: E402
from tianlong.memory.index import MemoryScope  # noqa: E402
from tianlong.runtime.cli import main  # noqa: E402
from tianlong.runtime.session import GameSession  # noqa: E402
from tianlong.scenarios import build_warehouse  # noqa: E402


def test_unparsed_input_does_not_advance_time():
    s = GameSession(build_warehouse())
    before = s.authority.head().version
    r = s.turn("跳个舞")
    assert not r.advanced and r.parsed.clarification
    assert s.authority.head().version == before


def test_full_turns_wire_everything():
    s = GameSession(build_warehouse())
    assert "钥匙" in s.intro()
    r1 = s.turn("拿走桌上的钥匙")
    assert r1.advanced and s.authority.head().target("key", Rel.AT) == "player"
    r2 = s.turn("等待")
    assert any(d.agent == "guard" and d.intent.op == Op.MOVE for d in r2.deliberations)
    assert set(r2.timings) == {"parse", "npc_decide", "settle", "index", "narrate"}
    assert "守卫" in r2.narration
    # 记忆已同步进向量索引，且只对守卫本人可见
    now = s.authority.head().clock
    hits = s.index.search(MemoryScope(s.ref.world_id, s.ref.branch_id, "guard", now), "响动")
    assert hits and all(h.record.owner == "guard" for h in hits)


def test_resume_rebuilds_derived_index():
    from tianlong.persistence import InMemoryWorldStore
    store = InMemoryWorldStore()
    first = GameSession(build_warehouse(), store=store)
    first.turn("拿走桌上的钥匙")
    first.turn("等待")
    again = GameSession(build_warehouse(), store=store)
    assert again.resumed and again.authority.head().version == first.authority.head().version
    now = again.authority.head().clock
    hits = again.index.search(MemoryScope(again.ref.world_id, again.ref.branch_id, "guard", now), "响动")
    assert hits, "新进程的空索引已从权威经历记录重建"


def test_dotenv_fills_only_missing(tmp_path, monkeypatch):
    from tianlong.runtime.cli import load_dotenv
    env = tmp_path / ".env"
    env.write_text("# 注释\nTL_A=from_file\nTL_B=from_file\nTL_EMPTY=\n", "utf-8")
    monkeypatch.setenv("TL_B", "from_env")
    monkeypatch.delenv("TL_A", raising=False)
    load_dotenv(env)
    import os
    assert os.environ["TL_A"] == "from_file" and os.environ["TL_B"] == "from_env" and "TL_EMPTY" not in os.environ


def test_cli_runs_scripted_input(monkeypatch, capsys):
    lines = iter(["拿走桌上的钥匙", "/beliefs", "/debug", "等待", "/quit"])
    monkeypatch.setattr("builtins.input", lambda _: next(lines))
    assert main(["--world", "warehouse", "--llm", "none"]) == 0
    out = capsys.readouterr().out
    assert "钥匙在我身上" in out and "── 真相 ──" in out
