"""
[INPUT]: 依赖 tianlong.runtime.suggest 的 suggestions，tianlong.runtime.session 的 GameSession，tianlong.language.parser 的 MoveKind，
         tianlong.scenarios 的 build_wuliang
[OUTPUT]: 行动建议验收：至多三句、互不重复、同一认知同一建议；每一句都被解释器听懂（不是“没听懂”），行动类不经模型；
          照着建议一路点下去，建议里从不出现玩家不认识的名字；有人冲我来就先给赔罪与脱身；研读过的不再提
[POS]: tests 的行动建议：证伪“建议点了却听不懂”“建议泄露了玩家不知道的人和物”“挨了打还劝人去研读”
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

from __future__ import annotations

import pytest

pytest.importorskip("langgraph")
pytest.importorskip("qdrant_client")

from tianlong.language.parser import MoveKind  # noqa: E402
from tianlong.runtime.session import GameSession  # noqa: E402
from tianlong.runtime.suggest import suggestions  # noqa: E402
from tianlong.scenarios import build_wuliang  # noqa: E402


def _session() -> GameSession:
    s = GameSession(build_wuliang(7))
    s.intro()
    return s


def _hidden_names(s: GameSession) -> set[str]:
    me = s.beliefs(s.player)
    known = {sk.name for sk in me.entities.values()}
    names = {e.name for e in s.authority.head().entities.values() if e.name and e.id not in me.entities}
    return {n for n in names if n not in known and not any(n in k for k in known)}


def test_opening_suggestions_are_few_distinct_deterministic_and_understood():
    s = _session()
    me = s.beliefs(s.player)
    sug = suggestions(me)
    assert 1 <= len(sug) <= 3 and len(set(sug)) == len(sug)
    assert suggestions(me) == sug
    for text in sug:
        p = s.interpreter.interpret(text, me, ())
        assert p.kind != MoveKind.UNCLEAR, text
        if p.kind == MoveKind.ACT:
            assert p.source == "rules", text             # 行动类落在规则/快路径的句式上，点了不必等模型


def test_following_suggestions_never_names_what_the_player_does_not_know():
    s = _session()
    seen: list[str] = []
    for step in range(10):
        me = s.beliefs(s.player)
        sug = suggestions(me)
        hidden = _hidden_names(s)
        assert sug and not [t for t in sug for n in hidden if n in t], (sug, hidden)
        for text in sug:
            assert s.interpreter.interpret(text, me, ()).kind != MoveKind.UNCLEAR, text
        pick = sug[step % len(sug)]
        seen.append(pick)
        s.turn(pick)
    assert len(set(seen)) >= 4                             # 建议随局面变化，不是原地打转


def test_threat_puts_apology_and_escape_first_and_studied_book_drops_out():
    s = _session()
    s.turn("研读易经")                                     # 开场龚光杰叫阵，第一回合之后他冲着我来
    me = s.beliefs(s.player)
    sug = suggestions(me)
    assert sug[0].startswith("向") and sug[0].endswith("赔罪")
    assert any(t.startswith("逃去") for t in sug)
    assert "研读易经" not in sug
