"""
[INPUT]: 依赖 tianlong.runtime.web 的 WebGame / make_server / MAX_INPUT，tianlong.runtime.session 的 GameSession，
         tianlong.scenarios 的 build_wuliang，标准库 http.client / threading
[OUTPUT]: 网页前端验收：本地起服务（临时端口），页面可取、开场只讲一次且刷新原样再给、开场与每回合都附行动建议；回合经 SSE 逐句推 text 事件、
          以 done 收尾，逐句拼起来就是整段叙述；元指令不推进时间；空输入 400；重开换一局；推送内容里没有 NPC 理由与真相
[POS]: tests 的网页前端：证伪“网页只能一次性拿到整段文字”“刷新页面开场重讲一遍、世界又从头来”“流里夹带了玩家不该看的东西”
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

from __future__ import annotations

import http.client
import json
import threading

import pytest

pytest.importorskip("langgraph")
pytest.importorskip("qdrant_client")

from tianlong.runtime.session import GameSession  # noqa: E402
from tianlong.runtime.web import MAX_INPUT, WebGame, make_server  # noqa: E402
from tianlong.scenarios import build_wuliang  # noqa: E402


@pytest.fixture()
def served():
    game = WebGame(lambda: GameSession(build_wuliang(7)), "天龙八部 · 无量山")
    server = make_server(game, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield game, server.server_address[1]
    finally:
        server.shutdown()
        server.server_close()


def _get(port: int, path: str) -> tuple[int, str, str]:
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=60)
    try:
        conn.request("GET", path)
        r = conn.getresponse()
        return r.status, r.getheader("Content-Type") or "", r.read().decode("utf-8")
    finally:
        conn.close()


def _post(port: int, path: str, body: dict | None = None) -> tuple[int, str, str]:
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=120)
    try:
        raw = json.dumps(body or {}, ensure_ascii=False).encode("utf-8")
        conn.request("POST", path, body=raw, headers={"Content-Type": "application/json"})
        r = conn.getresponse()
        return r.status, r.getheader("Content-Type") or "", r.read().decode("utf-8")
    finally:
        conn.close()


def _events(stream: str) -> list[tuple[str, dict]]:
    out = []
    for chunk in stream.split("\n\n"):
        lines = dict(line.split(": ", 1) for line in chunk.splitlines() if ": " in line)
        if "event" in lines:
            out.append((lines["event"], json.loads(lines["data"])))
    return out


def test_page_and_opening_told_once(served):
    game, port = served
    code, ctype, page = _get(port, "/")
    assert code == 200 and ctype.startswith("text/html") and "/api/turn" in page
    code, _, raw = _get(port, "/api/state")
    first = json.loads(raw)
    assert code == 200 and first["opening"] and first["title"] == "天龙八部 · 无量山"
    assert first["clock"] and first["place"] and first["voice"] is False and first["ended"] is False
    assert 1 <= len(first["suggest"]) <= 3                  # 开场就有可点的行动建议
    again = json.loads(_get(port, "/api/state")[2])
    assert again["opening"] == first["opening"]            # 刷新页面：同一段开场，不是新的一局
    assert _get(port, "/nope")[0] == 404


def test_turn_streams_sentences_then_done(served):
    game, port = served
    _get(port, "/api/state")
    before = game.session.authority.head().clock
    code, ctype, stream = _post(port, "/api/turn", {"text": "环顾四周"})
    assert code == 200 and ctype.startswith("text/event-stream")
    events = _events(stream)
    texts = [d["t"] for e, d in events if e == "text"]
    done = [d for e, d in events if e == "done"]
    assert texts and len(done) == 1 and events[-1][0] == "done"
    assert "".join(texts).strip() == done[0]["narration"].strip()   # 逐句拼起来就是整段叙述
    assert done[0]["advanced"] is True and game.session.authority.head().clock > before
    # 推给浏览器的只有玩家该看的：没有 NPC 理由、真相，也没有叙述上下文
    assert set(done[0]) == {"clock", "place", "kind", "advanced", "narration", "first_text_ms", "ended", "ending",
                            "epilogue", "suggest"}
    assert done[0]["suggest"] and all(isinstance(t, str) and t for t in done[0]["suggest"])


def test_meta_does_not_advance_and_empty_is_rejected(served):
    game, port = served
    _get(port, "/api/state")
    before = game.session.authority.head().clock
    done = [d for e, d in _events(_post(port, "/api/turn", {"text": "/hint"})[2]) if e == "done"][0]
    assert done["kind"] == "meta" and done["advanced"] is False and done["narration"]
    assert game.session.authority.head().clock == before
    assert _post(port, "/api/turn", {"text": "   "})[0] == 400
    assert _post(port, "/api/turn", {"nope": 1})[0] == 400
    assert _post(port, "/api/other")[0] == 404


def test_overlong_input_is_cut(served):
    game, port = served
    seen: list[str] = []
    real = game.session.turn

    def spy(text, **kw):
        seen.append(text)
        return real(text, **kw)

    game.session.turn = spy
    _post(port, "/api/turn", {"text": "等" * (MAX_INPUT + 50)})
    assert len(seen) == 1 and len(seen[0]) == MAX_INPUT


def test_restart_starts_a_fresh_world(served):
    game, port = served
    _get(port, "/api/state")
    _post(port, "/api/turn", {"text": "等一会儿"})
    old = game.session
    code, _, raw = _post(port, "/api/restart")
    assert code == 200 and game.session is not old
    fresh = json.loads(raw)
    assert fresh["opening"] and fresh["ended"] is False
    start = build_wuliang(7).state.clock
    assert old.authority.head().clock > start and game.session.authority.head().clock == start
