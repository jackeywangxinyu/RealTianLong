"""
[INPUT]: 依赖 tianlong.scenarios 的 build_wuliang，tianlong.runtime 的 GameSession / WorldAuthority，tianlong.agents 的 Orchestrator
[OUTPUT]: 天龙八部·无量山场景验收：世界自洽、开场冲突链自然涌现、入夜私奔按约动身、玩家可循原著路线抵达琅嬛福地并学成凌波微步
[POS]: tests 的内容层；剧情不是写死的脚本，而是由规则与认知涌现——这里断言的是“会发生”，不是“按剧本发生”
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

from __future__ import annotations

import pytest

pytest.importorskip("langgraph")
pytest.importorskip("qdrant_client")

from tianlong.core import Op, Rel, at  # noqa: E402
from tianlong.kernel import violations  # noqa: E402
from tianlong.runtime.session import GameSession  # noqa: E402
from tianlong.scenarios import SCENARIOS, build_wuliang  # noqa: E402


def test_world_is_consistent():
    sc = build_wuliang()
    assert not violations(sc.state)
    assert sc.player == "duanyu" and "wuliang" in SCENARIOS
    assert set(sc.profiles) == {e.id for e in sc.state.of_kind(__import__("tianlong.core", fromlist=["Kind"]).Kind.PERSON)}
    assert all(k.split("@")[0] in sc.state.entities for k in sc.lore), "外观描写只针对存在的实体"


def test_opening_conflict_chain_emerges():
    """龚光杰寻衅 → 钟灵放貂 → 左子穆护短、制住钟灵、搜出解药、救治弟子。"""
    s = GameSession(build_wuliang())
    events = []
    for _ in range(7):
        events += [(e.actor, e.op, e.intent.target, e.outcome.value) for e in s.turn("等待").events]
    assert ("gongguangjie", Op.ATTACK, "duanyu", "success") in events
    assert ("zhongling", Op.ATTACK, "gongguangjie", "success") in events
    assert ("zuozimu", Op.TAKE, "antidote", "success") in events
    assert ("zuozimu", Op.USE, "gongguangjie", "success") in events
    st = s.authority.head()
    assert not st.attr("gongguangjie", "poisoned") and st.target("antidote", Rel.AT) == "zuozimu"


def test_lovers_leave_at_the_appointed_hour():
    s = GameSession(build_wuliang())
    s.turn("去后院")
    moved = None
    for _ in range(200):        # 等待可能被身边的动静打断，逐次推进直到过了约定时辰
        if s.authority.head().clock > at(1, 19, 21):
            break
        r = s.turn("等到天黑") if s.authority.head().clock < at(1, 19, 0) else s.turn("等待")
        hit = [e for e in r.events if e.actor == "ganguanghao" and e.op == Op.MOVE]
        if hit:
            moved = hit[0].tick
            break
    assert moved == at(1, 19, 20), "约定的时辰一到就动身，而不是等到下一次闲置轮询"


def test_player_can_follow_the_canon_route_to_the_scrolls():
    s = GameSession(build_wuliang())
    for cmd in ["去后院", "往后山走", "去崖顶", "跳下断崖"]:
        s.turn(cmd)
    s.turn("查看玉璧")
    assert not s.beliefs("duanyu").knows("d_cave"), "白日里看不出玉璧的秘密"
    r = s.turn("等到天黑")
    if not s.authority.head().clock >= at(1, 19, 0):   # 被人打断则再等
        s.turn("等到天黑")
    r = s.turn("查看玉璧")
    assert "暗道" in r.narration
    for cmd in ["钻进石缝", "进石门", "磕头", "拿凌波微步", "拿北冥神功"]:
        s.turn(cmd)
    for _ in range(3):
        r = s.turn("研读凌波微步")
    assert "豁然贯通" in r.narration and s.authority.head().attr("duanyu", "evasion") is True
    s.turn("钻进隧道")
    st = s.authority.head()
    assert st.target("duanyu", Rel.AT) == "lancang"
    assert {st.target("scroll_bm", Rel.AT), st.target("scroll_lb", Rel.AT)} == {"duanyu"}


def test_first_sight_descriptions_only_once():
    s = GameSession(build_wuliang())
    s.intro()
    r1 = s.turn("去后院")
    r2 = s.turn("去大殿")
    assert "厢房" in r1.narration and "锦幡" not in r2.narration, "大殿开场已描写过，不再重复"
