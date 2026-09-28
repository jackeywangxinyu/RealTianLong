"""
[INPUT]: 依赖 core/world 的 WorldState，core/schema 的 Kind / Rel
[OUTPUT]: 对外提供 holder_of / place_of / neighbors / door_between / hops / persons_in / surfaces_in / contents / visible_in / is_concealed
[POS]: kernel 的空间物理；行动规则与感知规则共享的“谁在哪、能看到什么、隔几道门”查询，全部只读
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

from __future__ import annotations

from collections import deque

from tianlong.core.schema import Kind, Rel
from tianlong.core.world import WorldState

# ============================================================
#  位置链：物品 → 台面/人 → 地点
#  物品跟着携带者走，不需要额外的变化——AT 链天然表达了“揣在身上”
# ============================================================


def holder_of(s: WorldState, eid: str) -> str | None:
    return s.target(eid, Rel.AT)


def place_of(s: WorldState, eid: str) -> str | None:
    """沿 AT 链上溯到地点；地点自身返回自身。"""
    cur: str | None = eid
    for _ in range(8):  # 链长由种类约束保证 ≤ 3，这里只是防御
        if cur is None:
            return None
        if s.kind(cur) == Kind.PLACE:
            return cur
        cur = holder_of(s, cur)
    raise RuntimeError(f"AT 链过长或成环: {eid}")


def contents(s: WorldState, holder: str) -> tuple[str, ...]:
    return s.sources(holder, Rel.AT)


def persons_in(s: WorldState, place: str) -> tuple[str, ...]:
    return tuple(e for e in contents(s, place) if s.kind(e) == Kind.PERSON)


def surfaces_in(s: WorldState, place: str) -> tuple[str, ...]:
    return tuple(e for e in contents(s, place) if s.kind(e) == Kind.SURFACE)


# ============================================================
#  门与连通
# ============================================================


def neighbors(s: WorldState, place: str) -> tuple[tuple[str, str], ...]:
    """(门, 对面地点) 列表，按门 ID 排序。"""
    out = []
    for door in s.sources(place, Rel.CONNECTS):
        for other in s.targets(door, Rel.CONNECTS):
            if other != place:
                out.append((door, other))
    return tuple(out)


def door_between(s: WorldState, a: str, b: str) -> str | None:
    """a、b 之间优先返回未上锁的门，否则返回任意一扇；不相邻返回 None。"""
    doors = [d for d, other in neighbors(s, a) if other == b]
    if not doors:
        return None
    unlocked = [d for d in doors if not s.attr(d, "locked", False)]
    return (unlocked or doors)[0]


def hops(s: WorldState, src: str, max_hops: int) -> dict[str, int]:
    """从 src 出发、经门（不论锁否，声音能穿门）可达地点的跳数。"""
    dist = {src: 0}
    queue = deque([src])
    while queue:
        cur = queue.popleft()
        if dist[cur] >= max_hops:
            continue
        for _, nxt in neighbors(s, cur):
            if nxt not in dist:
                dist[nxt] = dist[cur] + 1
                queue.append(nxt)
    return dist


# ============================================================
#  可见性：身处某地点、随意环顾时能看到的东西
#  - 藏匿物（hidden）只有仔细查看才能发现
#  - 小物件被人携带时藏在身上（concealed），旁人看不见
# ============================================================


def is_concealed(s: WorldState, item: str) -> bool:
    holder = holder_of(s, item)
    return holder is not None and s.kind(holder) == Kind.PERSON and bool(s.attr(item, "small", False))


def visible_in(s: WorldState, place: str, viewer: str, reveal_hidden: bool = False) -> tuple[str, ...]:
    """viewer 在 place 中能看到的实体（不含门、不含自身）。"""
    seen: list[str] = []
    for e in contents(s, place):
        if e == viewer:
            continue
        kind = s.kind(e)
        if kind == Kind.ITEM and s.attr(e, "hidden", False) and not reveal_hidden:
            continue
        seen.append(e)
        if kind == Kind.SURFACE:
            seen.extend(
                i for i in contents(s, e) if reveal_hidden or not s.attr(i, "hidden", False)
            )
        elif kind == Kind.PERSON:
            seen.extend(i for i in contents(s, e) if not is_concealed(s, i))
    return tuple(seen)
