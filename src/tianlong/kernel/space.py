"""
[INPUT]: 依赖 core/world 的 WorldState，core/schema 的 Kind / Rel
[OUTPUT]: 对外提供 holder_of / place_of / neighbors / passable / hops / persons_in / surfaces_in / contents /
          visible_in / is_concealed / is_night / is_subdued / status_of / martial_power / venomous / WorldReader（目标语义的真相读者）
[POS]: kernel 的空间与身体物理；行动规则与感知规则共享的“谁在哪、能看到什么、隔几道门、身手如何”查询，全部只读。
       暗门（hidden）不出现在环顾里，只能靠仔细查看发现（night_only 的只在夜里显形）；单向通道（oneway）只能往一头走
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

from __future__ import annotations

from collections import deque

from tianlong.core import clock
from tianlong.core.attributes import DEFAULT_EDGE
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


def neighbors(s: WorldState, place: str, visible_only: bool = False) -> tuple[tuple[str, str], ...]:
    """(门, 对面地点) 列表，按门 ID 排序；visible_only 时略去暗门。"""
    out = []
    for door in s.sources(place, Rel.CONNECTS):
        if visible_only and s.attr(door, "hidden", False):
            continue
        for other in s.targets(door, Rel.CONNECTS):
            if other != place:
                out.append((door, other))
    return tuple(out)


def passable(s: WorldState, door: str, dest: str) -> bool:
    """单向通道（断崖、塌落的隧道）只能通往 oneway 所指的一端。"""
    oneway = s.attr(door, "oneway")
    return oneway is None or oneway == dest


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


# ============================================================
#  昼夜与身体
#  身手 = 内力底子（martial）+ 最锋利的兵刃（edge），受伤减损、中毒减半；
#  被制住（点穴）以 subdued_until 表示，时限一过自行解开——状态随时钟流逝，无需定时器
# ============================================================

WEAPON_BONUS = DEFAULT_EDGE


def is_night(s: WorldState) -> bool:
    return clock.is_night(s.clock)


def is_subdued(s: WorldState, person: str) -> bool:
    return int(s.attr(person, "subdued_until", 0) or 0) > s.clock


def status_of(s: WorldState, person: str, status: str) -> bool:
    if status == "subdued":
        return is_subdued(s, person)
    return bool(s.attr(person, status, False))


def weapons_of(s: WorldState, person: str) -> tuple[str, ...]:
    return tuple(i for i in contents(s, person) if s.attr(i, "weapon", False))


def martial_power(s: WorldState, person: str) -> float:
    power = float(s.attr(person, "martial", 0.0) or 0.0)
    weapons = weapons_of(s, person)
    if weapons:
        power += max(float(s.attr(w, "edge", WEAPON_BONUS) or WEAPON_BONUS) for w in weapons)
    if s.attr(person, "wounded", False):
        power -= 0.2
    if s.attr(person, "poisoned", False):
        power *= 0.5
    return power


def venomous(s: WorldState, person: str) -> bool:
    return any(s.attr(w, "venom", False) for w in weapons_of(s, person))


# ============================================================
#  WorldReader：目标语义（core/goals）的“真相读者”——奖励与评测用，角色永远拿不到它
# ============================================================


class WorldReader:
    def __init__(self, s: WorldState) -> None:
        self.s = s

    def holder(self, eid: str) -> str | None:
        return holder_of(self.s, eid) if self.s.has_entity(eid) else None

    def place(self, eid: str) -> str | None:
        return place_of(self.s, eid) if self.s.has_entity(eid) else None

    def owners(self, item: str) -> tuple[str, ...]:
        return self.s.sources(item, Rel.OWNS)

    def status(self, person: str, status: str) -> bool | None:
        return status_of(self.s, person, status) if self.s.has_entity(person) else None

    def persons_at(self, place: str) -> tuple[str, ...]:
        return persons_in(self.s, place)

    def distance(self, a: str, b: str) -> int | None:
        """沿可通行方向的门（不论锁否）的最短跳数。"""
        dist = {a: 0}
        queue = deque([a])
        while queue:
            cur = queue.popleft()
            if cur == b:
                return dist[cur]
            for door, nxt in neighbors(self.s, cur):
                if nxt not in dist and passable(self.s, door, nxt):
                    dist[nxt] = dist[cur] + 1
                    queue.append(nxt)
        return None
