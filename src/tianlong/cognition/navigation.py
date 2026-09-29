"""
[INPUT]: 依赖 cognition/beliefs 的 BeliefStore，core 的 Kind / Rel
[OUTPUT]: 对外提供 believed_place()（沿认为的 AT 链找地点）、routes_between()（认为连通两地的门）、route_to()（下一站及所走的门）、
          next_hop()（下一站地点）
[POS]: cognition 的主观导航；路线来自角色以为的地图——地图错了就会走错，这正是认知差异的一部分。
       行动必须绑定一条自己知道的路（门）：不知道的暗门不在地图上，记得的门后来锁了也照样去推
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

from __future__ import annotations

from collections import deque

from tianlong.cognition.beliefs import BeliefStore
from tianlong.core import Kind, Proposition, Rel


def believed_place(store: BeliefStore, eid: str) -> str | None:
    """物品 → 台面/人 → 地点：沿最可信的 AT 信念上溯。"""
    cur: str | None = eid
    for _ in range(4):
        if cur is None:
            return None
        sk = store.sketch(cur)
        if sk is not None and sk.kind == Kind.PLACE:
            return cur
        cur = store.location_of(cur)
    return None


def routes_between(store: BeliefStore, a: str, b: str, avoid: frozenset[str] = frozenset()) -> tuple[str, ...]:
    """认为连通 a 与 b 的门：认为没锁的在前（不知道锁没锁的次之，认为锁着的最后），同等按 ID。
    确知单向且方向不对的门不算路；avoid 里的门（例如自知过不去的锁门）不算路。"""
    out = []
    for door, sk in store.entities.items():
        if sk.kind != Kind.DOOR or door in avoid:
            continue
        ends = {bl.prop.value for bl in store.positives(door, Rel.CONNECTS.value)}
        if not {a, b} <= ends:
            continue
        oneway = next((bl.prop.value for bl in store.positives(door, "attr.oneway")), None)
        if oneway is not None and oneway != b:
            continue
        locked = store.believed(Proposition.attr(door, "locked", True))
        rank = 1 if locked is None else (2 if locked.holds else 0)
        out.append((rank, door))
    return tuple(d for _, d in sorted(out))


def route_to(store: BeliefStore, dest: str, avoid: frozenset[str] = frozenset()) -> tuple[str, str] | None:
    """沿认为存在的门 BFS：(下一站, 所走的门)；avoid 里的门不走（绕开自知过不去的锁门）。"""
    here = store.location_of(store.owner)
    if here is None or here == dest:
        return None
    adjacency: dict[str, set[str]] = {}
    for door, sk in store.entities.items():
        if sk.kind != Kind.DOOR or door in avoid:
            continue
        ends = [b.prop.value for b in store.positives(door, Rel.CONNECTS.value)]
        for a in ends:
            for b in ends:
                if a != b:
                    adjacency.setdefault(a, set()).add(b)  # type: ignore[arg-type]
    parent: dict[str, str] = {here: here}
    queue = deque([here])
    while queue:
        cur = queue.popleft()
        if cur == dest:
            break
        for nxt in sorted(adjacency.get(cur, ())):
            if nxt not in parent and routes_between(store, cur, nxt, avoid):
                parent[nxt] = cur
                queue.append(nxt)
    if dest not in parent:
        return None
    step = dest
    while parent[step] != here:
        step = parent[step]
    return step, routes_between(store, here, step, avoid)[0]


def believed_distance(store: BeliefStore, a: str, b: str, avoid: frozenset[str] = frozenset()) -> int | None:
    """沿认为存在、方向可走的门的最短跳数（avoid 里的门不算）；地图上走不到返回 None。"""
    dist = {a: 0}
    queue = deque([a])
    places = sorted(e for e, sk in store.entities.items() if sk.kind == Kind.PLACE)
    while queue:
        cur = queue.popleft()
        if cur == b:
            return dist[cur]
        for nxt in places:
            if nxt not in dist and routes_between(store, cur, nxt, avoid):
                dist[nxt] = dist[cur] + 1
                queue.append(nxt)
    return None


def next_hop(store: BeliefStore, dest: str) -> str | None:
    hop = route_to(store, dest)
    return hop[0] if hop else None
