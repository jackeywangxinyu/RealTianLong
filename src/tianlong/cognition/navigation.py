"""
[INPUT]: 依赖 cognition/beliefs 的 BeliefStore，core 的 Kind / Rel
[OUTPUT]: 对外提供 believed_place()（沿认为的 AT 链找地点）、next_hop()（沿认为存在的门走向目的地的下一站）
[POS]: cognition 的主观导航；路线来自角色以为的地图——地图错了就会走错，这正是认知差异的一部分
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

from __future__ import annotations

from collections import deque

from tianlong.cognition.beliefs import BeliefStore
from tianlong.core import Kind, Rel


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


def next_hop(store: BeliefStore, dest: str) -> str | None:
    here = store.location_of(store.owner)
    if here is None or here == dest:
        return None
    adjacency: dict[str, set[str]] = {}
    for door, sk in store.entities.items():
        if sk.kind != Kind.DOOR:
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
            if nxt not in parent:
                parent[nxt] = cur
                queue.append(nxt)
    if dest not in parent:
        return None
    step = dest
    while parent[step] != here:
        step = parent[step]
    return step
