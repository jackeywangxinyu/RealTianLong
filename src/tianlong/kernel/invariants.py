"""
[INPUT]: 依赖 core 的 WorldState / RELATIONS / Kind / Rel
[OUTPUT]: 对外提供 violations()（列出全部不变量违规）、assert_invariants()、InvariantViolation
[POS]: kernel 的最后一道闸门；每次结算后校验，任何违规都是规则实现的 bug，必须在提交前爆炸而非写进数据库
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

from __future__ import annotations

from tianlong.core import RELATIONS, Kind, Rel, WorldState


class InvariantViolation(Exception):
    pass


_LOCATED = (Kind.PERSON, Kind.ITEM, Kind.SURFACE)
_PERSON_HOLDERS = (Kind.PLACE,)
_SURFACE_HOLDERS = (Kind.PLACE,)


def violations(s: WorldState) -> list[str]:
    out: list[str] = []

    # ---- 关系两端的种类必须符合模式 ----
    for r in s.sorted_relations():
        spec = RELATIONS[r.type]
        if not (s.has_entity(r.src) and s.has_entity(r.dst)):
            out.append(f"悬空关系 {r}")
            continue
        if s.kind(r.src) not in spec.src_kinds or s.kind(r.dst) not in spec.dst_kinds:
            out.append(f"种类不符 {r}")

    for e in s.entities.values():
        at = s.targets(e.id, Rel.AT)
        # ---- 唯一位置：唯一物品不会被复制，也不会凭空消失 ----
        if e.kind in _LOCATED and len(at) != 1:
            out.append(f"{e.id} 的位置数为 {len(at)}")
        if e.kind not in _LOCATED and at:
            out.append(f"{e.kind} {e.id} 不应有位置")
        if e.kind == Kind.PERSON and at and s.kind(at[0]) not in _PERSON_HOLDERS:
            out.append(f"人 {e.id} 必须身处地点")
        if e.kind == Kind.SURFACE and at and s.kind(at[0]) not in _SURFACE_HOLDERS:
            out.append(f"台面 {e.id} 必须位于地点")
        # ---- 门恰好连接两个不同地点 ----
        if e.kind == Kind.DOOR and len(s.targets(e.id, Rel.CONNECTS)) != 2:
            out.append(f"门 {e.id} 连接数不为 2")
        # ---- 物品至多一个所有者 ----
        if e.kind == Kind.ITEM and len(s.sources(e.id, Rel.OWNS)) > 1:
            out.append(f"物品 {e.id} 有多个所有者")
    return out


def assert_invariants(s: WorldState) -> None:
    found = violations(s)
    if found:
        raise InvariantViolation("; ".join(found))
