"""
[INPUT]: 依赖 core/entities 的 Relation / Scalar，core/schema 的 Rel
[OUTPUT]: 对外提供 AddRelation / RemoveRelation / SetAttr / Change 联合类型、relocate() 构造器、change_sort_key()
[POS]: core 的“世界变化”语言；kernel 产出它、world 应用它、事件记录它、GNN 以它为监督标签
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

from __future__ import annotations

from dataclasses import dataclass

from tianlong.core.entities import Relation, Scalar
from tianlong.core.schema import Rel

# ============================================================
#  变化是带前置条件的原子操作：
#  RemoveRelation 要求关系存在，SetAttr 要求旧值匹配。
#  这样“基于过时版本算出的变化”在应用时会被拒绝，而不是悄悄覆盖。
# ============================================================


@dataclass(frozen=True, slots=True)
class AddRelation:
    rel: Relation


@dataclass(frozen=True, slots=True)
class RemoveRelation:
    rel: Relation


@dataclass(frozen=True, slots=True)
class SetAttr:
    entity: str
    key: str
    old: Scalar
    new: Scalar


Change = AddRelation | RemoveRelation | SetAttr


def relocate(entity: str, src_holder: str, dst_holder: str) -> tuple[Change, ...]:
    """实体从一个容纳者移到另一个：AT 是函数型关系，所以是一删一增。"""
    return (
        RemoveRelation(Relation(entity, Rel.AT, src_holder)),
        AddRelation(Relation(entity, Rel.AT, dst_holder)),
    )


def change_sort_key(c: Change) -> tuple:
    if isinstance(c, SetAttr):
        return ("attr", c.entity, c.key, repr(c.new))
    tag = "add" if isinstance(c, AddRelation) else "remove"
    return (tag, *c.rel.sort_key())
