"""
[INPUT]: 依赖 core/entities 的 Entity / Relation，core/changes 的 Change 族，core/schema 的 Kind / Rel，core/ids 的 digest，
         core/frozen 的 FrozenMap
[OUTPUT]: 对外提供 WorldState（不可变实际世界图，实体表与内部邻接索引都是 FrozenMap）、ChangeConflict 异常
[POS]: core 的“唯一真相”数据结构；只有 kernel 产生的变化能经由 apply() 形成新版本，persistence 负责持久化它。
       快照连内容也不可变：拿到 state.entities 的调用方改不动它，想改只能 dict() 拷一份再走 apply()
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, replace
from typing import Any

from tianlong.core.changes import AddRelation, Change, RemoveRelation, SetAttr
from tianlong.core.entities import Entity, Relation
from tianlong.core.frozen import FrozenMap
from tianlong.core.ids import digest
from tianlong.core.schema import Kind, Rel


class ChangeConflict(Exception):
    """变化的前置条件与当前状态不符：关系不存在、旧值不匹配、引用了不存在的实体。"""


# ============================================================
#  WorldState
#  - 不可变：所有角色基于同一版本观察，结算基于同一版本裁定
#  - 连内容也不可变：entities 与内部邻接索引包成 FrozenMap，frozen dataclass 冻不住的“字段里那个 dict”也封死
#  - 查询一律返回排序后的元组：frozenset 的迭代顺序随进程哈希种子变化，
#    任何“遍历集合”的地方都可能破坏回放确定性，所以在 API 层根除
# ============================================================


@dataclass(frozen=True, slots=True)
class WorldState:
    seed: int
    version: int
    clock: int
    entities: Mapping[str, Entity]
    relations: frozenset[Relation]
    _out: Mapping[tuple[str, Rel], tuple[str, ...]] = field(init=False, repr=False, compare=False)
    _in: Mapping[tuple[str, Rel], tuple[str, ...]] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.entities, FrozenMap):
            object.__setattr__(self, "entities", FrozenMap(self.entities))
        out: dict[tuple[str, Rel], list[str]] = defaultdict(list)
        inn: dict[tuple[str, Rel], list[str]] = defaultdict(list)
        for r in self.relations:
            out[(r.src, r.type)].append(r.dst)
            inn[(r.dst, r.type)].append(r.src)
        object.__setattr__(self, "_out", FrozenMap({k: tuple(sorted(v)) for k, v in out.items()}))
        object.__setattr__(self, "_in", FrozenMap({k: tuple(sorted(v)) for k, v in inn.items()}))

    # ------------------------------------------------------------
    #  构造
    # ------------------------------------------------------------

    @staticmethod
    def build(
        seed: int, clock: int, entities: Iterable[Entity], relations: Iterable[Relation], version: int = 0
    ) -> WorldState:
        ents = {e.id: e for e in sorted(entities, key=lambda e: e.id)}
        return WorldState(seed, version, clock, ents, frozenset(relations))

    # ------------------------------------------------------------
    #  查询
    # ------------------------------------------------------------

    def entity(self, eid: str) -> Entity:
        try:
            return self.entities[eid]
        except KeyError:
            raise KeyError(f"未知实体: {eid}") from None

    def has_entity(self, eid: str | None) -> bool:
        return eid is not None and eid in self.entities

    def kind(self, eid: str) -> Kind:
        return self.entity(eid).kind

    def attr(self, eid: str, key: str, default: Any = None) -> Any:
        return self.entity(eid).get(key, default)

    def targets(self, src: str, rel: Rel) -> tuple[str, ...]:
        return self._out.get((src, rel), ())

    def sources(self, dst: str, rel: Rel) -> tuple[str, ...]:
        return self._in.get((dst, rel), ())

    def target(self, src: str, rel: Rel) -> str | None:
        """函数型关系的唯一目标；不存在返回 None。"""
        ts = self.targets(src, rel)
        return ts[0] if ts else None

    def holds(self, rel: Relation) -> bool:
        return rel in self.relations

    def of_kind(self, kind: Kind) -> tuple[Entity, ...]:
        return tuple(e for e in self.entities.values() if e.kind == kind)

    def sorted_relations(self) -> tuple[Relation, ...]:
        return tuple(sorted(self.relations, key=Relation.sort_key))

    # ------------------------------------------------------------
    #  演化：apply 只改事实，stamp 才推进版本
    #  一次结算内可多次 apply（行动按先后顺序逐个生效），最后 stamp 一次
    # ------------------------------------------------------------

    def apply(self, changes: Iterable[Change]) -> WorldState:
        ents = dict(self.entities)
        rels = set(self.relations)
        for c in changes:
            if isinstance(c, AddRelation):
                for end in (c.rel.src, c.rel.dst):
                    if end not in ents:
                        raise ChangeConflict(f"关系引用了不存在的实体: {c.rel}")
                if c.rel in rels:
                    raise ChangeConflict(f"关系已存在: {c.rel}")
                rels.add(c.rel)
            elif isinstance(c, RemoveRelation):
                if c.rel not in rels:
                    raise ChangeConflict(f"关系不存在: {c.rel}")
                rels.remove(c.rel)
            elif isinstance(c, SetAttr):
                if c.entity not in ents:
                    raise ChangeConflict(f"属性引用了不存在的实体: {c.entity}")
                current = ents[c.entity].get(c.key)
                if current != c.old:
                    raise ChangeConflict(f"{c.entity}.{c.key} 旧值不符: 期望 {c.old!r} 实际 {current!r}")
                ents[c.entity] = ents[c.entity].with_attr(c.key, c.new)
            else:  # pragma: no cover - 类型系统已穷举
                raise TypeError(f"未知变化类型: {c!r}")
        return WorldState(self.seed, self.version, self.clock, ents, frozenset(rels))

    def stamp(self, version: int, clock: int) -> WorldState:
        return replace(self, version=version, clock=clock)

    # ------------------------------------------------------------
    #  指纹：回放验收的判据——相同存档 + 相同行动 → 相同指纹
    # ------------------------------------------------------------

    def fingerprint(self) -> str:
        ents = tuple((e.id, e.kind.value, e.name, e.attrs) for e in sorted(self.entities.values(), key=lambda e: e.id))
        rels = tuple(r.sort_key() for r in self.sorted_relations())
        return digest(self.seed, self.version, self.clock, ents, rels)
