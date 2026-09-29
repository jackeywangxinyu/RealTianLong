"""
[INPUT]: 汇总 core 各模块
[OUTPUT]: 对外再导出 core 的全部公共类型，供上层 `from tianlong.core import ...`
[POS]: core 包入口；core 只依赖标准库，是整座依赖图的最底层
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

from tianlong.core.attributes import (
    ATTR_SPECS,
    ATTRIBUTES,
    ATTRS_VERSION,
    DYNAMIC_ATTRS,
    INTROSPECTIVE_ATTRS,
    OBSERVABLE_ATTRS,
    PRIVATE_ATTRS,
    SKILLS,
    STATUS_ATTRS,
    TACTILE_ATTRS,
    Access,
    AttrSpec,
    AttrType,
    applies,
    is_private_attr,
)
from tianlong.core.changes import AddRelation, Change, RemoveRelation, SetAttr, change_sort_key, relocate
from tianlong.core.clock import TICK_MINUTES, at, clock_label, is_night, minutes_until_night
from tianlong.core.entities import Entity, Relation, Scalar
from tianlong.core.events import (
    ADMISSION_REASONS,
    REASONS,
    RULE_REASONS,
    EntitySketch,
    Event,
    Intent,
    Modality,
    Observation,
    Outcome,
    PerceivedEvent,
    Percept,
    reason_key,
)
from tianlong.core.ids import derive_seed, digest, make_id
from tianlong.core.propositions import Fact, Proposition
from tianlong.core.schema import (
    ATTR_PREFIX,
    OP_SIGNATURES,
    RELATIONS,
    Kind,
    Manner,
    Op,
    OpSignature,
    Rel,
    RelSpec,
    is_functional,
)
from tianlong.core.world import ChangeConflict, WorldState

__all__ = [
    "ATTR_SPECS", "ATTRIBUTES", "ATTRS_VERSION", "DYNAMIC_ATTRS", "INTROSPECTIVE_ATTRS", "OBSERVABLE_ATTRS", "PRIVATE_ATTRS", "SKILLS", "STATUS_ATTRS",
    "TACTILE_ATTRS", "Access", "AttrSpec", "AttrType", "applies", "is_private_attr",
    "AddRelation", "Change", "RemoveRelation", "SetAttr", "change_sort_key", "relocate",
    "TICK_MINUTES", "at", "clock_label", "is_night", "minutes_until_night",
    "Entity", "Relation", "Scalar",
    "EntitySketch", "Event", "Intent", "Modality", "Observation", "Outcome", "Percept", "PerceivedEvent",
    "ADMISSION_REASONS", "REASONS", "RULE_REASONS", "reason_key",
    "derive_seed", "digest", "make_id",
    "Fact", "Proposition",
    "ATTR_PREFIX", "OP_SIGNATURES", "RELATIONS", "Kind", "Manner", "Op", "OpSignature", "Rel", "RelSpec", "is_functional",
    "ChangeConflict", "WorldState",
]
