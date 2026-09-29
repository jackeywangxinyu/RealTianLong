"""
[INPUT]: 依赖 core/schema 的 Kind，core/ids 的 digest
[OUTPUT]: 对外提供 AttrType / Access / AttrSpec / ATTRIBUTES / ATTR_SPECS / SKILLS / DEFAULT_EDGE / 按获知途径派生的集合
          （OBSERVABLE_ATTRS / STATUS_ATTRS / PRIVATE_ATTRS / TACTILE_ATTRS / INTROSPECTIVE_ATTRS）、DYNAMIC_ATTRS、ATTRS_VERSION、
          is_private_attr()、applies()、true_value()（从实体存储值 + 时钟派生规则真正使用的值）
[POS]: core 的类型化属性规格——整个系统关于“实体有哪些属性、什么类型、谁能以什么途径知道、会不会被行动改变”的唯一真相源。
       kernel 据此决定感知给谁什么，cognition 据此把认知投影成“值 + 是否已知 + 是否适用”，learning 据此构造特征列与预测目标。
       数值不再一律压成布尔：内力、锋利、难度、进度、点穴剩余时限都以数值进入环境模型
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

from tianlong.core.ids import digest
from tianlong.core.schema import Kind

if TYPE_CHECKING:
    from tianlong.core.entities import Entity, Scalar


class AttrType(StrEnum):
    BOOL = "bool"
    NUM = "num"
    CAT = "cat"


class Access(StrEnum):
    """一个属性能以什么途径被角色知道。"""

    APPEARANCE = "appearance"        # 肉眼可见的静态外观：亲眼见过即知（瓷瓶上写着“解药”）
    STATUS = "status"                # 身体状态：在场者一目了然
    INTERACTIVE = "interactive"      # 只能靠交互获知：推门才知锁没锁、查看才知藏没藏
    TACTILE = "tactile"              # 拿在手里才知道：兵刃锋利与否、是否淬毒、帛书所授何功
    INTROSPECTIVE = "introspective"  # 只有本人知道：内力、修习进度、所学技能
    HIDDEN = "hidden"                # 只有世界知道：秘籍难度、点穴剩余时限、暗门是否只在夜里显形


SKILLS = ("evasion", "absorb")       # 规则里有效果的技能：闪避、吸功


@dataclass(frozen=True, slots=True)
class AttrSpec:
    key: str
    type: AttrType
    kinds: frozenset[Kind]
    access: Access
    scale: float = 1.0                # NUM 的归一化尺度
    categories: tuple[str, ...] = ()  # CAT 的取值（不在其中 = 无）
    dynamic: bool = False             # 行动会改变它：动态模型的预测目标


def _a(key: str, t: AttrType, kinds: tuple[Kind, ...], access: Access, **kw) -> AttrSpec:
    return AttrSpec(key, t, frozenset(kinds), access, **kw)


_B, _N, _C = AttrType.BOOL, AttrType.NUM, AttrType.CAT
_I, _D, _P = Kind.ITEM, Kind.DOOR, Kind.PERSON

ATTRIBUTES: tuple[AttrSpec, ...] = (
    # ---- 物件外观 ----
    _a("small", _B, (_I,), Access.APPEARANCE),
    _a("weapon", _B, (_I,), Access.APPEARANCE),
    _a("cures", _C, (_I,), Access.APPEARANCE, categories=("poisoned", "wounded")),
    # ---- 物件机制 ----
    _a("edge", _N, (_I,), Access.TACTILE, scale=0.5),
    _a("venom", _B, (_I,), Access.TACTILE),
    _a("teaches", _C, (_I,), Access.TACTILE, categories=SKILLS),
    _a("difficulty", _N, (_I,), Access.HIDDEN, scale=5.0),
    # ---- 可交互才知 ----
    _a("hidden", _B, (_I, _D), Access.INTERACTIVE, dynamic=True),
    _a("locked", _B, (_D,), Access.INTERACTIVE, dynamic=True),
    _a("oneway", _B, (_D,), Access.INTERACTIVE),          # 世界里存的是去向地点；特征里是“是否单向” + ONEWAY_TO 边
    _a("night_only", _B, (_D,), Access.HIDDEN),
    # ---- 身体状态 ----
    _a("wounded", _B, (_P,), Access.STATUS, dynamic=True),
    _a("poisoned", _B, (_P,), Access.STATUS, dynamic=True),
    _a("subdued", _B, (_P,), Access.STATUS, dynamic=True),  # 由 subdued_until 与时钟派生
    _a("subdued_left", _N, (_P,), Access.HIDDEN, scale=20.0, dynamic=True),
    # ---- 内在 ----
    _a("martial", _N, (_P,), Access.INTROSPECTIVE, scale=1.5, dynamic=True),
    *(_a(s, _B, (_P,), Access.INTROSPECTIVE, dynamic=True) for s in SKILLS),
    *(_a(f"progress_{s}", _N, (_P,), Access.INTROSPECTIVE, scale=5.0, dynamic=True) for s in SKILLS),
)
ATTR_SPECS: dict[str, AttrSpec] = {a.key: a for a in ATTRIBUTES}
# 行动会改变的属性：动态模型的属性预测目标（布尔三态 + 数值回归），声明即覆盖范围
DYNAMIC_ATTRS: tuple[str, ...] = tuple(a.key for a in ATTRIBUTES if a.dynamic)
# 规格指纹：属性的增删、类型、获知途径或尺度一变，依赖它的模型与存档都必须知道
ATTRS_VERSION = digest(tuple((a.key, a.type.value, tuple(sorted(k.value for k in a.kinds)), a.access.value, a.scale,
                              a.categories, a.dynamic) for a in ATTRIBUTES))


def applies(kind: Kind | str, key: str) -> bool:
    """属性对这种实体是否有意义（“门受没受伤”不是未知，而是不适用）。"""
    spec = ATTR_SPECS.get(key)
    return spec is not None and kind in spec.kinds


def _keys(*access: Access) -> frozenset[str]:
    return frozenset(a.key for a in ATTRIBUTES if a.access in access)


# 肉眼可见的静态外观；locked / hidden 等必须通过交互或推理获知
OBSERVABLE_ATTRS = _keys(Access.APPEARANCE)
# 动态身体状态：环顾时对在场者如实可见，以带极性的属性事实进入信念（“他受伤了/没受伤”）
STATUS_ATTRS = ("wounded", "poisoned", "subdued")
# 拿在手里才知道的机制属性：只对携带者以事实给出
TACTILE_ATTRS = _keys(Access.TACTILE)
# 只有本人知道的内在数值与技能：只以“自我感知”给本人
INTROSPECTIVE_ATTRS = _keys(Access.INTROSPECTIVE)
# 永不以事实形式外泄给旁人的内部数值（内力、修习进度、点穴时限）：旁人只能从后果推断
PRIVATE_ATTRS = frozenset({"martial", "subdued_until"})
PRIVATE_PREFIXES = ("progress_",)


def is_private_attr(key: str) -> bool:
    return key in PRIVATE_ATTRS or key.startswith(PRIVATE_PREFIXES)


# ============================================================
#  规则真正使用的值：存储值 + 派生（点穴随时钟自解、兵刃缺省锋利、单向通道是否存在）
#  kernel 的物理与环境视图都经由这里读值，保证“模型看到的”就是“规则用到的”
# ============================================================

DEFAULT_EDGE = 0.2    # 兵刃未注明锋利时的加成


def true_value(e: Entity, key: str, clock: int) -> Scalar:
    if key == "subdued":
        return int(e.get("subdued_until", 0) or 0) > clock
    if key == "subdued_left":
        return max(0, int(e.get("subdued_until", 0) or 0) - clock)
    if key == "edge":
        return float(e.get("edge", DEFAULT_EDGE) or DEFAULT_EDGE) if e.get("weapon", False) else 0.0
    if key == "oneway":
        return e.get("oneway") is not None
    spec = ATTR_SPECS[key]
    raw = e.get(key)
    if spec.type == AttrType.BOOL:
        return bool(raw)
    if spec.type == AttrType.NUM:
        return float(raw or 0.0)
    return raw if raw in spec.categories else "none"
