"""
[INPUT]: 依赖 core/schema 的 Rel / ATTR_PREFIX / is_functional，core/entities 的 Scalar / Relation
[OUTPUT]: 对外提供 Proposition（命题内容）、Fact（带真假极性的命题）
[POS]: core 的“可被相信的内容”；感知产出 Fact，信念存储 Proposition，言语传递 Fact——但命题本身从不声明谁相信它
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

from __future__ import annotations

from dataclasses import dataclass

from tianlong.core.entities import Relation, Scalar
from tianlong.core.schema import ATTR_PREFIX, Rel, is_functional

# ============================================================
#  命题 ≠ 信念
#  “钥匙位于桌面”是命题；“船长相信钥匙位于桌面”是信念。
#  把两者拆开存储，才不会把“某人相信 X”错写成“X 为真”。
# ============================================================


@dataclass(frozen=True, slots=True)
class Proposition:
    subject: str
    predicate: str   # 关系类型（如 "AT"）或属性谓词（如 "attr.locked"）
    value: Scalar    # 关系命题的宾语实体 ID，或属性值；询问时可为 None

    @staticmethod
    def rel(src: str, rel: Rel, dst: str | None) -> Proposition:
        return Proposition(src, rel.value, dst)

    @staticmethod
    def attr(entity: str, key: str, value: Scalar) -> Proposition:
        return Proposition(entity, ATTR_PREFIX + key, value)

    @staticmethod
    def of(relation: Relation) -> Proposition:
        return Proposition(relation.src, relation.type.value, relation.dst)

    @property
    def is_attr(self) -> bool:
        return self.predicate.startswith(ATTR_PREFIX)

    @property
    def attr_key(self) -> str:
        return self.predicate[len(ATTR_PREFIX):]

    @property
    def rel_type(self) -> Rel:
        return Rel(self.predicate)

    @property
    def slot(self) -> tuple[str, str]:
        """互斥槽位：函数型谓词下，同一 (subject, predicate) 至多一个值为真。"""
        return (self.subject, self.predicate)

    @property
    def functional(self) -> bool:
        return is_functional(self.predicate)

    def sort_key(self) -> tuple[str, str, str]:
        return (self.subject, self.predicate, repr(self.value))


@dataclass(frozen=True, slots=True)
class Fact:
    prop: Proposition
    holds: bool = True

    def sort_key(self) -> tuple[str, str, str, bool]:
        return (*self.prop.sort_key(), self.holds)
