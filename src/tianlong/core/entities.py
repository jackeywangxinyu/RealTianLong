"""
[INPUT]: 依赖 core/schema 的 Kind / Rel
[OUTPUT]: 对外提供 Scalar 类型别名、Entity、Relation 两个不可变值对象
[POS]: core 的图元素；被 world（实际世界图）、changes（变化）、events（观察草图）共同引用
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from tianlong.core.schema import Kind, Rel

Scalar = str | int | float | bool | None


# ============================================================
#  Entity：属性以有序元组保存
#  - 不可变 + 可哈希 + 可 pickle（LangGraph 检查点、RLlib 多进程都需要）
#  - 有序保证 repr 稳定，进而保证 fingerprint 可回放
# ============================================================


@dataclass(frozen=True, slots=True)
class Entity:
    id: str
    kind: Kind
    name: str
    attrs: tuple[tuple[str, Scalar], ...] = ()

    @staticmethod
    def make(id: str, kind: Kind, name: str, **attrs: Scalar) -> Entity:
        return Entity(id, kind, name, tuple(sorted(attrs.items())))

    def get(self, key: str, default: Any = None) -> Any:
        for k, v in self.attrs:
            if k == key:
                return v
        return default

    def with_attr(self, key: str, value: Scalar) -> Entity:
        rest = {k: v for k, v in self.attrs if k != key}
        if value is not None:
            rest[key] = value
        return replace(self, attrs=tuple(sorted(rest.items())))

    def attr_dict(self) -> dict[str, Scalar]:
        return dict(self.attrs)


@dataclass(frozen=True, slots=True)
class Relation:
    src: str
    type: Rel
    dst: str

    def sort_key(self) -> tuple[str, str, str]:
        return (self.src, self.type.value, self.dst)
