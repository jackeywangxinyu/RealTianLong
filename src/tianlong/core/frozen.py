"""
[INPUT]: 依赖标准库 collections.abc
[OUTPUT]: 对外提供 FrozenMap（只读映射：dict 子类，封死全部就地修改，可 pickle / deepcopy / JSON 序列化，与 dict 判等）
[POS]: core 的快照护栏。frozen dataclass 只冻住“字段指向谁”，冻不住字段里那个 dict 的内容——
       WorldState.entities、BeliefStore.beliefs 若是普通 dict，任何拿到快照的调用方都能绕过提交直接改写世界与认知。
       FrozenMap 把这条旁路封死：想改，只能 dict(m) 拷一份自己的，再经 apply()/revise() 形成新版本
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, NoReturn

# ============================================================
#  为什么是 dict 子类，而不是 MappingProxyType 或自写 Mapping
#  - MappingProxyType 不能 pickle（Ray / 多进程数据生成都要传快照），也不是 JSON 可序列化的
#  - dict 子类让 json.dumps、dict(m)、== dict 全部原样可用，拷贝与查询都走 C 实现，热路径不变慢
#  - 构造在 __new__ 里用 dict.update 一次填满，__init__ 为空：事后再调 __init__ 也改不动内容
#  - __reduce__ 以 (FrozenMap, (dict(self),)) 重建，避开 pickle 默认的“逐项 __setitem__”
# ============================================================


def _blocked(self: FrozenMap, *args: Any, **kwargs: Any) -> NoReturn:
    raise TypeError("FrozenMap 是只读快照：请 dict(m) 拷贝后修改，再经提交形成新版本")


class FrozenMap(dict):
    __slots__ = ()

    def __new__(cls, *args: Any, **kwargs: Any) -> FrozenMap:
        self = super().__new__(cls)
        dict.update(self, *args, **kwargs)
        return self

    def __init__(self, *args: Any, **kwargs: Any) -> None:  # 内容已在 __new__ 填好；重复调用不得改写
        pass

    # ---- 全部就地修改一律拒绝 ----
    __setitem__ = _blocked
    __delitem__ = _blocked
    update = _blocked
    pop = _blocked
    popitem = _blocked
    clear = _blocked
    setdefault = _blocked
    __ior__ = _blocked

    # ---- 序列化与复制 ----
    def __reduce__(self) -> tuple[type[FrozenMap], tuple[dict]]:
        return (FrozenMap, (dict(self),))

    def __copy__(self) -> FrozenMap:
        return self          # 不可变：浅拷贝就是自己

    def __hash__(self) -> int:  # type: ignore[override]
        return hash(frozenset(self.items()))

    @classmethod
    def fromkeys(cls, keys: Iterable[Any], value: Any = None) -> FrozenMap:  # type: ignore[override]
        return cls(dict.fromkeys(keys, value))
