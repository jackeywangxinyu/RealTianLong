"""
[INPUT]: 依赖 core 的 WorldState / Intent / Manner / Op / Percept，kernel/perception 的 Witnessing / Fragment，kernel/resolution 的 Resolution
[OUTPUT]: 对外提供 ActionRule 抽象基类、MANNER_LOUDNESS / MANNER_INITIATIVE 方式系数
[POS]: kernel/rules 的契约；每种行动一个子类，kernel 只认识这个接口——新增行动 = 新增子类 + 注册，不改 kernel。
       resolve() 裁定结果，fragments() 说明结果在物理世界里留下了哪些可感知的片段，perceive() 默认按片段投影给在场者与隔壁
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from typing import ClassVar

from tianlong.core import Intent, Manner, Op, Percept, WorldState
from tianlong.kernel.perception import Fragment, Witnessing
from tianlong.kernel.resolution import Resolution

MANNER_LOUDNESS: dict[Manner, float] = {Manner.NORMAL: 1.0, Manner.CAREFUL: 0.4, Manner.ROUGH: 1.6}
MANNER_INITIATIVE: dict[Manner, float] = {Manner.NORMAL: 0.0, Manner.CAREFUL: -0.2, Manner.ROUGH: 0.2}


class ActionRule(ABC):
    """一种行动的物理：前置条件、效果、响度、先手度、感知方式。

    resolve() 面对的是真实世界状态——这里是整个系统唯一允许“知道真相”的地方。
    """

    op: ClassVar[Op]
    loudness_base: ClassVar[float] = 0.3
    initiative: ClassVar[float] = 0.5   # 同一 tick 内的先手基线：越快的动作越先生效
    usable_when_subdued: ClassVar[bool] = False   # 被点了穴仍能做的事（说话、等待）
    public_reasons: ClassVar[frozenset[str]] = frozenset()   # 旁观者看得出来的失败原因；其余原因只有行动者自知

    @abstractmethod
    def resolve(self, s: WorldState, it: Intent) -> Resolution: ...

    def loudness(self, it: Intent) -> float:
        return self.loudness_base * MANNER_LOUDNESS[it.manner]

    def fragments(self, w: Witnessing) -> tuple[Fragment, ...]:
        return w.default_fragments(self.public_reasons)

    def perceive(self, w: Witnessing) -> Iterator[tuple[str, Percept]]:
        yield w.actor_percept()
        yield from w.observe(self.fragments(w))
