"""
[INPUT]: 依赖标准库 dataclasses / enum
[OUTPUT]: 对外提供 GoalKind / Goal / Profile
[POS]: core 的角色设定卡；agents 的脚本策略据此行动，learning 的奖励据此计算，language 据 persona 渲染对白——目标是角色条件化的，不存在统一的“剧情精彩度”
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class GoalKind(StrEnum):
    PROTECT = "protect"  # 让物品留在 home 或回到所有者手中
    ACQUIRE = "acquire"  # 把物品弄到自己身上
    DELIVER = "deliver"  # 把物品交到 recipient 手中


@dataclass(frozen=True, slots=True)
class Goal:
    kind: GoalKind
    item: str
    home: str | None = None
    recipient: str | None = None
    weight: float = 1.0


@dataclass(frozen=True, slots=True)
class Profile:
    agent: str
    role: str
    persona: str
    goals: tuple[Goal, ...] = ()
    is_player: bool = False
    trust: tuple[tuple[str, float], ...] = ()   # 对他人说法的信任度（未列出者取默认值）
