"""
[INPUT]: 依赖 core 的 WorldState / Percept / Profile
[OUTPUT]: 对外提供 Scenario（初始世界 + 角色设定 + 初始认知）
[POS]: scenarios 的容器类型；初始认知以“过去的感知”给出，于是信念从第一刻起就只有一个来源——感知，没有“直接注入信念”的后门
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from tianlong.core import Percept, WorldState
from tianlong.core.profiles import Profile


@dataclass(frozen=True, slots=True)
class Scenario:
    world_id: str
    state: WorldState
    profiles: Mapping[str, Profile]
    priors: Mapping[str, tuple[Percept, ...]]

    @property
    def player(self) -> str | None:
        return next((p.agent for p in self.profiles.values() if p.is_player), None)

    @property
    def npcs(self) -> tuple[str, ...]:
        return tuple(sorted(a for a, p in self.profiles.items() if not p.is_player))
