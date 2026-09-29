"""
[INPUT]: 依赖 cognition 的 BeliefStore，core/profiles 的 Profile，memory/recall 的 RecallResult，memory/view 的 MemoryView
[OUTPUT]: 对外提供 AgentPort（一个角色能触碰的全部外部能力：认知、回忆、长期记忆摘要、决策依据版本与时间）
[POS]: agents 的隔离边界；LangGraph 节点只能经由 port 读取“自己的认知、自己的回忆”——拿不到世界状态，也拿不到别人的心智
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from tianlong.cognition import BeliefStore
from tianlong.core.profiles import Profile
from tianlong.memory.recall import RecallResult
from tianlong.memory.view import MemoryView


@dataclass(frozen=True)
class AgentPort:
    agent: str
    profile: Profile
    world_id: str
    branch_id: str
    version: int                                   # 决策依据的世界版本（写进意图的 based_on）
    now: int
    beliefs: Callable[[], BeliefStore]
    recall: Callable[[str], RecallResult] | None = None
    memory: Callable[[], MemoryView] | None = None   # 长期记忆的结构化摘要（只汇总本人 known_at <= now 的经历）
