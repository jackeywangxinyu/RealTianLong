"""
[INPUT]: 依赖 cognition/beliefs 的 BeliefStore，cognition/navigation 的 believed_place / believed_distance，core 的 Kind / Rel / Proposition
[OUTPUT]: 对外提供 BeliefReader（core/goals.GoalReader 协议的信念实现）
[POS]: cognition 的目标读者：同一套目标语义（core/goals）在角色自己的认知上求值——“我以为钥匙还在原处，所以守护目标还算满足”。
       不知道就返回 None，绝不拿真相填空。角色观测里的目标状态与主观预测的目标进展都由它给出
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

from __future__ import annotations

from tianlong.cognition.beliefs import BeliefStore
from tianlong.cognition.navigation import believed_distance, believed_place
from tianlong.core import Kind, Proposition, Rel


class BeliefReader:
    def __init__(self, store: BeliefStore) -> None:
        self.store = store

    def holder(self, eid: str) -> str | None:
        return self.store.location_of(eid)

    def place(self, eid: str) -> str | None:
        return believed_place(self.store, eid)

    def owners(self, item: str) -> tuple[str, ...]:
        return self.store.subjects(Rel.OWNS.value, item)

    def status(self, person: str, status: str) -> bool | None:
        b = self.store.believed(Proposition.attr(person, status, True))
        return None if b is None else b.holds

    def persons_at(self, place: str) -> tuple[str, ...]:
        return tuple(p for p, sk in sorted(self.store.entities.items())
                     if sk.kind == Kind.PERSON and believed_place(self.store, p) == place)

    def distance(self, a: str, b: str) -> int | None:
        return believed_distance(self.store, a, b)
