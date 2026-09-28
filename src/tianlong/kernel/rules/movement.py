"""
[INPUT]: 依赖 kernel/rules/base 的 ActionRule，kernel/space 的门与位置查询，kernel/resolution 的 succeed / fail
[OUTPUT]: 对外提供 MoveRule、WaitRule
[POS]: kernel/rules 的空间移动；移动的目击者是出发地与目的地两边的人
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

from __future__ import annotations

from collections.abc import Iterator

from tianlong.core import Fact, Intent, Op, Percept, Proposition, WorldState, relocate
from tianlong.kernel import space
from tianlong.kernel.perception import Witnessing
from tianlong.kernel.resolution import Resolution, fail, succeed
from tianlong.kernel.rules.base import ActionRule


class MoveRule(ActionRule):
    op = Op.MOVE
    loudness_base = 0.2
    initiative = 0.4

    def resolve(self, s: WorldState, it: Intent) -> Resolution:
        here = space.place_of(s, it.actor)
        dest = it.target
        assert here is not None and dest is not None
        if dest == here:
            return fail("already_there")
        door = space.door_between(s, here, dest)
        if door is None:
            return fail("not_adjacent")
        if s.attr(door, "locked", False):
            # 推不开门本身就是获知：这扇门锁着
            return fail("door_locked", (Fact(Proposition.attr(door, "locked", True)),))
        passed = (Fact(Proposition.attr(door, "locked", True), False),)
        return succeed(relocate(it.actor, here, dest), learned=passed)

    def witness_places(self, w: Witnessing) -> tuple[str, ...]:
        dest = w.event.intent.target
        return tuple(p for p in (w.event.place, dest) if p)


class WaitRule(ActionRule):
    op = Op.WAIT
    loudness_base = 0.0
    initiative = 0.0

    def resolve(self, s: WorldState, it: Intent) -> Resolution:
        return succeed()

    def perceive(self, w: Witnessing) -> Iterator[tuple[str, Percept]]:
        # 等待不产生任何可感知的东西；在场与否由每 tick 的环顾负责
        return iter(())
