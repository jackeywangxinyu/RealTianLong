"""
[INPUT]: 依赖 kernel/rules/base 的 ActionRule，kernel/space 的门与位置查询，kernel/perception 的 Witnessing / Fragment，
         kernel/resolution 的 succeed / fail
[OUTPUT]: 对外提供 MoveRule、WaitRule（带姿态的等待让在场的人看见）
[POS]: kernel/rules 的空间移动。MOVE = 目的地 + 路线（Intent.obj 是门）：内核只检查这条路是否真在身边、真通往那里、
       方向对不对、锁没锁，从不替角色在全知地图上挑一条可走的路——不知道暗门的人就走不了暗门，记错了路就会走不通。
       观察按实际片段：成功才在目的地留下“抵达”；推不开的门只在门这一侧留下动作与响动；没路可走什么也不留下
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import replace

from tianlong.core import (
    Fact,
    Intent,
    Op,
    Outcome,
    PerceivedEvent,
    Percept,
    Proposition,
    Rel,
    WorldState,
    relocate,
)
from tianlong.kernel import space
from tianlong.kernel.perception import Fragment, Witnessing, change_facts
from tianlong.kernel.resolution import Resolution, fail, succeed
from tianlong.kernel.rules.base import ActionRule

# 在门边试过才失败的原因：在场者看得见这次尝试，隔壁听得见响动
_AT_THE_DOOR = frozenset({"door_locked", "one_way"})


class MoveRule(ActionRule):
    op = Op.MOVE
    loudness_base = 0.2
    initiative = 0.4

    def resolve(self, s: WorldState, it: Intent) -> Resolution:
        here = space.place_of(s, it.actor)
        dest, door = it.target, it.obj
        assert here is not None and dest is not None and door is not None
        if dest == here:
            return fail("already_there")
        ends = s.targets(door, Rel.CONNECTS)
        if here not in ends:
            # 以为这里有这条路，其实没有：找过才知道
            return fail("route_not_here", (Fact(Proposition.rel(door, Rel.CONNECTS, here), False),))
        if dest not in ends:
            return fail("route_mismatch", (Fact(Proposition.rel(door, Rel.CONNECTS, dest), False),))
        if not space.passable(s, door, dest):
            # 断崖只能往下：试过才知道爬不回去
            return fail("one_way", (Fact(Proposition.attr(door, "oneway", s.attr(door, "oneway"))),))
        if s.attr(door, "locked", False):
            # 推不开门本身就是获知：这扇门锁着
            return fail("door_locked", (Fact(Proposition.attr(door, "locked", True)),))
        passed = (Fact(Proposition.attr(door, "locked", True), False),)
        return succeed(relocate(it.actor, here, dest), learned=passed)

    def fragments(self, w: Witnessing) -> tuple[Fragment, ...]:
        origin, outcome = w.event.place, w.event.outcome
        if not origin:
            return ()
        view = w.public_view(self.public_reasons)
        loud = w.loudness
        if outcome == Outcome.SUCCESS:
            facts = change_facts(w.resolution.changes)
            return (Fragment(origin, view, facts, loud),
                    Fragment(str(w.event.intent.target), replace(view, place=str(w.event.intent.target)), facts, 0.0))
        if w.event.reason in _AT_THE_DOOR:
            # 旁人只看见他走向那道门、没能过去；他想去哪儿是没说出口的意图
            attempt = PerceivedEvent(Op.MOVE.value, origin, w.actor, None, view.obj, Outcome.FAILURE)
            return (Fragment(origin, attempt, (), loud),)
        return ()


class WaitRule(ActionRule):
    op = Op.WAIT
    loudness_base = 0.0
    initiative = 0.0
    usable_when_subdued = True

    def resolve(self, s: WorldState, it: Intent) -> Resolution:
        return succeed()

    def perceive(self, w: Witnessing) -> Iterator[tuple[str, Percept]]:
        # 不带姿态的等待不产生任何可感知的东西，在场与否由每 tick 的环顾负责；
        # 带姿态（坐下喝茶、拔出长剑、磕头）的等待，在场的人亲眼看见——姿态与言语行为只是修辞，不附带任何事实
        if not w.event.intent.utterance or not w.event.place:
            return
        yield w.actor_percept()
        view = w.public_view()
        for person in w.witnesses((w.event.place,)):
            yield person, w.sight(view, (), w.event.place)
