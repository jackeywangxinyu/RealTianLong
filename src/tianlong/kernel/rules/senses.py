"""
[INPUT]: 依赖 kernel/rules/base 的 ActionRule，kernel/space 的可见性查询，kernel/perception 的 Witnessing
[OUTPUT]: 对外提供 InspectRule
[POS]: kernel/rules 的主动感知；仔细查看能发现藏匿物、搜身能发现藏在身上的小物件，并给出“完整看清”的负证据范围
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

from __future__ import annotations

from collections.abc import Iterator

from tianlong.core import Fact, Intent, Kind, Op, Percept, Proposition, Rel, WorldState
from tianlong.kernel import space
from tianlong.kernel.perception import Witnessing
from tianlong.kernel.resolution import Resolution, fail, succeed
from tianlong.kernel.rules.base import ActionRule


def _found(s: WorldState, holders: tuple[str, ...]) -> tuple[Fact, ...]:
    """这些容纳者的全部直接内容（含藏匿物）；藏匿物额外标注 hidden，免得随后环顾时被误判为“不见了”。"""
    facts: list[Fact] = []
    for h in holders:
        for e in space.contents(s, h):
            if s.kind(e) == Kind.PERSON:
                continue
            facts.append(Fact(Proposition.rel(e, Rel.AT, h)))
            if s.attr(e, "hidden", False):
                facts.append(Fact(Proposition.attr(e, "hidden", True)))
    return tuple(facts)


class InspectRule(ActionRule):
    op = Op.INSPECT
    loudness_base = 0.2
    initiative = 0.3

    def resolve(self, s: WorldState, it: Intent) -> Resolution:
        target = it.target
        assert target is not None
        here = space.place_of(s, it.actor)
        kind = s.kind(target)
        if kind == Kind.PLACE:
            if target != here:
                return fail("out_of_reach")
            holders = (target, *space.surfaces_in(s, target))
        elif kind == Kind.SURFACE:
            if space.holder_of(s, target) != here:
                return fail("out_of_reach")
            holders = (target,)
        else:  # PERSON：搜身
            if target == it.actor:
                return fail("self_target")
            if space.place_of(s, target) != here:
                return fail("out_of_reach")
            holders = (target,)
        return succeed(learned=_found(s, holders), scopes=holders)

    def perceive(self, w: Witnessing) -> Iterator[tuple[str, Percept]]:
        # 在场者看见搜查过程，也看见搜出了什么（但不获得“完整看清”的范围）
        yield w.actor_percept()
        seers = w.witnesses(self.witness_places(w))
        for seer in seers:
            yield seer, w.sight(facts=w.resolution.learned)
        yield from w.sounds(exclude=seers)
