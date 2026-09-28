"""
[INPUT]: 依赖 kernel/rules/base 的 ActionRule，kernel/space 的可及性查询，core 的 relocate / SetAttr
[OUTPUT]: 对外提供 TakeRule、PutRule、GiveRule
[POS]: kernel/rules 的物件搬运；只改 AT（位置），从不改 OWNS（所有权）——拿走不等于拥有
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

from __future__ import annotations

from tianlong.core import Fact, Intent, Kind, Manner, Op, Proposition, Rel, SetAttr, WorldState, relocate
from tianlong.kernel import space
from tianlong.kernel.resolution import Resolution, fail, succeed
from tianlong.kernel.rules.base import ActionRule


def _within_reach(s: WorldState, actor: str, holder: str) -> bool:
    """holder 是行动者所在地点，或该地点里的台面。"""
    here = space.place_of(s, actor)
    if holder == here:
        return True
    return s.kind(holder) == Kind.SURFACE and space.holder_of(s, holder) == here


class TakeRule(ActionRule):
    op = Op.TAKE
    loudness_base = 0.8   # 钥匙串会叮当作响
    initiative = 0.6

    def resolve(self, s: WorldState, it: Intent) -> Resolution:
        item = it.target
        assert item is not None
        holder = space.holder_of(s, item)
        if holder == it.actor:
            return fail("already_held")
        if holder is None or space.place_of(s, item) != space.place_of(s, it.actor):
            return fail("not_found")
        if s.kind(holder) == Kind.PERSON:
            # 近在眼前被别人拿着：伸手时自然看清了
            return fail("held_by_other", (Fact(Proposition.rel(item, Rel.AT, holder)),))
        if not _within_reach(s, it.actor, holder):
            return fail("out_of_reach")
        changes = relocate(item, holder, it.actor)
        if s.attr(item, "hidden", False):
            changes += (SetAttr(item, "hidden", True, None),)
        return succeed(changes)


class PutRule(ActionRule):
    op = Op.PUT
    loudness_base = 0.3
    initiative = 0.5

    def resolve(self, s: WorldState, it: Intent) -> Resolution:
        dest, item = it.target, it.obj
        assert dest is not None and item is not None
        if space.holder_of(s, item) != it.actor:
            return fail("not_holding")
        if not _within_reach(s, it.actor, dest):
            return fail("out_of_reach")
        changes = relocate(item, it.actor, dest)
        if it.manner == Manner.CAREFUL:
            # 小心地放 = 藏起来：随意环顾者看不见，只有仔细查看才能发现
            changes += (SetAttr(item, "hidden", None, True),)
        return succeed(changes)


class GiveRule(ActionRule):
    op = Op.GIVE
    loudness_base = 0.2
    initiative = 0.5

    def resolve(self, s: WorldState, it: Intent) -> Resolution:
        recipient, item = it.target, it.obj
        assert recipient is not None and item is not None
        if recipient == it.actor:
            return fail("self_target")
        if space.holder_of(s, item) != it.actor:
            return fail("not_holding")
        if space.place_of(s, recipient) != space.place_of(s, it.actor):
            return fail("out_of_reach")
        return succeed(relocate(item, it.actor, recipient))
