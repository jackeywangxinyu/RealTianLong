"""
[INPUT]: 依赖 kernel/rules/base 的 ActionRule，kernel/space 的门查询，core 的 SetAttr / MATCHES 关系
[OUTPUT]: 对外提供 UnlockRule、LockRule
[POS]: kernel/rules 的门锁机制；锁状态是隐藏属性，只能通过开锁、推门等交互获知
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

from __future__ import annotations

from tianlong.core import Fact, Intent, Op, Proposition, Rel, Relation, SetAttr, WorldState
from tianlong.kernel import space
from tianlong.kernel.resolution import Resolution, fail, succeed
from tianlong.kernel.rules.base import ActionRule


class _KeyRule(ActionRule):
    """开锁与上锁共享的前置条件：手持钥匙、门在身边、钥匙匹配。"""

    loudness_base = 0.4
    initiative = 0.5
    lock_to: bool   # 目标锁状态

    def resolve(self, s: WorldState, it: Intent) -> Resolution:
        door, key = it.target, it.obj
        assert door is not None and key is not None
        if space.holder_of(s, key) != it.actor:
            return fail("not_holding")
        here = space.place_of(s, it.actor)
        if here not in s.targets(door, Rel.CONNECTS):
            return fail("out_of_reach")
        if not s.holds(Relation(key, Rel.MATCHES, door)):
            return fail("wrong_key", (Fact(Proposition.rel(key, Rel.MATCHES, door), False),))
        matches = Fact(Proposition.rel(key, Rel.MATCHES, door))
        current = s.attr(door, "locked")
        if bool(current) == self.lock_to:
            state = Fact(Proposition.attr(door, "locked", True), self.lock_to)
            return fail("already_locked" if self.lock_to else "already_unlocked", (matches, state))
        return succeed((SetAttr(door, "locked", current, self.lock_to),), learned=(matches,))


class UnlockRule(_KeyRule):
    op = Op.UNLOCK
    lock_to = False


class LockRule(_KeyRule):
    op = Op.LOCK
    lock_to = True
