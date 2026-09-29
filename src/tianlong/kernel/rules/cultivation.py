"""
[INPUT]: 依赖 kernel/rules/base 的 ActionRule，kernel/space 的持有查询，kernel/perception 的 Witnessing，core 的 SetAttr
[OUTPUT]: 对外提供 StudyRule（研读秘籍）、UseRule（施用物品）
[POS]: kernel/rules 的修习与施治。研读是逐次累积的：秘籍的 teaches 指明所授技能、difficulty 指明所需次数，
       进度是私密数值；旁观者只看见你在读东西，不知道你读到了哪、学没学成。施用只看物品的 cures 是否对症
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import replace

from tianlong.core import Intent, Op, Outcome, Percept, SetAttr, WorldState
from tianlong.kernel import space
from tianlong.kernel.perception import Witnessing
from tianlong.kernel.resolution import Resolution, fail, succeed
from tianlong.kernel.rules.base import ActionRule


class StudyRule(ActionRule):
    op = Op.STUDY
    loudness_base = 0.05
    initiative = 0.2

    def resolve(self, s: WorldState, it: Intent) -> Resolution:
        scroll = it.target
        assert scroll is not None
        if space.holder_of(s, scroll) != it.actor:
            return fail("not_holding")
        skill = s.attr(scroll, "teaches")
        if not skill:
            return fail("nothing_to_learn")
        if s.attr(it.actor, skill, False):
            return fail("already_learned")
        key = f"progress_{skill}"
        done = int(s.attr(it.actor, key, 0) or 0) + 1
        changes = [SetAttr(it.actor, key, s.attr(it.actor, key), done)]
        if done >= int(s.attr(scroll, "difficulty", 1) or 1):
            changes.append(SetAttr(it.actor, skill, s.attr(it.actor, skill), True))
            return Resolution(Outcome.SUCCESS, "mastered", tuple(changes))
        return Resolution(Outcome.SUCCESS, "progress", tuple(changes))

    def perceive(self, w: Witnessing) -> Iterator[tuple[str, Percept]]:
        # 修习所得只有自己知道；旁人只看见他埋头读着什么
        yield w.actor_percept()
        seers = w.witnesses(self.witness_places(w))
        quiet = replace(w.full_view(), reason=None)     # 不让旁人看出学没学成
        for seer in seers:
            yield seer, w.sight(view=quiet, facts=())
        yield from w.sounds(exclude=seers)


class UseRule(ActionRule):
    op = Op.USE
    loudness_base = 0.2
    initiative = 0.6

    def resolve(self, s: WorldState, it: Intent) -> Resolution:
        patient, item = it.target, it.obj
        assert patient is not None and item is not None
        if space.holder_of(s, item) != it.actor:
            return fail("not_holding")
        if space.place_of(s, patient) != space.place_of(s, it.actor):
            return fail("out_of_reach")
        cure = s.attr(item, "cures")
        if not cure or not s.attr(patient, cure, False):
            return fail("no_effect")
        return succeed((SetAttr(patient, cure, s.attr(patient, cure), None),))
