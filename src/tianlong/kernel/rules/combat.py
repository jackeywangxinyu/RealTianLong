"""
[INPUT]: 依赖 kernel/rules/base 的 ActionRule，kernel/space 的身手与状态查询，core 的 SetAttr / derive_seed
[OUTPUT]: 对外提供 AttackRule、SUBDUE_TICKS
[POS]: kernel/rules 的武斗物理。一次出手的结局只由身手、兵刃、身法与带种子的随机数裁定：
       首次得手 → 受伤；对已受伤者得手 → 点穴制住（限时自解）；带毒兵刃得手 → 中毒而非受伤。
       技能以效果命名：evasion（闪避，如凌波微步）令攻击更易落空；absorb（吸功，如北冥神功）令徒手攻击者内力反被吸走——
       无论这一击是否得手。书名是内容，效果才是物理
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

from __future__ import annotations

import random

from tianlong.core import Intent, Op, Outcome, SetAttr, WorldState, derive_seed
from tianlong.kernel import space
from tianlong.kernel.resolution import Resolution, fail
from tianlong.kernel.rules.base import ActionRule

SUBDUE_TICKS = 20          # 点穴约莫一炷香后自解
EVASION_DODGE = 0.5         # 凌波微步的闪避加成
ABSORB_DRAIN = 0.1        # 北冥神功每次吸走的内力
SWING = 0.15               # 临场发挥的随机幅度


class AttackRule(ActionRule):
    op = Op.ATTACK
    loudness_base = 0.9
    initiative = 0.7
    public_reasons = frozenset({"parried", "evaded"})   # 被挡开、被闪开，旁人看得一清二楚

    def resolve(self, s: WorldState, it: Intent) -> Resolution:
        target = it.target
        assert target is not None
        if target == it.actor:
            return fail("self_target")
        if space.place_of(s, target) != space.place_of(s, it.actor):
            return fail("out_of_reach")

        changes: list = []
        # ---- 吸功：徒手击打者，内力反被吸去 ----
        if s.attr(target, "absorb", False) and not space.weapons_of(s, it.actor):
            mine = float(s.attr(it.actor, "martial", 0.0) or 0.0)
            drain = round(min(ABSORB_DRAIN, mine), 3)
            if drain > 0:
                theirs = float(s.attr(target, "martial", 0.0) or 0.0)
                changes += [SetAttr(it.actor, "martial", s.attr(it.actor, "martial"), round(mine - drain, 3)),
                            SetAttr(target, "martial", s.attr(target, "martial"), round(theirs + drain, 3))]

        rng = random.Random(derive_seed(s.seed, s.clock, "attack", it.actor, target))
        attack = space.martial_power(s, it.actor) + rng.uniform(-SWING, SWING)
        helpless = space.is_subdued(s, target)
        defense = float("-inf") if helpless else space.martial_power(s, target) + rng.uniform(-SWING, SWING)
        if not helpless and s.attr(target, "evasion", False):
            defense += EVASION_DODGE
        if attack <= defense:
            reason = "evaded" if s.attr(target, "evasion", False) else "parried"
            return Resolution(Outcome.FAILURE, reason, tuple(changes))

        # ---- 得手：毒 → 伤 → 制 ----
        if space.venomous(s, it.actor) and not s.attr(target, "poisoned", False):
            changes.append(SetAttr(target, "poisoned", s.attr(target, "poisoned"), True))
        elif not s.attr(target, "wounded", False):
            changes.append(SetAttr(target, "wounded", s.attr(target, "wounded"), True))
        else:
            changes.append(SetAttr(target, "subdued_until", s.attr(target, "subdued_until"), s.clock + SUBDUE_TICKS))
        return Resolution(Outcome.SUCCESS, None, tuple(changes))
