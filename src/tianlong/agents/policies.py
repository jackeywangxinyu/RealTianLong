"""
[INPUT]: 依赖 agents/policy_kit 的 Situation / Choice / Policy / PolicyKit，agents/tactics 的 MartialTactics，
         cognition/navigation 的 believed_place，cognition/goals 的 BeliefReader，core/goals 的 GoalRegistry，
         core 的 Op / Manner / Rel / Kind / Modality / Fact / Proposition，core/profiles 的 Goal / GoalKind
[OUTPUT]: 对外提供 ScriptedPolicy（角色条件化的规则策略），并再导出 Situation / Choice / Policy
[POS]: agents 的决策作曲者：自救 → 还手 → 救治盟友 → 回应提问 → 按目标（守护/获取/递送/守地/寻仇/灭口/护人，受时间闸门约束）→
       查探响动 → 等待。目标所需的东西或人下落不明时，凭自己的地图与勘察记录去找（而不是原地干等）；
       等待带结构化原因（没事可做/目标未到时辰/自以为已达成/不知道而卡住/没有可行候选/找不到动作）。
       它是阶段 A 的初始策略，也是阶段 C 模仿学习的示范者；RL 策略实现同一协议即可替换
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

from __future__ import annotations

from collections.abc import Callable

from tianlong.agents.policy_kit import Choice, Policy, Situation
from tianlong.agents.tactics import MartialTactics
from tianlong.cognition.goals import BeliefReader
from tianlong.cognition.navigation import believed_place
from tianlong.core import Fact, Kind, Manner, Modality, Op, Proposition, Rel
from tianlong.core.goals import GoalRegistry
from tianlong.core.profiles import Goal, GoalKind

_REGISTRY = GoalRegistry()

__all__ = ["Choice", "Policy", "ScriptedPolicy", "Situation"]


# ============================================================
#  ScriptedPolicy
#  所有判断都来自信念与近期经历；“最近做过”由 SELF 经历判断，避免反复盘问同一个人
# ============================================================


class ScriptedPolicy(MartialTactics):
    def choose(self, sit: Situation) -> Choice:
        steps: list[Callable[[Situation], Choice | None]] = [
            self._cure_self, self._retaliate, self._heal_allies, self._answer_questions,
        ]
        steps += [self._goal_step(g) for g in sit.profile.goals if g.active(sit.now)]
        steps += [self._investigate_noise]
        for step in steps:
            choice = step(sit)
            if choice is not None:
                return choice
        return self._wait(sit)

    def _wait(self, sit: Situation) -> Choice:
        """等待，并说明为什么：模仿学习要区分“合理地等”与“因为不知道而卡住”。"""
        goals = sit.profile.goals
        active = [g for g in goals if g.active(sit.now)]
        if not goals:
            return Choice(0, "没什么要做的，原地等待", "idle")
        if not active:
            return Choice(0, "时辰未到，先等着", "goal_inactive")
        reader = BeliefReader(sit.beliefs)
        status = [_REGISTRY.satisfied(reader, sit.agent, g, sit.profile.allies) for g in active]
        if all(s is True for s in status):
            return Choice(0, "一切如常，原地等待", "goal_done")
        if len(sit.candidates) <= 1:
            return Choice(0, "无事可为，只能等待", "no_candidate")
        if any(s is None for s in status) or any(self._unknown_target(sit, g) for g in active):
            return Choice(0, "不知从何下手，只好等待", "stuck_unknown")
        return Choice(0, "想不出办法，原地等待", "expert_no_action")

    @staticmethod
    def _unknown_target(sit: Situation, g: Goal) -> bool:
        b = sit.beliefs
        return any(x is not None and believed_place(b, x) is None for x in (g.item, g.person, g.recipient))

    # ------------------------------------------------------------
    #  回应提问：如实说出自己最相信的下落
    # ------------------------------------------------------------

    def _answer_questions(self, sit: Situation) -> Choice | None:
        """欠着的问题（持久记录，不随经历缓冲滚掉）：问话的人在眼前、自己又知道答案，就如实相告。"""
        b = sit.beliefs
        here = set(self._persons_here(b))
        for ob in b.obligations:
            if ob.kind != "answer" or ob.counterpart not in here:
                continue
            best = b.best(ob.topic.prop.subject, ob.topic.prop.predicate)
            if best is not None and not self._said(sit, ob.counterpart, Fact(best.prop, True)):
                name = self._name(b, ob.counterpart)
                late = "（前番问过）" if sit.now - ob.since > 2 else ""
                return self._pick(sit, f"{name}问我{late}，如实相告", Op.TELL, ob.counterpart,
                                  topic=Fact(best.prop, True))
        return None

    # ------------------------------------------------------------
    #  目标
    # ------------------------------------------------------------

    def _goal_step(self, g: Goal) -> Callable[[Situation], Choice | None]:
        return {
            GoalKind.PROTECT: lambda s: self._protect(s, g),
            GoalKind.ACQUIRE: lambda s: self._acquire(s, g),
            GoalKind.DELIVER: lambda s: self._deliver(s, g),
            GoalKind.GUARD: lambda s: self._guard(s, g),
            GoalKind.HOSTILE: lambda s: self._hostile(s, g),
            GoalKind.ESCAPE: lambda s: self._escape(s, g),
            GoalKind.DEFEND: lambda s: self._defend(s, g),
        }[g.kind]

    def _protect(self, sit: Situation, g: Goal) -> Choice | None:
        if g.item is None:
            return None
        b, me = sit.beliefs, sit.agent
        here = b.location_of(me)
        owners = b.subjects(Rel.OWNS.value, g.item)
        loc = b.location_of(g.item)
        name = self._name(b, g.item)
        if loc is not None and (loc == g.home or loc in owners):
            return None  # 一切如常
        if loc == me and g.home:
            home_place = believed_place(b, g.home)
            if here == home_place:
                return self._pick(sit, f"把{name}放回原处", Op.PUT, g.home, g.item, Manner.NORMAL)
            return self._go_towards(sit, home_place, f"带着{name}回去")
        if loc is not None:
            loc_kind = b.sketch(loc).kind if b.sketch(loc) else None
            if loc_kind == Kind.PERSON:
                return self._held_by_other(sit, g, loc, owners)
            if believed_place(b, loc) == here:
                return self._pick(sit, f"{name}被挪了位置，先收回来", Op.TAKE, g.item, manner=Manner.NORMAL)
            return self._go_towards(sit, believed_place(b, loc), f"去{self._name(b, loc)}找回{name}")
        # ---- 不知下落 ≠ 丢失：只有确知“不在原处”才开始找（盘问、搜身）；不知道就只去原处看一眼 ----
        home_belief = b.believed(Proposition.rel(g.item, Rel.AT, g.home)) if g.home else None
        if home_belief is None:
            if not self._home_searched(sit, g):
                return self._look_in_on(sit, g)
            # 亲手把原处仔细翻过、仍不见它：当作不在原处，才开始盘问与搜寻
        elif home_belief.holds:
            return None
        # ---- 下落不明：先问、再搜、再查看现场 ----
        suspects = [p for p, sk in sorted(b.entities.items())
                    if sk.kind == Kind.PERSON and p != me and p not in owners and b.location_of(p) == here]
        for s in suspects:
            if not self._did_recently(sit, Op.ASK, s):
                q = Fact(Proposition.rel(g.item, Rel.AT, None), True)
                return self._pick(sit, f"{name}不见了，先问问{self._name(b, s)}", Op.ASK, s, topic=q)
            if not self._did_recently(sit, Op.INSPECT, s):
                return self._pick(sit, f"问不出结果，只好搜{self._name(b, s)}的身", Op.INSPECT, s)
        if here and not self._did_recently(sit, Op.INSPECT, here):
            return self._pick(sit, f"仔细找找{name}", Op.INSPECT, here)
        return self._explore(sit, g.item)

    @staticmethod
    def _home_searched(sit: Situation, g: Goal) -> bool:
        b = sit.beliefs
        return g.home is not None and (g.home in b.searched or believed_place(b, g.home) in b.searched)

    def _look_in_on(self, sit: Situation, g: Goal) -> Choice | None:
        """守护之物从没亲眼见过：走到它该在的地方，仔细看一眼——核实，而不是怀疑任何人。"""
        if g.home is None:
            return None
        b = sit.beliefs
        where = believed_place(b, g.home)
        if where is None:
            return self._explore(sit, g.home)
        name = self._name(b, g.item or "")
        if where == b.location_of(sit.agent):
            choice = self._pick(sit, f"仔细看看{name}还在不在", Op.INSPECT, g.home) \
                or self._pick(sit, f"仔细看看{name}还在不在", Op.INSPECT, where)
        else:
            choice = self._go_towards(sit, where, f"去{self._name(b, where)}看看{name}还在不在")
        return Choice(choice.index, choice.rationale, "explore") if choice else None

    def _held_by_other(self, sit: Situation, g: Goal, holder: str, owners: tuple[str, ...]) -> Choice | None:
        """东西在别人身上：失主亲自去讨要；其他守护者去报告失主。说过的话不重复说。"""
        b, me = sit.beliefs, sit.agent
        here = b.location_of(me)
        name = self._name(b, g.item)
        if me in owners:
            holder_place = believed_place(b, holder)
            if holder_place is None:
                # 不知道那人在哪：向在场的人打听
                for p, sk in sorted(b.entities.items()):
                    if sk.kind == Kind.PERSON and p not in (me, holder) and b.location_of(p) == here \
                            and not self._did_recently(sit, Op.ASK, p):
                        q = Fact(Proposition.rel(holder, Rel.AT, None), True)
                        return self._pick(sit, f"打听{self._name(b, holder)}在哪里", Op.ASK, p, topic=q)
                return self._explore(sit, holder)       # 没人可问：凭自己的地图去找那人
            if holder_place != here:
                return self._go_towards(sit, holder_place, f"去找{self._name(b, holder)}讨回{name}")
            if not self._did_recently(sit, Op.ASK, holder):
                q = Fact(Proposition.rel(g.item, Rel.AT, None), True)
                return self._pick(sit, f"当面质问{self._name(b, holder)}", Op.ASK, holder, topic=q)
            return None
        fact = Fact(Proposition.rel(g.item, Rel.AT, holder), True)
        for owner in owners:
            if owner == me or self._said(sit, owner, fact):
                continue
            if b.location_of(owner) == here:
                return self._pick(sit, f"向失主报告{name}的下落", Op.TELL, owner, topic=fact)
            where = believed_place(b, owner)
            choice = self._go_towards(sit, where, f"去告诉失主{name}在谁那里") if where else self._explore(sit, owner)
            if choice is not None:
                return choice
        return None

    def _acquire(self, sit: Situation, g: Goal) -> Choice | None:
        if g.item is None:
            return None
        b, me = sit.beliefs, sit.agent
        loc = self._whereabouts(sit, g.item)
        if loc == me:
            return None
        if loc is None or believed_place(b, loc) is None:
            return self._explore(sit, g.item)          # 不知道在哪：凭自己的地图去找，而不是原地干等
        if believed_place(b, loc) == b.location_of(me):
            return self._pick(sit, f"悄悄拿走{self._name(b, g.item)}", Op.TAKE, g.item, manner=Manner.CAREFUL)
        return self._go_towards(sit, believed_place(b, loc), f"去找{self._name(b, g.item)}")

    def _deliver(self, sit: Situation, g: Goal) -> Choice | None:
        if g.item is None:
            return None
        b, me = sit.beliefs, sit.agent
        if b.location_of(g.item) != me:
            return self._acquire(sit, g)
        if g.recipient is None:
            return None
        if b.location_of(g.recipient) == b.location_of(me):
            return self._pick(sit, f"把{self._name(b, g.item)}交给{self._name(b, g.recipient)}", Op.GIVE,
                              g.recipient, g.item)
        where = believed_place(b, g.recipient)
        if where is None:
            return self._explore(sit, g.recipient)
        return self._go_towards(sit, where, f"去找{self._name(b, g.recipient)}")

    # ------------------------------------------------------------
    #  查探：守护范围内传来响动
    # ------------------------------------------------------------

    def _investigate_noise(self, sit: Situation) -> Choice | None:
        b = sit.beliefs
        guarded = {believed_place(b, g.home) for g in sit.profile.goals if g.kind == GoalKind.PROTECT and g.home}
        for ep in reversed(b.episodes):
            if ep.modality == Modality.SOUND and sit.now - ep.tick <= 3 and ep.event.place in guarded:
                choice = self._go_towards(sit, ep.event.place, f"{self._name(b, ep.event.place)}有动静，去看看")
                if choice is not None:
                    return choice
        return None
