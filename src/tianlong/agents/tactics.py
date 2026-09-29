"""
[INPUT]: 依赖 agents/policy_kit 的 PolicyKit / Situation / Choice，cognition/navigation 的 believed_place，
         core 的 Op / Kind，core/profiles 的 Goal / GoalKind；下落不明时借 PolicyKit._explore 凭个人勘察记录去找
[OUTPUT]: 对外提供 MartialTactics（江湖行为积木：自救、还手、救治盟友、寻仇、守地、灭口、护人）
[POS]: agents 的武斗与人情层。每个方法都是“若满足条件则在候选集中选一项，否则返回 None”，由 ScriptedPolicy 按优先级串联。
       判断全部来自信念：谁受伤中毒、谁动过手、解药在谁身上，都是角色自己看见或经历过的——没看见就不知道
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

from __future__ import annotations

from tianlong.agents.policy_kit import Choice, PolicyKit, Situation
from tianlong.cognition.navigation import believed_place
from tianlong.core import Kind, Manner, Op
from tianlong.core.profiles import Goal, GoalKind


class MartialTactics(PolicyKit):
    # ------------------------------------------------------------
    #  自救：中了毒、手里有对症的解药就先服下
    # ------------------------------------------------------------

    def _cure_self(self, sit: Situation) -> Choice | None:
        b, me = sit.beliefs, sit.agent
        for status in ("poisoned", "wounded"):
            if self._status(b, me, status):
                for item in self._held(b):
                    if self._cures(b, item) == status:
                        return self._pick(sit, f"服下{self._name(b, item)}", Op.USE, me, item)
        return None

    # ------------------------------------------------------------
    #  还手：自己或盟友刚挨了打，打人者就在眼前
    # ------------------------------------------------------------

    def _retaliate(self, sit: Situation) -> Choice | None:
        b, me = sit.beliefs, sit.agent
        here = set(self._persons_here(b))
        friends = {me, *sit.profile.allies, *(g.person for g in sit.profile.goals if g.person and g.kind == GoalKind.DEFEND)}
        for victim in sorted(friends):
            for foe in self._attackers_of(sit, victim):
                if foe in here and foe not in friends and not self._status(b, foe, "subdued"):
                    who = "我" if victim == me else self._name(b, victim)
                    return self._pick(sit, f"{self._name(b, foe)}对{who}动手，岂能袖手", Op.ATTACK, foe)
        return None

    # ------------------------------------------------------------
    #  救治：盟友中毒——手里有解药就用；解药在已被制住的下毒者身上就搜出来
    # ------------------------------------------------------------

    def _heal_allies(self, sit: Situation) -> Choice | None:
        b = sit.beliefs
        here = self._persons_here(b)
        patients = [p for p in (*sit.profile.allies,) if p in here and self._status(b, p, "poisoned")]
        for patient in patients:
            for item in self._held(b):
                if self._cures(b, item) == "poisoned":
                    return self._pick(sit, f"用{self._name(b, item)}救{self._name(b, patient)}", Op.USE, patient, item)
            for culprit in self._attackers_of(sit, patient, window=30):
                if culprit not in here or not self._status(b, culprit, "subdued"):
                    continue
                cure = next((i for i, sk in sorted(b.entities.items()) if sk.kind == Kind.ITEM
                             and self._cures(b, i) == "poisoned" and b.location_of(i) == culprit), None)
                if cure is not None:
                    return self._pick(sit, f"从{self._name(b, culprit)}身上取出{self._name(b, cure)}", Op.TAKE, cure)
                if not self._did_recently(sit, Op.INSPECT, culprit):
                    return self._pick(sit, f"搜{self._name(b, culprit)}的身，找解药", Op.INSPECT, culprit)
        return None

    # ------------------------------------------------------------
    #  目标：寻仇 / 守地 / 灭口 / 护人
    # ------------------------------------------------------------

    def _hostile(self, sit: Situation, g: Goal) -> Choice | None:
        b = sit.beliefs
        foe = g.person
        if foe is None or self._status(b, foe, g.until) or self._status(b, foe, "subdued"):
            return None      # 气已出了
        if foe in self._persons_here(b):
            return self._pick(sit, f"找{self._name(b, foe)}算账", Op.ATTACK, foe)
        where = believed_place(b, foe)
        if where is None:
            return self._explore(sit, foe)
        return self._go_towards(sit, where, f"去找{self._name(b, foe)}")

    def _guard(self, sit: Situation, g: Goal) -> Choice | None:
        b = sit.beliefs
        if g.home is None:
            return None
        if self._here(b) != g.home:
            return self._go_towards(sit, g.home, f"回{self._name(b, g.home)}守着")
        for p in self._persons_here(b):
            if p not in sit.profile.allies and not self._status(b, p, "subdued"):
                return self._pick(sit, f"{self._name(b, g.home)}岂容{self._name(b, p)}擅闯", Op.ATTACK, p)
        return None

    def _escape(self, sit: Situation, g: Goal) -> Choice | None:
        """悄悄走；只有四下无人、偏偏撞上一个外人时才灭口——满堂同门面前动手等于自投罗网。"""
        b = sit.beliefs
        if g.home is None or self._here(b) == g.home:
            return None
        outsiders = [p for p in self._persons_here(b)
                     if p not in sit.profile.allies and not self._status(b, p, "subdued")]
        started = self._here(b) != self._origin(sit)
        if started and len(outsiders) == 1:
            return self._pick(sit, f"被{self._name(b, outsiders[0])}撞见了，不能留活口", Op.ATTACK, outsiders[0])
        return self._go_towards(sit, g.home, f"趁夜悄悄往{self._name(b, g.home)}去", Manner.CAREFUL)

    @staticmethod
    def _origin(sit: Situation) -> str | None:
        """动身之前所在之处：最早一条关于自己位置的亲眼所见（出发地的同门不算“撞见”）。"""
        b = sit.beliefs
        mine = [x for x in b.beliefs.values() if x.prop.subject == b.owner and x.prop.predicate == "AT"]
        return min(mine, key=lambda x: x.learned_at).prop.value if mine else None  # type: ignore[return-value]

    def _defend(self, sit: Situation, g: Goal) -> Choice | None:
        b = sit.beliefs
        if g.person is None or g.person in self._persons_here(b):
            return None      # 在一起时由“还手”负责出头
        where = believed_place(b, g.person)
        if where is None:
            return self._explore(sit, g.person)
        return self._go_towards(sit, where, f"跟着{self._name(b, g.person)}")

    # ------------------------------------------------------------

    @staticmethod
    def _held(b) -> list[str]:
        return [i for i, sk in sorted(b.entities.items())
                if sk.kind == Kind.ITEM and b.location_of(i) == b.owner]

