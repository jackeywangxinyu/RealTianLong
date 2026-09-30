"""
[INPUT]: 依赖 agents/policy_kit 的 PolicyKit / Situation / Choice / STALE，cognition/navigation 的 believed_place，
         cognition/agenda 的 SOFT_SOCIAL，core 的 Op / Kind / Manner / Social / HOSTILE_SOCIAL / derive_seed，
         core/profiles 的 Goal / GoalKind；
         下落不明时借 PolicyKit._explore 凭个人勘察记录去找，开口借 PolicyKit._say / _last_spoke
[OUTPUT]: 对外提供 MartialTactics（江湖行为积木：自救、还手、救治盟友、寻仇（先礼后兵）、守地（一次闯入只动一次手）、灭口、护人、
          见义出声）、reply_act()（按性情 × 态度 × 对方言语行为选回话的言语行为）、HOT / CALM / DEFIANT_SOCIAL /
          PARLEY_GRACE / PARLEY_LIMIT，并再导出 cognition/agenda 的 SOFT_SOCIAL
[POS]: agents 的武斗与人情层。每个方法都是“若满足条件则在候选集中选一项（或说一句只有言语行为的话），否则返回 None”，
       由 ScriptedPolicy 按优先级串联。判断全部来自信念与社交状态：谁受伤中毒、谁动过手、解药在谁身上、谁对我说过什么、
       我对谁怀着怎样的态度、谁自何时起在我眼前，都是角色自己看见、听见或经历过的——没看见就不知道。
       开口只是修辞（Choice.free，index 指向 WAIT）：学习层只认下标，看到的是等待；为此“叫阵之后多久动手”另有照面时长兜底，
       话没说出口（被当作等待）也不会让寻仇永远停在嘴上。“他服过软、且饶他”以认知里持久的 yielded 为准，
       不从 8 条的线索缓冲里重新推断——旁人闲谈把那条线索挤掉，也不会平白动手
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

from __future__ import annotations

import random

from tianlong.agents.policy_kit import STALE, Choice, PolicyKit, Situation
from tianlong.cognition.agenda import SOFT_SOCIAL
from tianlong.cognition.navigation import believed_place
from tianlong.core import HOSTILE_SOCIAL, Kind, Manner, Modality, Op, Social, derive_seed
from tianlong.core.profiles import Goal, GoalKind

HOT = 0.3            # 脾气 ≥ 此值：受不得激
CALM = -0.3          # 脾气 ≤ 此值：能忍
DEFIANT_SOCIAL = HOSTILE_SOCIAL | {Social.REFUSE}                           # 嘴硬
PARLEY_GRACE = 2     # 叫阵之后对方几个 tick 不应，便动手
PARLEY_LIMIT = 3     # 照面这么久还没叫成阵（话被当作等待、或一直腾不出空），也不再多等
WARN_EVERY = 5       # 守卫喝令离开的间隔


# ============================================================
#  回话：性情 × 态度 × 对方的言语行为
# ============================================================


def reply_act(incoming: Social | None, *, temper: float = 0.0, attitude: int = 0, hostile: bool = False,
              question: bool = False, chatty: float = 0.0, greeted: bool = False) -> Social:
    """确定性的回话表。hostile：我对此人怀着寻仇的目标；question：对方问的是一句不带命题的话；
    greeted：我已向此人见过礼——再被招呼就寒暄一句，不再郑重回礼。"""
    hot, calm = temper >= HOT, temper <= CALM
    cold = hostile or attitude <= -2
    if incoming in HOSTILE_SOCIAL:
        table = {
            Social.INSULT: Social.INSULT if hot else Social.EXPLAIN if calm else Social.THREATEN,
            Social.TAUNT: Social.INSULT if hot else Social.REMARK if calm else Social.TAUNT,
            Social.THREATEN: Social.THREATEN if hot else Social.EXPLAIN if calm else Social.REFUSE,
            Social.CHALLENGE: Social.AGREE if hot else Social.REFUSE,
            # 被呵斥：火爆或有仇的顶回去；心存好感或性子温吞的才应承；不冷不热的——嘴快的挖苦一句，其余回绝
            Social.COMMAND: (Social.REFUSE if hot or cold else Social.AGREE if attitude > 0 or calm
                             else Social.TAUNT if chatty >= 0.5 else Social.REFUSE),
        }
        return table[incoming]                                   # type: ignore[index]
    if incoming in SOFT_SOCIAL:
        if hot and (cold or attitude < 0):
            return Social.TAUNT                                  # 火爆的仇家：服软也要挖苦
        return Social.AGREE if attitude >= 0 and not cold else Social.REMARK
    if cold:
        return {Social.GREET: Social.TAUNT if hot else Social.REMARK, Social.FAREWELL: Social.TAUNT if hot else
                Social.FAREWELL, Social.REFUSE: Social.THREATEN if hot else Social.REMARK,
                Social.JOKE: Social.TAUNT if hot else Social.REMARK}.get(
            incoming, Social.REFUSE if question and hot else Social.REMARK)  # type: ignore[arg-type]
    warm = {
        Social.THANK: Social.AGREE, Social.PRAISE: Social.JOKE if chatty >= 0.5 else Social.AGREE,
        Social.GREET: (Social.JOKE if chatty >= 0.5 else Social.REMARK) if greeted else Social.GREET,
        Social.COMFORT: Social.AGREE, Social.FAREWELL: Social.FAREWELL,
        Social.JOKE: Social.JOKE if chatty >= 0.5 else Social.REMARK, Social.AGREE: Social.REMARK,
        Social.REFUSE: Social.COMMAND if hot else Social.REMARK, Social.EXPLAIN: Social.AGREE,
    }
    if question and incoming in (None, Social.EXPLAIN, Social.REMARK):
        return Social.EXPLAIN                                    # 被问到：知道的掌故就讲讲
    return warm.get(incoming, Social.REMARK)                     # type: ignore[arg-type]


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
        friends = self._friends(sit)
        for victim in sorted(friends):
            for foe in self._attackers_of(sit, victim):
                if foe in here and foe not in friends and not self._status(b, foe, "subdued"):
                    who = "我" if victim == me else self._name(b, victim)
                    return self._pick(sit, f"{self._name(b, foe)}对{who}动手，岂能袖手", Op.ATTACK, foe)
        return None

    # ------------------------------------------------------------
    #  救治：盟友中毒——手里有解药就用；解药在被制住的人身上就取出来；不知在谁身上就搜
    #  凭身体状态与物品下落的信念，而不是凭 12 条经历缓冲里还剩没剩下“谁动的手”：混战里那一条早被挤掉了
    # ------------------------------------------------------------

    def _heal_allies(self, sit: Situation) -> Choice | None:
        b = sit.beliefs
        here = self._persons_here(b)
        friends = self._friends(sit)
        patients = [p for p in sit.profile.allies if p in here and self._status(b, p, "poisoned")]
        helpless = [p for p in here if p not in friends and self._status(b, p, "subdued")]
        for patient in patients:
            for item in self._held(b):
                if self._cures(b, item) == "poisoned":
                    return self._pick(sit, f"用{self._name(b, item)}救{self._name(b, patient)}", Op.USE, patient, item)
            for p in helpless:
                cure = next((i for i, sk in sorted(b.entities.items()) if sk.kind == Kind.ITEM
                             and self._cures(b, i) == "poisoned" and b.location_of(i) == p), None)
                if cure is not None:
                    return self._pick(sit, f"从{self._name(b, p)}身上取出{self._name(b, cure)}", Op.TAKE, cure)
            for p in self._suspects(sit, patient, helpless):
                last = b.searched.get(p)
                if last is not None and sit.now - last <= STALE:
                    continue                  # 刚搜过、没搜出解药：不再翻同一个人
                return self._pick(sit, f"搜{self._name(b, p)}的身，找解药", Op.INSPECT, p)
        return None

    def _suspects(self, sit: Situation, patient: str, helpless: list[str]) -> list[str]:
        """被制住的外人里谁最可能带着解药：亲眼见他对病人动过手的在前，其次是我对他心怀恶感的，其余按名次。"""
        b = sit.beliefs
        seen = self._attackers_of(sit, patient, window=10 ** 6)
        return sorted(helpless, key=lambda p: (seen.index(p) if p in seen else len(seen), b.attitude(p), p))

    # ------------------------------------------------------------
    #  目标：寻仇（先礼后兵）/ 守地 / 灭口 / 护人
    # ------------------------------------------------------------

    def _hostile(self, sit: Situation, g: Goal) -> Choice | None:
        b = sit.beliefs
        foe = g.person
        if foe is None or self._status(b, foe, g.until) or self._status(b, foe, "subdued"):
            return None      # 气已出了
        if foe in self._persons_here(b):
            strike, word = self._parley(sit, foe)
            if not strike:
                return word  # 叫阵、挖苦、等他回话；或者 None——且饶他这一回
            return self._pick(sit, f"找{self._name(b, foe)}算账", Op.ATTACK, foe)
        where = believed_place(b, foe)
        if where is None:
            return self._explore(sit, foe)
        return self._go_towards(sit, where, f"去找{self._name(b, foe)}")

    def _parley(self, sit: Situation, foe: str) -> tuple[bool, Choice | None]:
        """先礼后兵：(是否动手, 不动手时说的话或 None)。
        头一回照面先叫阵；对方嘴硬、想走、或叫阵后 PARLEY_GRACE 个 tick 不应才动手；服软则火爆的人按确定性的机会照打不误，
        其余挖苦一句便罢手（态度坏到 -2 以下另当别论）。叫阵只叫一次：再照面（他跑过一回）就不必多言。
        “服过软”取认知里持久的 yielded（与线索里残存的服软取较新者）：线索缓冲滚掉了那一条，饶过的人照旧饶过。"""
        b, me, prof = sit.beliefs, sit.agent, sit.profile
        name = self._name(b, foe)
        if self._status(b, me, "subdued") or b.attitude(foe) <= -2:
            return True, None                                   # 自己被制（本也动不了手）或积怨已深：无须多言
        since = b.company.get(foe, sit.now)
        challenged = self._last_spoke(sit, foe, (Social.CHALLENGE,))
        start = since if challenged is None else min(since, challenged)
        said = b.cues_from(foe, start)
        if any(c.social in DEFIANT_SOCIAL for c in said):
            return True, None                                   # 嘴硬
        if any(c.social == Social.FAREWELL for c in said) or self._tried_to_leave(sit, foe, start):
            return True, None                                   # 想走
        soft = max([c.tick for c in said if c.social in SOFT_SOCIAL] + [b.yielded.get(foe, -1)])
        if soft >= start:
            roll = random.Random(derive_seed("parley", me, foe, soft)).random()
            if prof.temper >= HOT and roll < prof.temper:
                return True, None                               # 火爆脾气：服软也不饶
            mocked = self._last_spoke(sit, foe, (Social.TAUNT,))
            if mocked is None or mocked < soft:
                return False, self._say(sit, foe, Social.TAUNT, f"{name}服了软，挖苦他几句")
            return False, None                                  # 且饶他这一回
        if challenged is None:
            if sit.now - since >= PARLEY_LIMIT:
                return True, None                               # 照面已久，话没叫出口也罢
            return False, self._say(sit, foe, Social.CHALLENGE, f"先向{name}叫阵")
        if sit.now - challenged >= PARLEY_GRACE:
            return True, None                                   # 叫阵不应
        pressed = self._last_spoke(sit, foe, (Social.TAUNT,))
        if pressed is None or pressed <= challenged:
            return False, self._say(sit, foe, Social.TAUNT, f"{name}不吭声，再激他一激")
        return False, Choice(self._wait_index(sit), f"等{name}回话")

    @staticmethod
    def _tried_to_leave(sit: Situation, who: str, since: int) -> bool:
        """since 以来亲眼见他动身离开（或往门口去没走成）：抵达的片段不算。"""
        return any(ep.modality == Modality.SIGHT and ep.event.kind == Op.MOVE.value and ep.event.actor == who
                   and ep.tick >= since and ep.event.target != ep.event.place for ep in sit.beliefs.episodes)

    def _guard(self, sit: Situation, g: Goal) -> Choice | None:
        """守地：一次闯入只动一次手（没受伤的才打），此后只在他硬闯（想从这里过、或顶撞喝令）时再出手，
        其余时候喝令离开——受了伤、老老实实待着或正往回走的人不再挨打，更不会一解穴就又被点住。"""
        b = sit.beliefs
        if g.home is None:
            return None
        if self._here(b) != g.home:
            return self._go_towards(sit, g.home, f"回{self._name(b, g.home)}守着")
        place = self._name(b, g.home)
        for p in self._persons_here(b):
            if p in sit.profile.allies or self._status(b, p, "subdued"):
                continue
            name = self._name(b, p)
            since = b.company.get(p, sit.now)
            struck = self._last_did(sit, Op.ATTACK, p)
            warned = self._last_spoke(sit, p, (Social.COMMAND,))
            dealt = (struck is not None and struck >= since) or (warned is not None and warned >= since)
            if not dealt and not self._status(b, p, "wounded"):
                return self._pick(sit, f"{place}岂容{name}擅闯", Op.ATTACK, p)
            after = max(x for x in (since, struck, warned) if x is not None)
            if self._pushing(sit, p, after):
                return self._pick(sit, f"{name}还敢硬闯", Op.ATTACK, p)
            if warned is None or sit.now - warned >= WARN_EVERY:
                return self._say(sit, p, Social.COMMAND, f"喝令{name}离开{place}")
        return None

    def _pushing(self, sit: Situation, who: str, since: int) -> bool:
        """硬闯：since 之后想从这里动身（还在眼前，就是没走成），或顶撞了喝令。"""
        return self._tried_to_leave(sit, who, since + 1) or any(
            c.social in DEFIANT_SOCIAL for c in sit.beliefs.cues_from(who, since + 1))

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
    #  见义出声：眼见有人对自己人、自己要护的人、或已有好感的主角动手，自己又没有更要紧的事——喝一声住手
    # ------------------------------------------------------------

    def _witness(self, sit: Situation) -> Choice | None:
        b, me = sit.beliefs, sit.agent
        if sit.profile.is_player or self._status(b, me, "subdued"):
            return None
        here = set(self._persons_here(b))
        allies = set(sit.profile.allies)
        foes = {g.person for g in sit.profile.goals if g.kind == GoalKind.HOSTILE}
        guarded = {g.person for g in sit.profile.goals if g.kind == GoalKind.DEFEND and g.person}
        # 只替自己人、自己要护的人、以及对之已有好感的主角出声：满堂宾客不会人人替一个素不相识的书生出头
        fond = {sit.player} if sit.player and b.attitude(sit.player) >= 1 else set()
        protected = allies | guarded | fond
        for ep in reversed(b.episodes):
            ev = ep.event
            if ep.modality != Modality.SIGHT or ev.kind != Op.ATTACK.value or sit.now - ep.tick > 1:
                continue
            victim, foe = ev.target, ev.actor
            if victim not in protected or victim in foes or b.attitude(victim or "") < 0 \
                    or not foe or foe == me or foe in allies or foe not in here:
                continue
            last = self._last_spoke(sit, foe, (Social.COMMAND, Social.THREATEN))
            if last is not None and sit.now - last <= 3:
                continue
            social = Social.COMMAND if b.attitude(foe) >= -1 else Social.THREATEN
            return self._say(sit, foe, social, f"喝止{self._name(b, foe)}")
        return None

    # ------------------------------------------------------------

    @staticmethod
    def _friends(sit: Situation) -> set[str]:
        return {sit.agent, *sit.profile.allies,
                *(g.person for g in sit.profile.goals if g.person and g.kind == GoalKind.DEFEND)}

    @staticmethod
    def _last_did(sit: Situation, op: Op, target: str) -> int | None:
        ticks = [ep.tick for ep in sit.beliefs.episodes
                 if ep.modality == Modality.SELF and ep.event.kind == op.value and ep.event.target == target]
        return max(ticks) if ticks else None

    @staticmethod
    def _held(b) -> list[str]:
        return [i for i, sk in sorted(b.entities.items())
                if sk.kind == Kind.ITEM and b.location_of(i) == b.owner]
