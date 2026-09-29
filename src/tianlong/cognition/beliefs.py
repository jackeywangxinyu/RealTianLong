"""
[INPUT]: 依赖 core 的 Proposition / Fact / Percept / Modality / EntitySketch / PerceivedEvent / Rel
[OUTPUT]: 对外提供 Belief / Episode / BeliefChange / BeliefStore（不可变的个人认知图）及其 revise() 修正规则、effective_confidence()
[POS]: cognition 的核心数据结构；每个角色一份，只由感知折叠而成——它可以过时、可以错、可以自相矛盾，这正是游戏需要保留的认知差异。
       surveyed / searched 记着“我上次看清、上次仔细翻查某个容纳者是什么时候”：探索与“还没找过哪里”只凭这份个人记录，
       不读地图真相
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field

from tianlong.core import EntitySketch, Modality, PerceivedEvent, Percept, Proposition, Rel

# ============================================================
#  可信度模型
#  亲身所得（自己做的、亲眼看的、环顾所见）= 1.0
#  他人说法 = 对说话者的信任度；听到响动不产生事实，只产生经历
# ============================================================

FIRSTHAND = frozenset({Modality.SELF, Modality.SIGHT, Modality.SCENE})
DEFAULT_TRUST = 0.6
EPISODE_CAPACITY = 12
# 易变事实（位置、锁状态）的可信度随时间衰减：半小时前亲眼所见，未必敌得过刚刚可靠之人的报告
VOLATILE_HALF_LIFE = 60


def effective_confidence(b: Belief, now: int) -> float:
    volatile = b.prop.is_attr or b.prop.predicate == Rel.AT.value
    if not volatile or now <= b.learned_at:
        return b.confidence
    return b.confidence * 0.5 ** ((now - b.learned_at) / VOLATILE_HALF_LIFE)


def confidence_of(modality: Modality, informant: str | None, trust: Mapping[str, float]) -> float:
    if modality in FIRSTHAND:
        return 1.0
    if modality == Modality.SPEECH and informant is not None:
        return trust.get(informant, DEFAULT_TRUST)
    return 0.5


@dataclass(frozen=True, slots=True)
class Belief:
    prop: Proposition
    holds: bool
    confidence: float
    modality: Modality
    learned_at: int
    informant: str | None = None

    @property
    def hearsay(self) -> bool:
        return self.modality not in FIRSTHAND


@dataclass(frozen=True, slots=True)
class Episode:
    """短期经历：最近感知到的事件（含只听到响动的片段），会成为 GNN 中的事件节点。"""

    tick: int
    modality: Modality
    event: PerceivedEvent
    informant: str | None = None


@dataclass(frozen=True, slots=True)
class BeliefChange:
    """一次修正中真正改变了的信念：before/after 其一可为 None。记忆层据此判断什么值得记住。"""

    before: Belief | None
    after: Belief | None


# ============================================================
#  BeliefStore
# ============================================================


@dataclass(frozen=True, slots=True)
class BeliefStore:
    owner: str
    entities: Mapping[str, EntitySketch] = field(default_factory=dict)
    beliefs: Mapping[Proposition, Belief] = field(default_factory=dict)
    episodes: tuple[Episode, ...] = ()
    trust: Mapping[str, float] = field(default_factory=dict)
    last_tick: int = 0
    surveyed: Mapping[str, int] = field(default_factory=dict)   # 容纳者 → 最近一次看清其直接内容的时刻（环顾或查看）
    searched: Mapping[str, int] = field(default_factory=dict)   # 容纳者 → 最近一次亲手仔细翻查（含藏匿物）的时刻

    # ------------------------------------------------------------
    #  查询：一律排序返回，保证特征构造与候选生成的确定性
    # ------------------------------------------------------------

    def knows(self, eid: str | None) -> bool:
        return eid is not None and eid in self.entities

    def sketch(self, eid: str) -> EntitySketch | None:
        return self.entities.get(eid)

    def believed(self, prop: Proposition) -> Belief | None:
        return self.beliefs.get(prop)

    def holds(self, prop: Proposition, min_confidence: float = 0.0) -> bool:
        b = self.beliefs.get(prop)
        return b is not None and b.holds and b.confidence >= min_confidence

    def positives(self, subject: str, predicate: str) -> tuple[Belief, ...]:
        """某槽位上所有“认为为真”的信念（可能多条：互相矛盾的说法并存）。"""
        found = [b for p, b in self.beliefs.items() if p.subject == subject and p.predicate == predicate and b.holds]
        now = self.last_tick
        return tuple(sorted(found, key=lambda b: (-effective_confidence(b, now), -b.learned_at, b.prop.sort_key())))

    def subjects(self, predicate: str, value: str) -> tuple[str, ...]:
        """反查：认为 (X, predicate, value) 为真的所有 X（例如谁拥有钥匙）。"""
        return tuple(sorted({p.subject for p, b in self.beliefs.items()
                             if b.holds and p.predicate == predicate and p.value == value}))

    def best(self, subject: str, predicate: str) -> Belief | None:
        ps = self.positives(subject, predicate)
        return ps[0] if ps else None

    def location_of(self, eid: str) -> str | None:
        b = self.best(eid, Rel.AT.value)
        return b.prop.value if b is not None else None  # type: ignore[return-value]

    def sorted_beliefs(self) -> tuple[Belief, ...]:
        return tuple(sorted(self.beliefs.values(), key=lambda b: b.prop.sort_key()))

    # ------------------------------------------------------------
    #  修正：把一条感知折叠进认知
    # ------------------------------------------------------------

    def revise(self, percept: Percept) -> tuple[BeliefStore, tuple[BeliefChange, ...]]:
        entities = dict(self.entities)
        for sk in percept.sketches:
            if sk.seen or sk.id not in entities:
                entities[sk.id] = sk      # 亲眼所见才更新外观；只闻其名不抹掉已见过的样子
        beliefs = dict(self.beliefs)
        changes: list[BeliefChange] = []
        conf = confidence_of(percept.modality, percept.informant, self.trust)
        now = max(self.last_tick, percept.tick)

        def put(b: Belief) -> None:
            old = beliefs.get(b.prop)
            if old is not None and effective_confidence(old, now) > b.confidence:
                return  # 低可信度的说法不覆盖（衰减后仍）更可信的认知
            beliefs[b.prop] = b
            if old is None or old.holds != b.holds:
                changes.append(BeliefChange(old, b))

        def drop(prop: Proposition) -> None:
            old = beliefs.pop(prop, None)
            if old is not None:
                changes.append(BeliefChange(old, None))

        for fact in percept.facts:
            b = Belief(fact.prop, fact.holds, conf, percept.modality, percept.tick, percept.informant)
            if fact.holds and fact.prop.functional:
                # 函数型槽位：可信度不高于新证据的旧值被取代；更可信的旧值保留——矛盾的说法由此并存
                for rival in self._slot_rivals(beliefs, fact.prop):
                    if effective_confidence(rival, now) <= conf:
                        drop(rival.prop)
            put(b)

        # ---- 负证据：看清了某个容纳者，却没看到原以为在那里的东西 ----
        seen = {f.prop for f in percept.facts if f.holds}
        for scope in percept.scopes:
            for prop, b in list(beliefs.items()):
                if not (b.holds and prop.predicate == Rel.AT.value and prop.value == scope) or prop in seen:
                    continue
                if percept.modality == Modality.SCENE and self._believes_hidden(beliefs, prop.subject):
                    continue  # 随意环顾看不见藏匿物；只有仔细查看的“看清”才能否定它
                put(Belief(prop, False, 1.0, percept.modality, percept.tick))

        surveyed, searched = self.surveyed, self.searched
        if percept.scopes:
            surveyed = {**surveyed, **{sc: max(percept.tick, surveyed.get(sc, percept.tick)) for sc in percept.scopes}}
            if percept.modality == Modality.SELF:      # 自己动手查看得来的“完整看清”：藏匿物也在其中
                searched = {**searched, **{sc: max(percept.tick, searched.get(sc, percept.tick))
                                           for sc in percept.scopes}}

        episodes = self.episodes
        if percept.event is not None:
            episodes = (*episodes, Episode(percept.tick, percept.modality, percept.event, percept.informant))
            episodes = episodes[-EPISODE_CAPACITY:]

        store = BeliefStore(self.owner, entities, beliefs, episodes, self.trust, now, surveyed, searched)
        return store, tuple(changes)

    def revise_all(self, percepts: Iterable[Percept]) -> tuple[BeliefStore, tuple[BeliefChange, ...]]:
        store, changes = self, []
        for p in percepts:
            store, cs = store.revise(p)
            changes.extend(cs)
        return store, tuple(changes)

    @staticmethod
    def _slot_rivals(beliefs: Mapping[Proposition, Belief], prop: Proposition) -> list[Belief]:
        return [
            b for p, b in beliefs.items()
            if b.holds and p.subject == prop.subject and p.predicate == prop.predicate and p != prop
        ]

    @staticmethod
    def _believes_hidden(beliefs: Mapping[Proposition, Belief], eid: str) -> bool:
        b = beliefs.get(Proposition.attr(eid, "hidden", True))
        return b is not None and b.holds

