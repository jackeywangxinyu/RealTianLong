"""
[INPUT]: 依赖 runtime/authority 的 WorldAuthority / Settlement，runtime/versions 的 current_versions / check_save，
         agents 的 Orchestrator / NpcContext / AgentPort / Scheduler / Policy / OutcomePredictor，
         memory 的 QdrantMemoryIndex / Recall / MemoryIndexer / MemoryScope，language 的 IntentParser / Narrator / Speaker / LLMClient，
         language/render 的 Rendered / RenderStatus，persistence 的 WorldStore / InMemoryWorldStore / WorldRef / TurnEnvelope /
         RequestConflict / VersionConflict，scenarios 的 Scenario，cognition 的 Candidate / believed_place，
         language/templates 的 render_fact（读档开场），memory/view 的 MemoryView（NPC 的长期记忆摘要，增量汇总——水位含边界、按记录 ID 去重，与读档后重建逐项相同）
[OUTPUT]: 对外提供 GameSession（可玩会话：读档接续并恢复调度标记与已描写实体、请求幂等、存档版本闸门）、
          TurnReport（一回合的全部产物：世界侧结果与文字侧结果分开记录，含分阶段耗时）
[POS]: runtime 的装配中心：一回合 = 解析玩家输入 → 基于同一版本扇出 NPC 决策 → 权威结算（同一事务附上请求进度与会话运行态）
       → 同步记忆索引 → 按玩家视角叙述（过语义闸门）→ 幂等记下叙述。
       带 request_id 的请求：同 ID 同内容返回既有结果、不再结算；同 ID 异内容抛 RequestConflict；提交后崩溃的重试只重写文字，
       多 tick 等待中途崩溃的重试只走剩下的 tick。请求绑定由存储在提交内检查（查询与结算之间没有可钻的空隙）：
       并发的重复投递只有一次能提交某个 tick，被越过的一方即停、不多走一个 tick，以落库的那一份为准返回，
       对方尚未走完时只给出目前为止的文字、不落库。建档后尚无提交就读档，开场已描写的实体按开场规则补回。
       CLI、测试、未来的 Web 前端都只和它打交道
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass, field, replace

from tianlong.agents.npc_graph import NpcContext
from tianlong.agents.orchestrator import Deliberation, Orchestrator
from tianlong.agents.policies import Policy, ScriptedPolicy
from tianlong.agents.port import AgentPort
from tianlong.agents.predictors import HeuristicPredictor, OutcomePredictor
from tianlong.agents.scheduler import Scheduler
from tianlong.cognition import BeliefStore, Candidate
from tianlong.cognition.navigation import believed_place
from tianlong.core import (
    Event,
    Fact,
    Intent,
    Modality,
    Op,
    Percept,
    Rel,
    clock_label,
    digest,
    make_id,
    minutes_until_night,
)
from tianlong.language.llm import LLMClient
from tianlong.language.narrator import Narrator, lore_keys
from tianlong.language.parser import IntentParser, Parsed
from tianlong.language.render import Rendered, RenderStatus
from tianlong.language.speaker import LLMSpeaker, Speaker, TemplateSpeaker
from tianlong.language.templates import render_fact
from tianlong.memory.index import MemoryIndex, MemoryScope, QdrantMemoryIndex
from tianlong.memory.indexer import MemoryIndexer
from tianlong.memory.recall import Recall
from tianlong.memory.view import MemoryView
from tianlong.persistence import (
    InMemoryWorldStore,
    RequestConflict,
    TurnEnvelope,
    VersionConflict,
    WorldRef,
    WorldStore,
)
from tianlong.runtime.authority import Settlement, WorldAuthority
from tianlong.runtime.versions import check_save, current_versions
from tianlong.scenarios import Scenario


@dataclass(frozen=True)
class TurnReport:
    clock: str
    parsed: Parsed
    narration: str
    advanced: bool                                    # False：输入没解析成行动，时间未推进
    events: tuple[Event, ...] = ()                    # 真相（调试用，不给玩家看）
    deliberations: tuple[Deliberation, ...] = ()      # NPC 的决策理由（调试用）
    settlement: Settlement | None = field(default=None, repr=False)
    timings: dict[str, float] = field(default_factory=dict)  # 各阶段耗时（毫秒）：系统成本可观测
    render: Rendered | None = None                    # 文字侧结果（来源 + 闸门结论）；None = 叙述取自已落库的结果
    request_id: str | None = None
    replayed: bool = False                            # True：本次调用没有推进世界，只返回（或补写文字）既有请求的结果


MAX_WAIT = 240   # 一次最多等四个时辰（240 分钟）


def _last(env: TurnEnvelope) -> int:
    """请求最后一次提交的世界版本（尚未提交任何 tick 时是开始时的版本）。"""
    return env.versions[-1] if env.versions else env.start_version


def _bound(env: TurnEnvelope | None, payload: str) -> TurnEnvelope:
    """已落库的请求必须绑定同一份内容：同 ID 异内容抛 RequestConflict。"""
    if env is None:
        raise RuntimeError("请求进度不在存储里：会话与存储不一致")
    if env.payload_hash != payload:
        raise RequestConflict(f"请求 {env.request_id} 已绑定另一份内容：拒绝执行")
    return env


def _compact(percepts: tuple[Percept, ...]) -> tuple[Percept, ...]:
    """请求进度里的感知只留事件感知与最后一次环顾：多 tick 等待从不讲中途的所见（只有移动/查看才附所见，而它们只占一个 tick），
    等上四个时辰也不让每次提交整份重写的进度随 tick 平方增长。正常叙述与崩溃后的重写用的是同一份。"""
    last = max((i for i, p in enumerate(percepts) if p.modality == Modality.SCENE), default=-1)
    return tuple(p for i, p in enumerate(percepts) if p.modality != Modality.SCENE or i == last)


class _Stopwatch:
    def __init__(self) -> None:
        self._t = time.perf_counter()
        self.laps: dict[str, float] = {}

    def lap(self, name: str) -> None:
        now = time.perf_counter()
        self.laps[name] = round(self.laps.get(name, 0.0) + (now - self._t) * 1000, 1)   # 多 tick 时累加
        self._t = now


class GameSession:
    def __init__(
        self,
        scenario: Scenario,
        store: WorldStore | None = None,
        index: MemoryIndex | None = None,
        llm: LLMClient | None = None,
        branch_id: str = "main",
        policies: Mapping[str, Policy] | None = None,
        predictor: OutcomePredictor | None = None,
        max_candidates: int = 64,
        allow_migration: bool = False,
    ) -> None:
        if scenario.player is None:
            raise ValueError("场景没有玩家角色")
        self.scenario = scenario
        self.player: str = scenario.player
        store = store or InMemoryWorldStore()
        self.store: WorldStore = store
        self.index = index or QdrantMemoryIndex()
        self.recall = Recall(store, self.index)
        self.indexer = MemoryIndexer(store, self.index)
        ref = WorldRef(scenario.world_id, branch_id)
        self.versions = current_versions()
        self.migrated_from: dict[str, str] | None = None   # 显式迁移时记下旧档的版本表（不改写它，也不补写旧档未记录的信息）
        self.resumed = store.exists(ref)
        session_state: Mapping | None = None
        if self.resumed:
            # 读档：先过版本闸门；世界与认知来自权威存储；向量索引是派生数据，从经历记录重建
            stored = store.save_versions(ref)
            if check_save(stored, self.versions, allow_migration):
                self.migrated_from = dict(stored)
            self.authority = WorldAuthority(store, ref)
            for agent in scenario.profiles:
                self.index.upsert(store.recent_memories(ref, agent, 0))
            session_state = store.session_state(ref)
            if session_state is None:
                # 建档之后还没有任何提交：开场已描写过的实体只记在上一个会话的内存里，按开场同样的规则补回，免得首回合重讲一遍
                session_state = {"described": self._opening_keys()}
        else:
            self.authority = WorldAuthority.found(store, scenario, branch_id, versions=self.versions)
        self.orchestrator = Orchestrator()
        # 会话运行态随每次提交落库，读档原样恢复：否则读档那一刻所有 NPC 都“该决策了”，初见描写也会重来一遍
        self.scheduler = Scheduler()
        self._described: set[str] = set()   # 已向玩家描写过外观的实体：只在初见时描写
        self._restore(session_state)
        self.parser = IntentParser(llm, aliases=scenario.aliases)
        self.narrator = Narrator(llm, scenario.setting, scenario.lore, scenario.style, scenario.aliases)
        self._universe = frozenset(e.name for e in scenario.state.entities.values())  # 闸门拒绝用的名字全集
        self.speaker: Speaker = (LLMSpeaker(llm, universe=self._universe, aliases=scenario.aliases) if llm
                                 else TemplateSpeaker())
        # 长期记忆摘要的增量缓存（派生数据）：(水位 tick, 水位 tick 上已并入的记录 ID, 摘要)
        self._memory_views: dict[str, tuple[int, frozenset[str], MemoryView]] = {}
        self.policies = dict(policies or {})
        self.predictor = predictor or HeuristicPredictor()
        self.max_candidates = max_candidates   # 学得的策略按训练时的候选上限看世界

    # ------------------------------------------------------------
    #  读
    # ------------------------------------------------------------

    @property
    def ref(self):
        return self.authority.ref

    def beliefs(self, agent: str) -> BeliefStore:
        return self.authority.store.beliefs(self.ref, agent)

    def clock(self) -> str:
        return clock_label(self.authority.head().clock)

    def session_state(self) -> dict:
        """会话运行态：随每次世界提交一起落库的那一份（调度标记 + 已描写实体）。"""
        return {"scheduler": self.scheduler.to_state(), "described": sorted(self._described)}

    def _restore(self, state: Mapping | None) -> None:
        """会话运行态以落库的那一份为准：读档时，以及一次请求被同一请求的另一次投递越过之后。"""
        state = state or {}
        self.scheduler = Scheduler.from_state(state.get("scheduler", {}), self.scheduler.idle_interval)
        self._described = set(state.get("described", ()))

    def _opening_keys(self) -> list[str]:
        """开场讲的初始认知里应当描写外观的实体：新游戏的 intro() 描写它们，建档后尚无提交就读档时据此补回“已描写”。"""
        return lore_keys(self.player, self.scenario.priors.get(self.player, ()), self.scenario.lore)

    def _known(self, me: BeliefStore) -> frozenset[str]:
        return self._universe | {sk.name for sk in me.entities.values()}

    def intro(self) -> str:
        """开场：新游戏讲初始认知；读档讲玩家此刻以为的周遭（不是世界真相）。"""
        me = self.beliefs(self.player)
        if not self.resumed:
            prior = self.scenario.priors.get(self.player, ())
            fresh = self._opening_keys()
            self._described.update(fresh)
            return self.narrator.narrate(self.player, prior, me.entities, show_scene=True, fresh=fresh,
                                         known=self._known(me))
        here = believed_place(me, self.player)
        around = [
            render_fact(Fact(b.prop, True), me.entities, self.player, me="你")
            for b in me.sorted_beliefs()
            if b.holds and b.prop.predicate == Rel.AT.value and b.prop.subject != self.player
            and believed_place(me, b.prop.subject) == here
        ]
        where = me.sketch(here).name if here and me.sketch(here) else "某处"
        return f"（读档）你在{where}。" + ("；".join(around) if around else "")

    # ------------------------------------------------------------
    #  一回合
    # ------------------------------------------------------------

    def turn(self, text: str, request_id: str | None = None) -> TurnReport:
        """request_id 为 None 时行为与从前一致；给出时，请求以 request_id + 原文摘要绑定，重试不会二次结算。"""
        clock = _Stopwatch()
        payload = digest("turn", text)
        if request_id is not None:
            prior = self.store.request(self.ref, request_id)
            if prior is not None:
                return self._resume_request(_bound(prior, payload), text, clock)
        head = self.authority.head()
        me = self.beliefs(self.player)
        parsed = self.parser.parse(text, me)
        clock.lap("parse")
        if parsed.candidate is None:
            render = Rendered(parsed.clarification or "……", RenderStatus.TEMPLATE)
            return TurnReport(clock_label(head.clock), parsed, render.text, advanced=False, timings=clock.laps,
                              render=render, request_id=request_id)
        intent = parsed.candidate.to_intent(self._intent_id(self.player, head.version), self.player, head.version,
                                            parsed.utterance)
        env = TurnEnvelope(request_id or "", payload, intent, self._ticks_for(parsed, head.clock), head.version,
                           head.clock, source=parsed.source)
        env, events, deliberations, settlement = self._advance(env, clock, persist=request_id is not None)
        if not env.done:
            # 被越过：同一请求的另一次投递抢先推进了世界——以落库的那一份为准，本次不再多走
            stored = self.store.request(self.ref, env.request_id)
            return self._resume_request(_bound(stored, payload), text, clock, execute=False,
                                        committed=settlement is not None)
        render = self._render(env, text, clock)
        if request_id is not None:
            self.store.record_render(self.ref, request_id, render.text)
        return TurnReport(clock_label(head.clock), parsed, render.text, True, tuple(events),
                          tuple(deliberations), settlement, clock.laps, render, request_id)

    def _resume_request(self, env: TurnEnvelope, text: str, clock: _Stopwatch, execute: bool = True,
                        committed: bool = False) -> TurnReport:
        """既有请求：世界侧没走完且没人越过它就只走剩下的 tick；文字没落库就按已持久化的感知重写；否则原样返回。
        execute=False：本次执行已被同一请求的另一次投递越过，只以落库的那一份为准、不再推进世界；
        那一份若还没走完（对方仍在进行），先给出目前为止的文字但不落库——终稿由走完它的那一方写。
        committed：本次调用在被越过之前是否已提交过 tick（决定 replayed）。"""
        parsed = Parsed(Candidate.of(env.intent), env.intent.utterance, source=env.source, repeat=env.planned_ticks)
        deliberations: list[Deliberation] = []
        settlement = None
        stuck = False
        if execute and not env.done:
            env, version = self._progress(env.request_id)
            if not env.done and version == _last(env):
                env, _, deliberations, settlement = self._advance(env, clock, persist=True)
                if not env.done:                    # 续跑途中又被另一次投递越过
                    env = _bound(self.store.request(self.ref, env.request_id), env.payload_hash)
            else:
                stuck = not env.done                # 别的写入者越过了它：这个请求再也走不完，按已走的 tick 定稿
        render = None
        narration = env.narration
        if narration is None or settlement is not None:
            render = self._render(env, text, clock)
            narration = render.text
            if env.done or stuck:
                self.store.record_render(self.ref, env.request_id, render.text)
                stored = self.store.request(self.ref, env.request_id)   # 先写者为准：返回的一定是已落库的那一份
                narration = stored.narration if stored and stored.narration else render.text
        wanted = {v - 1 for v in env.versions}       # 本请求各 tick 的意图都基于提交前的那个版本
        events = tuple(e for e in self.store.events(self.ref) if e.intent.based_on in wanted)
        return TurnReport(clock_label(env.start_clock), parsed, narration, True, events, tuple(deliberations),
                          settlement, clock.laps, render, env.request_id,
                          replayed=not (committed or settlement is not None))

    def _progress(self, request_id: str) -> tuple[TurnEnvelope, int]:
        """请求进度与世界版本的一致快照：前后两次读到同一份进度，夹在中间读到的版本才与它相符
        （每个 tick 的进度与世界同一事务落库、只增不减）。免得把另一次投递刚提交的 tick 误当成“别人越过了它”。"""
        env = self.store.request(self.ref, request_id)
        while True:
            version = self.authority.head().version
            again = self.store.request(self.ref, request_id)
            if again.versions == env.versions:
                return again, version
            env = again

    def _advance(self, env: TurnEnvelope, clock: _Stopwatch,
                 persist: bool) -> tuple[TurnEnvelope, list[Event], list[Deliberation], Settlement | None]:
        """一次输入可能跨越多个 tick（“等到天黑”），身边一有动静就停下；每个 tick 的进度与世界同一事务落库。
        带请求时被越过即停、不再多走：世界已不在本请求最后提交的版本上，或这一 tick 的提交被拒（同一请求的另一次投递抢先）。
        此时返回的进度未完结，内存里的会话运行态改回落库的那一份，调用方以落库的请求进度为准。"""
        events: list[Event] = []
        deliberations: list[Deliberation] = []
        settlement = None
        while not env.done:
            now = self.authority.head()
            if persist and env.versions and now.version != env.versions[-1]:
                break
            intent = replace(env.intent, id=self._intent_id(self.player, now.version), based_on=now.version)
            try:
                settlement, delibs, env = self._tick(intent, env, clock, persist)
            except (VersionConflict, RequestConflict):
                if not persist or self.store.request(self.ref, env.request_id) is None:
                    raise                            # 与本请求无关的写入者抢先：原样抛出
                break
            events += settlement.events
            deliberations += delibs
        if not env.done:
            self._restore(self.store.session_state(self.ref))
        return env, events, deliberations, settlement

    def _render(self, env: TurnEnvelope, text: str, clock: _Stopwatch) -> Rendered:
        """只依据已持久化的请求进度渲染：提交后崩溃的重试写出的是同一回合的文字。"""
        me = self.beliefs(self.player)
        lapse = clock_label(env.ticks[-1]) if env.planned_ticks > 1 and env.ticks else ""
        render = self.narrator.narrate_rendered(self.player, env.percepts, me.entities, fresh=env.fresh, command=text,
                                                lapse=lapse, known=self._known(me))
        clock.lap("narrate")
        return render

    def _intent_id(self, agent: str, version: int) -> str:
        return make_id("int", self.ref.world_id, self.ref.branch_id, agent, version)

    def _ticks_for(self, parsed: Parsed, now: int) -> int:
        if parsed.candidate is None or parsed.candidate.op != Op.WAIT:
            return 1
        wanted = minutes_until_night(now) if parsed.until == "night" else parsed.repeat
        return max(1, min(wanted, MAX_WAIT))

    def _interrupted(self, percepts: tuple[Percept, ...]) -> bool:
        """有人在身边做了什么、说了什么、或传来响动——等待就此打住，让玩家决定。"""
        return any(p.event is not None and p.event.actor != self.player for p in percepts)

    def _tick(self, player_intent: Intent, env: TurnEnvelope, clock: _Stopwatch,
              persist: bool) -> tuple[Settlement, list[Deliberation], TurnEnvelope]:
        head = self.authority.head()
        due, routine = self._npc_split(head.clock)
        deliberations = self.orchestrator.decide(self._contexts(due, head.version, head.clock))
        clock.lap("npc_decide")
        intents = [player_intent, *(d.intent for d in deliberations)]
        intents += [Intent(self._intent_id(a, head.version), a, Op.WAIT, based_on=head.version) for a in routine]
        # 调度标记与已描写实体先在副本上推进，随世界同一事务落库；提交成功后才替换内存中的那份
        sched = Scheduler.from_state(self.scheduler.to_state(), self.scheduler.idle_interval)
        for d in deliberations:
            sched.record(d.agent, head.clock, d.intent)
        after: dict = {}

        def annotate(s: Settlement) -> tuple[TurnEnvelope | None, dict]:
            mine = tuple(o.percept for o in s.observations_of(self.player))
            fresh = tuple(k for k in lore_keys(self.player, mine, self.narrator.lore) if k not in self._described)
            done = len(env.versions) + 1 >= env.planned_ticks or self._interrupted(mine)
            progressed = replace(env, versions=(*env.versions, s.state.version), ticks=(*env.ticks, s.state.clock),
                                 percepts=_compact(env.percepts + mine), fresh=env.fresh + fresh, done=done)
            described = self._described | set(fresh)
            after.update(env=progressed, described=described)
            return (progressed if persist else None), {"scheduler": sched.to_state(), "described": sorted(described)}

        settlement = self.authority.settle(intents, annotate)
        if "env" not in after:
            if persist:     # 带请求：多半是同一请求的另一次投递抢先结算了这一 tick，交由 _advance 按“被越过”处理
                raise VersionConflict(f"版本 {head.version} 的意图早已由别的写入者结算")
            raise RuntimeError(f"版本 {head.version} 的意图早已结算过：会话与存储不一致")
        self.scheduler, self._described = sched, after["described"]
        clock.lap("settle")
        self.indexer.drain()
        clock.lap("index")
        return settlement, deliberations, after["env"]

    # ------------------------------------------------------------
    #  NPC 装配：每个角色只拿到自己的 port
    # ------------------------------------------------------------

    def _npc_split(self, now: int) -> tuple[list[str], list[str]]:
        due, routine = [], []
        for a in self.scenario.npcs:
            (due if self.scheduler.due(a, self.beliefs(a), now, self.scenario.profiles[a]) else routine).append(a)
        return due, routine

    def _memory_view(self, agent: str, now: int) -> MemoryView:
        """长期记忆摘要：从权威经历记录汇总（可重建的派生数据），按角色增量缓存。
        known_at 不是逐次提交唯一的（本 tick 末的环顾与下一次结算写下的记录同一个 tick）：水位含边界、按记录 ID 去重，
        与从全部记录重建（读档后）逐项相同。"""
        seen, ids, view = self._memory_views.get(agent, (0, frozenset(), MemoryView()))
        fresh = [m for m in self.authority.store.recent_memories(self.ref, agent, seen)
                 if m.known_at <= now and not (m.known_at == seen and m.id in ids)]
        if fresh:
            view = view.add(fresh, now)
            top = max(seen, max(m.known_at for m in fresh))
            ids = (ids if top == seen else frozenset()) | {m.id for m in fresh if m.known_at == top}
            self._memory_views[agent] = (top, ids, view)
        return view

    def _contexts(self, agents: list[str], version: int, now: int) -> dict[str, NpcContext]:
        out = {}
        for a in agents:
            scope = MemoryScope(self.ref.world_id, self.ref.branch_id, a, now)
            port = AgentPort(
                a, self.scenario.profiles[a], self.ref.world_id, self.ref.branch_id, version, now,
                beliefs=lambda a=a: self.beliefs(a),
                recall=lambda q, scope=scope: self.recall.recall(scope, q),
                memory=lambda a=a, now=now: self._memory_view(a, now),
            )
            out[a] = NpcContext(port, self.policies.get(a) or ScriptedPolicy(), self.predictor, self.speaker,
                                self.max_candidates)
        return out
