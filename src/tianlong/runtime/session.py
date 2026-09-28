"""
[INPUT]: 依赖 runtime/authority 的 WorldAuthority，agents 的 Orchestrator / NpcContext / AgentPort / Scheduler / Policy / OutcomePredictor，
         memory 的 QdrantMemoryIndex / Recall / MemoryIndexer / MemoryScope，language 的 IntentParser / Narrator / Speaker / LLMClient，
         persistence 的 WorldStore / InMemoryWorldStore，scenarios 的 Scenario
[OUTPUT]: 对外提供 GameSession（可玩会话，支持读档：存储里已有该世界则接续并重建向量索引）、TurnReport（一回合的全部产物）
[POS]: runtime 的装配中心：一回合 = 解析玩家输入 → 基于同一版本扇出 NPC 决策 → 权威结算 → 同步记忆索引 → 按玩家视角叙述。
       CLI、测试、未来的 Web 前端都只和它打交道
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from tianlong.agents.npc_graph import NpcContext
from tianlong.agents.orchestrator import Deliberation, Orchestrator
from tianlong.agents.policies import Policy, ScriptedPolicy
from tianlong.agents.port import AgentPort
from tianlong.agents.predictors import HeuristicPredictor, OutcomePredictor
from tianlong.agents.scheduler import Scheduler
from tianlong.cognition import BeliefStore
from tianlong.cognition.navigation import believed_place
from tianlong.core import Event, Fact, Intent, Op, Rel, clock_label, make_id
from tianlong.language.llm import LLMClient
from tianlong.language.narrator import Narrator
from tianlong.language.parser import IntentParser, Parsed
from tianlong.language.speaker import LLMSpeaker, Speaker, TemplateSpeaker
from tianlong.language.templates import render_fact
from tianlong.memory.index import MemoryIndex, MemoryScope, QdrantMemoryIndex
from tianlong.memory.indexer import MemoryIndexer
from tianlong.memory.recall import Recall
from tianlong.persistence import InMemoryWorldStore, WorldRef, WorldStore
from tianlong.runtime.authority import Settlement, WorldAuthority
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
    ) -> None:
        if scenario.player is None:
            raise ValueError("场景没有玩家角色")
        self.scenario = scenario
        self.player: str = scenario.player
        store = store or InMemoryWorldStore()
        self.index = index or QdrantMemoryIndex()
        self.recall = Recall(store, self.index)
        self.indexer = MemoryIndexer(store, self.index)
        ref = WorldRef(scenario.world_id, branch_id)
        self.resumed = store.exists(ref)
        if self.resumed:
            # 读档：世界与认知来自权威存储；向量索引是派生数据，从经历记录重建
            self.authority = WorldAuthority(store, ref)
            for agent in scenario.profiles:
                self.index.upsert(store.recent_memories(ref, agent, 0))
        else:
            self.authority = WorldAuthority.found(store, scenario, branch_id)
        self.orchestrator = Orchestrator()
        self.scheduler = Scheduler()
        self.parser = IntentParser(llm)
        self.narrator = Narrator(llm)
        self.speaker: Speaker = LLMSpeaker(llm) if llm else TemplateSpeaker()
        self.policies = dict(policies or {})
        self.predictor = predictor or HeuristicPredictor()

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

    def intro(self) -> str:
        """开场：新游戏讲初始认知；读档讲玩家此刻以为的周遭（不是世界真相）。"""
        me = self.beliefs(self.player)
        if not self.resumed:
            prior = self.scenario.priors.get(self.player, ())
            return self.narrator.narrate(self.player, prior, me.entities, show_scene=True)
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

    def turn(self, text: str) -> TurnReport:
        head = self.authority.head()
        me = self.beliefs(self.player)
        parsed = self.parser.parse(text, me)
        if parsed.candidate is None:
            return TurnReport(clock_label(head.clock), parsed, parsed.clarification or "……", advanced=False)
        iid = make_id("int", self.ref.world_id, self.ref.branch_id, self.player, head.version)
        player_intent = parsed.candidate.to_intent(iid, self.player, head.version, parsed.utterance)
        return self._advance(parsed, player_intent)

    def _advance(self, parsed: Parsed, player_intent: Intent) -> TurnReport:
        head = self.authority.head()
        due, routine = self._npc_split(head.clock)
        deliberations = self.orchestrator.decide(self._contexts(due, head.version, head.clock))
        intents = [player_intent, *(d.intent for d in deliberations)]
        intents += [Intent(make_id("int", self.ref.world_id, self.ref.branch_id, a, head.version), a, Op.WAIT,
                           based_on=head.version) for a in routine]
        settlement = self.authority.settle(intents)
        for d in deliberations:
            self.scheduler.record(d.agent, head.clock, d.intent)
        self.indexer.drain()
        percepts = [o.percept for o in settlement.observations_of(self.player)]
        narration = self.narrator.narrate(self.player, percepts, self.beliefs(self.player).entities)
        return TurnReport(clock_label(head.clock), parsed, narration, True, settlement.events,
                          tuple(deliberations), settlement)

    # ------------------------------------------------------------
    #  NPC 装配：每个角色只拿到自己的 port
    # ------------------------------------------------------------

    def _npc_split(self, now: int) -> tuple[list[str], list[str]]:
        due, routine = [], []
        for a in self.scenario.npcs:
            (due if self.scheduler.due(a, self.beliefs(a), now) else routine).append(a)
        return due, routine

    def _contexts(self, agents: list[str], version: int, now: int) -> dict[str, NpcContext]:
        out = {}
        for a in agents:
            scope = MemoryScope(self.ref.world_id, self.ref.branch_id, a, now)
            port = AgentPort(
                a, self.scenario.profiles[a], self.ref.world_id, self.ref.branch_id, version, now,
                beliefs=lambda a=a: self.beliefs(a),
                recall=lambda q, scope=scope: self.recall.recall(scope, q),
            )
            out[a] = NpcContext(port, self.policies.get(a) or ScriptedPolicy(), self.predictor, self.speaker)
        return out
