"""
[INPUT]: 依赖 kernel 的 Kernel，persistence 的 WorldStore / CommitBatch / WorldRef / TurnEnvelope，cognition 的 BeliefStore / BeliefChange，
         memory/records 的 records_for，scenarios 的 Scenario，core 的 Intent / Event / Observation
[OUTPUT]: 对外提供 WorldAuthority（每个世界实例唯一的权威写入器，含 found() 建世界——信任与自己人随设定写进各人心里——并记下存档版本）、Settlement（一次结算的结果）、
          Annotate（提交前由调用方附上请求进度与会话运行态的钩子）
[POS]: runtime 的写入闸口：意图 → 内核裁定 → 认知折叠 → 经历提炼 → 附注（请求进度 + 会话运行态）→ 一次原子提交。
       角色决策可以并行，事实提交只在这里串行发生；重复提交同一意图返回既有结果，绝不二次扣钱或移动物品
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

from __future__ import annotations

import threading
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from tianlong.cognition import BeliefChange, BeliefStore
from tianlong.core import Event, Intent, Observation, WorldState
from tianlong.core.memories import MemoryRecord
from tianlong.kernel import Kernel
from tianlong.memory.records import records_for
from tianlong.persistence import CommitBatch, TurnEnvelope, WorldRef, WorldStore
from tianlong.scenarios import Scenario


@dataclass(frozen=True, slots=True)
class Settlement:
    state: WorldState
    events: tuple[Event, ...]
    observations: tuple[Observation, ...]
    belief_changes: Mapping[str, tuple[BeliefChange, ...]] = field(default_factory=dict)
    memories: tuple[MemoryRecord, ...] = ()
    replayed: bool = False   # True：这批意图早已结算过，本次只是返回既有结果

    def observations_of(self, agent: str) -> tuple[Observation, ...]:
        return tuple(o for o in self.observations if o.observer == agent)


# 提交前的附注：看到本次结算（尚未落库）后，给出随同一事务写入的请求进度与会话运行态（任一可为 None）
Annotate = Callable[[Settlement], tuple[TurnEnvelope | None, Mapping[str, Any] | None]]


class WorldAuthority:
    def __init__(self, store: WorldStore, ref: WorldRef, kernel: Kernel | None = None) -> None:
        self.store = store
        self.ref = ref
        self.kernel = kernel or Kernel()
        self._lock = threading.Lock()

    @classmethod
    def found(cls, store: WorldStore, scenario: Scenario, branch_id: str = "main",
              kernel: Kernel | None = None, versions: Mapping[str, str] | None = None) -> WorldAuthority:
        """建立世界：初始认知由场景给出的“过去的感知”折叠而成（信任与自己人随设定写进心里）；
        versions 记下建档时的存档/规则/属性/目标版本。"""
        ref = WorldRef(scenario.world_id, branch_id)
        beliefs = {
            a: BeliefStore(a, trust=dict(p.trust), allies=p.allies).revise_all(scenario.priors.get(a, ()))[0]
            for a, p in scenario.profiles.items()
        }
        store.create(ref, scenario.state, beliefs, versions)
        return cls(store, ref, kernel)

    def head(self) -> WorldState:
        return self.store.head(self.ref)

    def settle(self, intents: Sequence[Intent], annotate: Annotate | None = None) -> Settlement:
        """annotate 在提交前被调用一次（重放时不调用）：它返回的请求进度与会话运行态与世界变化同一事务落库。"""
        with self._lock:
            # ---- 1. 幂等：已结算过的意图直接返回既有事件 ----
            prior = {it.id: self.store.event_for_intent(self.ref, it.id) for it in intents}
            done = tuple(e for e in prior.values() if e is not None)
            fresh = [it for it in intents if prior[it.id] is None]
            if intents and not fresh:
                return Settlement(self.head(), done, (), replayed=True)

            # ---- 2. 内核裁定（纯函数）----
            state = self.head()
            result = self.kernel.step(state, fresh)

            # ---- 3. 认知折叠：按观察顺序逐条修正，同时提炼经历 ----
            by_agent: dict[str, list[Observation]] = defaultdict(list)
            for o in result.observations:
                by_agent[o.observer].append(o)
            beliefs: dict[str, BeliefStore] = {}
            changes: dict[str, tuple[BeliefChange, ...]] = {}
            memories: list[MemoryRecord] = []
            for agent in sorted(by_agent):
                store = self.store.beliefs(self.ref, agent)
                agent_changes: list[BeliefChange] = []
                for o in by_agent[agent]:
                    store, cs = store.revise(o.percept)
                    agent_changes.extend(cs)
                    memories.extend(records_for(self.ref.world_id, self.ref.branch_id, o, cs, store.entities))
                beliefs[agent] = store
                changes[agent] = tuple(agent_changes)

            # ---- 4. 附注 + 一次原子提交 ----
            settlement = Settlement(result.state, done + result.events, result.observations, changes, tuple(memories))
            request, session_state = annotate(settlement) if annotate is not None else (None, None)
            self.store.commit(CommitBatch(
                self.ref, state.version, result.state, result.events, result.observations, beliefs, tuple(memories),
                request, session_state,
            ))
            return settlement
