"""
[INPUT]: 依赖 langgraph 的 StateGraph / Send / Runtime，agents/npc_graph 的 build_npc_graph / NpcContext / checkpoint_serde
[OUTPUT]: 对外提供 Deliberation（一次决策的可解释轨迹）、Orchestrator（一个 tick 内多个 NPC 的并行决策）
[POS]: agents 的多智能体编排：基于同一版本观察，把需要决策的角色扇出（Send）并行运行各自的决策流程，汇总结构化意图。
       它只产出意图，从不写世界——提交与冲突结算归 WorldAuthority
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

from __future__ import annotations

import operator
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Annotated, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime
from langgraph.types import Send

from tianlong.agents.npc_graph import NpcContext, build_npc_graph, checkpoint_serde
from tianlong.core import Intent


@dataclass(frozen=True, slots=True)
class Deliberation:
    agent: str
    intent: Intent
    rationale: str
    recalled: tuple[str, ...]
    considered: int


class _TickState(TypedDict, total=False):
    agents: list[str]
    deliberations: Annotated[list[Deliberation], operator.add]


@dataclass
class _TickContext:
    npcs: Mapping[str, NpcContext]


class Orchestrator:
    def __init__(self) -> None:
        self.checkpointer = InMemorySaver(serde=checkpoint_serde())
        self.npc_graph = build_npc_graph(self.checkpointer)
        self.graph = self._build()

    def _build(self):
        npc_graph = self.npc_graph

        def fan_out(state: _TickState) -> list[Send]:
            return [Send("deliberate", {"agent": a}) for a in state["agents"]]

        def deliberate(state: dict, runtime: Runtime[_TickContext]) -> _TickState:
            ctx = runtime.context.npcs[state["agent"]]
            port = ctx.port
            thread = {"configurable": {"thread_id": f"{port.world_id}:{port.branch_id}:{port.agent}:{port.version}"}}
            out = npc_graph.invoke({"agent": port.agent}, config=thread, context=ctx)
            d = Deliberation(port.agent, out["intent"], out["rationale"],
                             tuple(out.get("recent", [])) + tuple(out.get("related", [])),
                             len(out["candidates"]))
            return {"deliberations": [d]}

        g = StateGraph(_TickState, context_schema=_TickContext)
        g.add_node("deliberate", deliberate)
        g.add_conditional_edges(START, fan_out, ["deliberate"])
        g.add_edge("deliberate", END)
        return g.compile()

    def decide(self, npcs: Mapping[str, NpcContext]) -> list[Deliberation]:
        if not npcs:
            return []
        out = self.graph.invoke({"agents": sorted(npcs), "deliberations": []}, context=_TickContext(npcs))
        return sorted(out["deliberations"], key=lambda d: d.agent)
