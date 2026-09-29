"""
[INPUT]: 依赖 langgraph 的 StateGraph / Send / Runtime，agents/npc_graph 的 build_npc_graph / NpcContext / checkpoint_serde
[OUTPUT]: 对外提供 Deliberation（一次决策的可解释轨迹）、Orchestrator（一个 tick 内多个 NPC 的并行决策；决策轨迹检查点须显式开启，开启后按条数修剪）
[POS]: agents 的多智能体编排：基于同一版本观察，把需要决策的角色扇出（Send）并行运行各自的决策流程，汇总结构化意图。
       它只产出意图，从不写世界——提交与冲突结算归 WorldAuthority。
       检查点默认关闭：剖析显示它占 NPC 决策耗时的 65~75%，而会话从不读它（重试靠意图 ID 去重，不靠恢复轨迹）；
       要看决策轨迹（调试、测试）就显式传 checkpoint=True
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

from __future__ import annotations

import operator
from collections import deque
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
    """checkpoint：是否把每次决策的轨迹写进 LangGraph 检查点（默认否——没人读，却是决策耗时的大头）。
    keep_threads：开启检查点时保留最近多少条决策轨迹。每个角色每个 tick 一条线程，不修剪就会在长局中无限增长。"""

    def __init__(self, keep_threads: int = 64, checkpoint: bool = False) -> None:
        self.checkpointer: InMemorySaver | None = InMemorySaver(serde=checkpoint_serde()) if checkpoint else None
        self.npc_graph = build_npc_graph(self.checkpointer)
        self.graph = self._build()
        self.keep_threads = keep_threads
        self._threads: deque[str] = deque()

    @staticmethod
    def thread_id(port) -> str:
        return f"{port.world_id}:{port.branch_id}:{port.agent}:{port.version}"

    def _build(self):
        npc_graph = self.npc_graph
        traced = self.checkpointer is not None

        def fan_out(state: _TickState) -> list[Send]:
            return [Send("deliberate", {"agent": a}) for a in state["agents"]]

        def deliberate(state: dict, runtime: Runtime[_TickContext]) -> _TickState:
            ctx = runtime.context.npcs[state["agent"]]
            port = ctx.port
            config = {"configurable": {"thread_id": Orchestrator.thread_id(port)}} if traced else None
            out = npc_graph.invoke({"agent": port.agent}, config=config, context=ctx)
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
        if self.checkpointer is not None:
            for ctx in npcs.values():
                tid = self.thread_id(ctx.port)
                if tid not in self._threads:
                    self._threads.append(tid)
            while len(self._threads) > self.keep_threads:
                self.checkpointer.delete_thread(self._threads.popleft())
        return sorted(out["deliberations"], key=lambda d: d.agent)
