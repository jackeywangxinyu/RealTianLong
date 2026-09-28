"""
[INPUT]: 依赖 tianlong.agents 的 orchestrator / npc_graph / port / scheduler / policies，tianlong.runtime.authority
[OUTPUT]: 智能体层测试：意图 ID 确定性、只在候选集中选择、端到端社会链条、完整流程下的认知隔离、调度节流
[POS]: tests 的智能体层；验证“LangGraph 编排流程，但不替角色接入全世界的信息”
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

from __future__ import annotations

import pytest

pytest.importorskip("langgraph")

from tianlong.agents.npc_graph import NpcContext  # noqa: E402
from tianlong.agents.orchestrator import Orchestrator  # noqa: E402
from tianlong.agents.port import AgentPort  # noqa: E402
from tianlong.agents.scheduler import Scheduler  # noqa: E402
from tianlong.core import Intent, Op, Rel, Relation, WorldState  # noqa: E402
from tianlong.persistence import InMemoryWorldStore  # noqa: E402
from tianlong.runtime.authority import WorldAuthority  # noqa: E402
from tianlong.scenarios import build_warehouse  # noqa: E402
from tianlong.scenarios.base import Scenario  # noqa: E402


def contexts(auth: WorldAuthority, sc: Scenario) -> dict[str, NpcContext]:
    s = auth.head()
    return {
        a: NpcContext(AgentPort(a, sc.profiles[a], auth.ref.world_id, auth.ref.branch_id, s.version, s.clock,
                                beliefs=lambda a=a: auth.store.beliefs(auth.ref, a)))
        for a in sc.npcs
    }


def run(sc: Scenario, ticks: int, player_first: Intent | None = None):
    auth = WorldAuthority.found(InMemoryWorldStore(), sc)
    orch = Orchestrator()
    log = []
    for t in range(ticks):
        ds = orch.decide(contexts(auth, sc))
        intents = [d.intent for d in ds]
        if t == 0 and player_first is not None:
            intents.append(player_first)
        auth.settle(intents)
        log.append(ds)
    return auth, log


def test_intent_ids_are_deterministic_per_version():
    sc = build_warehouse()
    auth = WorldAuthority.found(InMemoryWorldStore(), sc)
    orch = Orchestrator()
    a = orch.decide(contexts(auth, sc))
    b = orch.decide(contexts(auth, sc))
    assert [d.intent.id for d in a] == [d.intent.id for d in b]
    auth.settle([d.intent for d in a])
    assert auth.settle([d.intent for d in b]).replayed, "流程重试不会二次结算"


def test_emergent_social_chain():
    sc = build_warehouse()
    _, log = run(sc, 13, Intent("p0", "player", Op.TAKE, "key"))
    acts = [(d.agent, d.intent.op, d.intent.target) for ds in log for d in ds if d.intent.op != Op.WAIT]
    guard_first = [a for a in acts if a[0] == "guard"][0]
    assert guard_first == ("guard", Op.MOVE, "warehouse"), "听到响动先去查看"
    assert ("guard", Op.INSPECT, "player") in acts
    assert ("guard", Op.TELL, "captain") in acts, "向失主报告"
    assert ("captain", Op.ASK, "player") in acts, "失主当面质问"
    assert acts.index(("guard", Op.TELL, "captain")) < acts.index(("captain", Op.ASK, "player"))


def test_npcs_idle_without_stimulus():
    sc = build_warehouse()
    _, log = run(sc, 5)
    assert all(d.intent.op == Op.WAIT for ds in log for d in ds)


def _with_ledger_elsewhere() -> Scenario:
    sc = build_warehouse()
    s = sc.state
    rels = set(s.relations) - {Relation("ledger", Rel.AT, "storeroom")} | {Relation("ledger", Rel.AT, "warehouse")}
    return Scenario(sc.world_id, WorldState.build(s.seed, s.clock, s.entities.values(), rels), sc.profiles, sc.priors)


def test_guard_decisions_ignore_unobserved_truth():
    """账簿其实被挪到了仓库，但守卫从未看见——前三个 tick 他的决策必须完全一致。"""
    first = Intent("p0", "player", Op.TAKE, "key")
    _, a = run(build_warehouse(), 2, first)
    _, b = run(_with_ledger_elsewhere(), 2, first)
    pick = lambda log: [(d.intent.op, d.intent.target, d.rationale) for ds in log for d in ds if d.agent == "guard"]  # noqa: E731
    assert pick(a) == pick(b)


def test_scheduler_throttles_idle_agents():
    sc = build_warehouse()
    auth = WorldAuthority.found(InMemoryWorldStore(), sc)
    sched = Scheduler(idle_interval=10)
    store = auth.store.beliefs(auth.ref, "guard")
    now = auth.head().clock
    assert sched.due("guard", store, now)
    sched.record("guard", now, Intent("w", "guard", Op.WAIT))
    assert not sched.due("guard", store, now + 1)
    assert sched.due("guard", store, now + 10)
    sched.record("guard", now, Intent("m", "guard", Op.MOVE, "warehouse"))
    assert sched.due("guard", store, now + 1), "手头有事的人每个 tick 都要决策"


def test_decision_trace_is_checkpointed():
    sc = build_warehouse()
    auth = WorldAuthority.found(InMemoryWorldStore(), sc)
    orch = Orchestrator()
    ctxs = contexts(auth, sc)
    orch.decide(ctxs)
    port = ctxs["guard"].port
    snap = orch.npc_graph.get_state(
        {"configurable": {"thread_id": f"{port.world_id}:{port.branch_id}:guard:{port.version}"}})
    assert snap.values["intent"].actor == "guard" and snap.values["candidates"]
