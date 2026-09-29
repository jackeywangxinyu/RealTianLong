"""
[INPUT]: 依赖 tianlong.learning.rl 的 env / observation / rewards / module / train / policy，tianlong.scenarios
[OUTPUT]: 强化学习层测试：观测合乎空间、掩码只屏蔽空位、奖励语义（进展/失败/冤枉人）、模仿学习可学、学得的策略接入决策图、PPO 冒烟
[POS]: tests 的 RL 层；PPO 冒烟用例标记 slow（默认不跑，`pytest -m slow` 显式运行）
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("ray.rllib")

from tianlong.core import Intent, Op  # noqa: E402
from tianlong.core.profiles import Goal, GoalKind  # noqa: E402
from tianlong.kernel import Kernel  # noqa: E402
from tianlong.learning.rl.env import TianlongEnv  # noqa: E402
from tianlong.learning.rl.module import GraphPolicyNet  # noqa: E402
from tianlong.learning.rl.rewards import FALSE_ACCUSATION, potential, step_reward  # noqa: E402
from tianlong.learning.rl.train import behavior_clone, collect_demos, evaluate, net_policy  # noqa: E402
from tianlong.scenarios import build_warehouse  # noqa: E402

pytestmark = pytest.mark.rl


def test_observations_fit_space_and_mask_only_padding():
    env = TianlongEnv({"horizon": 5})
    obs, _ = env.reset(seed=11)
    for a in env.agents:
        assert env.observation_spaces[a].contains(obs[a])
        n_valid = int(obs[a]["action_mask"].sum())
        assert n_valid == len(env._cands[a]), "掩码 = 候选集，不多不少"
    for _ in range(5):
        obs, rew, term, trunc, info = env.step(env.expert_actions())
    assert trunc["__all__"] and set(rew) == set(env.agents)


def test_reward_semantics_on_warehouse():
    sc = build_warehouse()
    s0 = sc.state
    k = Kernel()
    guard_goals = (Goal(GoalKind.PROTECT, "key", home="table"),)
    thief_goals = (Goal(GoalKind.ACQUIRE, "key"),)
    r = k.step(s0, [Intent("t", "player", Op.TAKE, "key", based_on=0)])
    assert step_reward(s0, r.state, "player", thief_goals, r.events) > 0.5, "拿到想要的东西是真实进展"
    assert step_reward(s0, r.state, "guard", guard_goals, r.events) < -0.9, "守护的东西被拿走是真实损失"
    # 守卫搜一个身上没有钥匙的人 = 冤枉人
    s1 = k.step(s0, [Intent("m", "guard", Op.MOVE, "warehouse", based_on=0)]).state
    r2 = k.step(s1, [Intent("i", "guard", Op.INSPECT, "player", based_on=1)])
    assert step_reward(s1, r2.state, "guard", guard_goals, r2.events) <= -FALSE_ACCUSATION
    assert potential(s0, "guard", guard_goals[0]) == 1.0


def test_behavior_cloning_learns_expert():
    env = TianlongEnv({"horizon": 12})
    demos = collect_demos(env, 12, seed=3)
    net = GraphPolicyNet(32)
    logs = []
    behavior_clone(net, demos, 4, log=logs.append)
    acc = float(logs[-1].split("acc=")[1])
    assert acc > 0.8, logs
    m = evaluate(env, net_policy(net), 3)
    assert set(m) == {"mean_return", "return_ci95", "goal_rate", "goal_rate_ci95", "false_accusations_per_ep",
                      "attacks_per_ep"}
    assert m["return_ci95"][0] <= m["mean_return"] <= m["return_ci95"][1]


def test_learned_policy_plugs_into_npc_pipeline():
    pytest.importorskip("langgraph")
    from tianlong.agents.npc_graph import NpcContext
    from tianlong.agents.orchestrator import Orchestrator
    from tianlong.agents.port import AgentPort
    from tianlong.learning.rl.policy import LearnedPolicy
    from tianlong.persistence import InMemoryWorldStore
    from tianlong.runtime.authority import WorldAuthority

    sc = build_warehouse()
    auth = WorldAuthority.found(InMemoryWorldStore(), sc)
    s = auth.head()
    policy = LearnedPolicy(GraphPolicyNet(32))
    ctx = {a: NpcContext(AgentPort(a, sc.profiles[a], auth.ref.world_id, auth.ref.branch_id, s.version, s.clock,
                                   beliefs=lambda a=a: auth.store.beliefs(auth.ref, a)),
                         policy=policy, max_candidates=policy.spec.max_cands)
           for a in sc.npcs}
    ds = Orchestrator().decide(ctx)
    assert all("策略网络" in d.rationale for d in ds)
    auth.settle([d.intent for d in ds])   # 学得的策略产出的意图照样只能经内核结算


@pytest.mark.slow
def test_ppo_smoke():
    from tianlong.learning.rl.train import RLConfig, train_ppo
    net = train_ppo(RLConfig(ppo_iterations=1, train_batch=200, horizon=10, hidden=32), GraphPolicyNet(32),
                    log=lambda *_: None)
    env = TianlongEnv({"horizon": 10})
    obs, _ = env.reset(seed=1)
    act = net_policy(net)(env, obs)
    assert all(obs[a]["action_mask"][i] == 1 for a, i in act.items())
    assert np.isfinite(evaluate(env, net_policy(net), 2)["mean_return"])
