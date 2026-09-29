"""
[INPUT]: 依赖 kernel 的 Kernel，cognition 的 BeliefStore / candidates，agents 的 ScriptedPolicy / HeuristicPredictor / Situation，
         scenarios/procedural 的 random_scenario，learning/samples 的 env_sample / agent_sample
[OUTPUT]: 对外提供 RolloutConfig、Rollouts、collect()
[POS]: learning 的数据工厂：在程序化小世界里用“脚本策略 + 分层随机探索”行动，由内核实际执行，
       同时记录环境样本与角色样本。每个 tick 只让一个角色行动，使环境标签只反映这一个行动的后果
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass, field

from tianlong.agents.policies import ScriptedPolicy, Situation
from tianlong.agents.predictors import HeuristicPredictor
from tianlong.cognition import BeliefStore, Candidate, candidates
from tianlong.core import Op, Outcome, derive_seed, make_id
from tianlong.kernel import Kernel
from tianlong.learning.samples import Sample, agent_sample, env_sample
from tianlong.scenarios.procedural import random_scenario


@dataclass(frozen=True)
class RolloutConfig:
    worlds: int = 300
    steps: int = 24
    epsilon: float = 0.8      # 随机探索比例：覆盖失败与少见行动
    seed: int = 0


@dataclass
class Rollouts:
    env: list[Sample] = field(default_factory=list)
    agent: list[Sample] = field(default_factory=list)
    world_of: list[int] = field(default_factory=list)   # 样本所属世界：按世界切分训练/验证，检验对新布局的泛化


def _stratified(rng: random.Random, cands: tuple[Candidate, ...]) -> Candidate:
    """先均匀选操作、再在该操作内选对象：言语候选数量庞大，不分层就会淹没其余行动；等待交给脚本策略去产生。"""
    by_op: dict = defaultdict(list)
    for c in cands:
        if c.op != Op.WAIT or len(cands) == 1:
            by_op[c.op].append(c)
    op = rng.choice(sorted(by_op, key=lambda o: o.value))
    return rng.choice(by_op[op])


def collect(cfg: RolloutConfig) -> Rollouts:
    kernel, policy, predictor = Kernel(), ScriptedPolicy(), HeuristicPredictor()
    out = Rollouts()
    for w in range(cfg.worlds):
        sc = random_scenario(cfg.seed * 1_000_003 + w)
        rng = random.Random(derive_seed("rollout", cfg.seed, w))
        state = sc.state
        stores = {a: BeliefStore(a, trust=dict(p.trust)).revise_all(sc.priors.get(a, ()))[0]
                  for a, p in sc.profiles.items()}
        agents = sorted(sc.profiles)
        for _ in range(cfg.steps):
            actor = rng.choice(agents)
            profile = sc.profiles[actor]
            interests = list(profile.interests())
            cands = candidates(stores[actor], interests)
            if rng.random() < cfg.epsilon:
                cand = _stratified(rng, cands)
            else:
                preds = tuple(predictor.predict(stores[actor], state.clock, cands, interests))
                cand = cands[policy.choose(Situation(actor, profile, stores[actor], state.clock, cands, preds)).index]
            intent = cand.to_intent(make_id("int", sc.world_id, actor, state.version), actor, state.version)
            result = kernel.step(state, [intent])
            success = any(e.intent.id == intent.id and e.outcome == Outcome.SUCCESS for e in result.events)
            new_stores = dict(stores)
            for o in result.observations:
                new_stores[o.observer], _ = new_stores[o.observer].revise(o.percept)
            out.env.append(env_sample(state, result.state, actor, cand, success))
            out.agent.append(agent_sample(stores[actor], new_stores[actor], state.clock, actor, cand, success))
            out.world_of.append(w)
            state, stores = result.state, new_stores
    return out
