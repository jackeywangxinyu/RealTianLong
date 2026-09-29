"""
[INPUT]: 依赖 ray.rllib 的 MultiAgentEnv，gymnasium，kernel 的 Kernel，cognition 的 BeliefStore / candidates，
         agents 的 HeuristicPredictor / ScriptedPolicy / Situation / OutcomePredictor，scenarios/procedural 的 random_scenario，
         learning/rl 的 observation / rewards
[OUTPUT]: 对外提供 TianlongEnv（RLlib 多智能体环境）、MAX_PERSONS
[POS]: learning/rl 的训练环境：每个角色一个智能体，同一 tick 同时出招、由同一个内核统一结算——与线上完全相同的转移机制。
       观测只来自各自的认知图；奖励来自真实目标进展。expert_actions() 给出脚本策略的示范，供模仿学习与评测
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

from __future__ import annotations

import gymnasium as gym
from ray.rllib.env.multi_agent_env import MultiAgentEnv

from tianlong.agents.policies import ScriptedPolicy, Situation
from tianlong.agents.predictors import HeuristicPredictor, OutcomePredictor, Prediction
from tianlong.cognition import BeliefStore, Candidate, candidates
from tianlong.core import Op, derive_seed, make_id
from tianlong.kernel import Kernel
from tianlong.learning.rl.observation import ObsSpec, encode_observation, observation_space
from tianlong.learning.rl.rewards import goal_achieved, step_reward
from tianlong.scenarios.procedural import random_scenario

MAX_PERSONS = 3


class TianlongEnv(MultiAgentEnv):
    def __init__(self, config: dict | None = None) -> None:
        super().__init__()
        cfg = dict(config or {})
        self.obs_spec = ObsSpec(**cfg.get("spec", {}))
        self.horizon = int(cfg.get("horizon", 30))
        self.seed_base = int(cfg.get("seed", 0))
        self.predictor: OutcomePredictor = cfg.get("predictor") or _load_predictor(cfg.get("predictor_path"))
        self.possible_agents = [f"h{i}" for i in range(MAX_PERSONS)]
        self.agents: list[str] = []
        obs_space = observation_space(self.obs_spec)
        self.observation_spaces = {a: obs_space for a in self.possible_agents}
        self.action_spaces = {a: gym.spaces.Discrete(self.obs_spec.max_cands) for a in self.possible_agents}
        self.kernel = Kernel()
        self._expert = ScriptedPolicy()
        self._episodes = 0
        self._cands: dict[str, tuple[Candidate, ...]] = {}
        self._preds: dict[str, tuple[Prediction, ...]] = {}

    # ------------------------------------------------------------
    #  gym 接口
    # ------------------------------------------------------------

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        world_seed = seed if seed is not None else derive_seed("env", self.seed_base, self._episodes) % 1_000_000_007
        self._episodes += 1
        sc = random_scenario(world_seed, max_persons=MAX_PERSONS)
        self.scenario = sc
        self.state = sc.state
        self.stores = {a: BeliefStore(a, trust=dict(p.trust)).revise_all(sc.priors.get(a, ()))[0]
                       for a, p in sc.profiles.items()}
        self.agents = sorted(sc.profiles)
        self.t = 0
        return {a: self._observe(a) for a in self.agents}, {a: {} for a in self.agents}

    def step(self, action_dict: dict):
        before = self.state
        intents = []
        for a in self.agents:
            idx = int(action_dict.get(a, 0))
            cands = self._cands[a]
            cand = cands[idx] if 0 <= idx < len(cands) else Candidate(Op.WAIT)
            intents.append(cand.to_intent(make_id("int", self.scenario.world_id, a, before.version), a, before.version))
        result = self.kernel.step(before, intents)
        for o in result.observations:
            self.stores[o.observer], _ = self.stores[o.observer].revise(o.percept)
        self.state = result.state
        self.t += 1
        rewards = {a: step_reward(before, self.state, a, self.scenario.profiles[a].goals, result.events)
                   for a in self.agents}
        done = self.t >= self.horizon
        obs = {a: self._observe(a) for a in self.agents}
        infos = {a: {"achieved": self.achieved(a)} for a in self.agents}
        return obs, rewards, {"__all__": False}, {**{a: done for a in self.agents}, "__all__": done}, infos

    # ------------------------------------------------------------
    #  观测、示范、评测
    # ------------------------------------------------------------

    def _observe(self, agent: str):
        store = self.stores[agent]
        profile = self.scenario.profiles[agent]
        interests = list(profile.interests())
        cands = candidates(store, interests, self.obs_spec.max_cands)
        preds = tuple(self.predictor.predict(store, self.state.clock, cands, interests))
        self._cands[agent], self._preds[agent] = cands, preds
        return encode_observation(store, self.state.clock, profile, cands, preds, self.obs_spec)

    def expert_actions(self) -> dict[str, int]:
        out = {}
        for a in self.agents:
            sit = Situation(a, self.scenario.profiles[a], self.stores[a], self.state.clock,
                            self._cands[a], self._preds[a])
            out[a] = self._expert.choose(sit).index
        return out

    def achieved(self, agent: str) -> bool:
        return all(goal_achieved(self.state, agent, g) for g in self.scenario.profiles[agent].goals)


def _load_predictor(path: str | None) -> OutcomePredictor:
    if not path:
        return HeuristicPredictor()
    from tianlong.learning.predictor import GNNPredictor

    return GNNPredictor.load(path)
