"""
[INPUT]: 依赖 ray.rllib 的 MultiAgentEnv，gymnasium，kernel 的 Kernel，cognition 的 BeliefStore / candidates，
         agents 的 HeuristicPredictor / ScriptedPolicy / Situation / OutcomePredictor，learning/task 的 TaskConfig，
         learning/rl 的 observation / rewards
[OUTPUT]: 对外提供 TianlongEnv（RLlib 多智能体环境：crops 暴露裁剪报告，tracker 暴露目标状态，episode_rewards 暴露分项奖励累计，
          coverage 暴露实际抽到的场景与目标分布）
[POS]: learning/rl 的训练环境：每个角色一个智能体，同一 tick 同时出招、由同一个内核统一结算——与线上完全相同的转移机制。
       世界从 TaskConfig 取样（江湖化比例、规模、启用目标族一路贯通到 reset），启用目标族之外的目标在 reset 时明确报错；
       观测只来自各自的认知图；奖励 = 目标跃迁的任务奖励 + 命名的塑形 + 分项成本（见 rewards）。
       expert_actions() 给出脚本示范，供模仿学习与评测；last_events 暴露上一步的真实事件，只供评测统计行为，不进观测
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

from __future__ import annotations

from collections import Counter

import gymnasium as gym
from ray.rllib.env.multi_agent_env import MultiAgentEnv

from tianlong.agents.policies import ScriptedPolicy, Situation
from tianlong.agents.predictors import HeuristicPredictor, OutcomePredictor, Prediction
from tianlong.cognition import BeliefStore, Candidate, candidates
from tianlong.core import Op, derive_seed, make_id
from tianlong.kernel import Kernel
from tianlong.learning.rl.observation import CropReport, ObsSpec, build_observation, observation_space
from tianlong.learning.rl.rewards import GoalTracker, RewardWeights, step_reward
from tianlong.learning.task import TaskConfig


class TianlongEnv(MultiAgentEnv):
    def __init__(self, config: dict | None = None) -> None:
        super().__init__()
        cfg = dict(config or {})
        task = TaskConfig.from_dict(cfg.get("task"))
        if "horizon" in cfg:                       # 兼容旧调用：horizon 直接给出
            task = TaskConfig.from_dict({**task.to_dict(), "horizon": int(cfg["horizon"])})
        self.task = task
        self.registry = task.registry()
        self.weights = RewardWeights(**cfg.get("reward", {}))
        self.obs_spec = ObsSpec(**cfg.get("spec", {}))
        self.horizon = task.horizon
        self.seed_base = int(cfg.get("seed", 0))
        self.predictor: OutcomePredictor = cfg.get("predictor") or _load_predictor(cfg.get("predictor_path"))
        self.zero_predictions = bool(cfg.get("zero_predictions", False))   # 训练期消融：策略从头到尾看不到世界模型预测
        self.possible_agents = [f"h{i}" for i in range(task.max_persons)]
        self.agents: list[str] = []
        obs_space = observation_space(self.obs_spec)
        self.observation_spaces = {a: obs_space for a in self.possible_agents}
        self.action_spaces = {a: gym.spaces.Discrete(self.obs_spec.max_cands) for a in self.possible_agents}
        self.kernel = Kernel()
        self._expert = ScriptedPolicy()
        self._episodes = 0
        self._cands: dict[str, tuple[Candidate, ...]] = {}
        self._preds: dict[str, tuple[Prediction, ...]] = {}
        self.crops: dict[str, CropReport] = {}    # 每个角色最近一次观测的裁剪报告：超预算必须看得见
        self.coverage: Counter = Counter()        # 实际抽到的场景与目标：配置是否真的到达了环境
        self.episode_rewards: dict[str, Counter] = {}
        self.last_events: tuple = ()

    # ------------------------------------------------------------
    #  gym 接口
    # ------------------------------------------------------------

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        world_seed = seed if seed is not None else derive_seed("env", self.seed_base, self._episodes) % 1_000_000_007
        self._episodes += 1
        sc = self.task.scenario(world_seed)
        self.scenario = sc
        self.state = sc.state
        self.tracker = GoalTracker(self.registry, sc.profiles)   # 启用目标族之外的目标：此刻报错
        self.tracker.start(self.state)
        self.stores = {a: BeliefStore(a, trust=dict(p.trust)).revise_all(sc.priors.get(a, ()))[0]
                       for a, p in sc.profiles.items()}
        self.agents = sorted(sc.profiles)
        self.t = 0
        self.episode_rewards = {a: Counter() for a in self.agents}
        self.coverage["episodes"] += 1
        self.coverage["jianghu_worlds"] += int(any(e.get("martial") is not None for e in self.state.entities.values()))
        for p in sc.profiles.values():
            for g in p.goals:
                self.coverage[f"goal:{g.kind.value}"] += 1
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
        self.last_events = result.events          # 评测用：统计行为（例如有没有学会动手抢）
        self.t += 1
        parts = step_reward(self.tracker, before, self.state, result.events, self.weights)
        rewards = {}
        for a in self.agents:
            rewards[a] = parts[a].total
            self.episode_rewards[a].update(parts[a].as_dict())
        done = self.t >= self.horizon
        obs = {a: self._observe(a) for a in self.agents}
        infos = {a: {"achieved": self.achieved(a), "reward": parts[a].as_dict()} for a in self.agents}
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
        ob = build_observation(store, self.state.clock, profile, cands, preds, self.obs_spec)
        if self.zero_predictions:
            ob.obs["cand_pred"][:] = 0.0
        # 动作编号对应裁剪后保留的候选：引用放不下的候选不会以悬空指针出现在策略面前
        self._cands[agent], self._preds[agent], self.crops[agent] = ob.candidates, ob.predictions, ob.report
        self.coverage["crop_events"] += int(ob.report.cropped)
        return ob.obs

    def situation(self, agent: str) -> Situation:
        return Situation(agent, self.scenario.profiles[agent], self.stores[agent], self.state.clock,
                         self._cands[agent], self._preds[agent])

    def expert_choices(self) -> dict:
        return {a: self._expert.choose(self.situation(a)) for a in self.agents}

    def expert_actions(self) -> dict[str, int]:
        return {a: c.index for a, c in self.expert_choices().items()}

    def achieved(self, agent: str) -> bool:
        return self.tracker.achieved(agent)


def _load_predictor(path: str | None) -> OutcomePredictor:
    if not path:
        return HeuristicPredictor()
    from tianlong.learning.predictor import GNNPredictor

    return GNNPredictor.load(path)
