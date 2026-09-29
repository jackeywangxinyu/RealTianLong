"""
[INPUT]: 依赖 scenarios/procedural 的 random_scenario，core/goals 的 GoalRegistry / DEFAULT_GOALS / GOALS_VERSION，core/profiles 的 GoalKind，
         core 的 digest
[OUTPUT]: 对外提供 TaskConfig（训练/评测/部署共用的任务分布定义）、TASK_VERSION、
          arg_type()（配置字段展平成 CLI 参数时的解析器：布尔严格按 true/false 解析）
[POS]: learning 的任务契约：场景混合（江湖化比例）、地图规模、人数、启用的目标族、修习机制覆盖旋钮、时限——
       GNN 数据、模仿学习示范、PPO 环境、评测与模型 manifest 读的是同一个解析后的 TaskConfig，
       “训练时的江湖参数没传到 PPO”这种断层在结构上不再可能。启用的目标族同时约束取样（只生成这些目标）与奖励
       （GoalTracker 只注册这些）；显式注入的未注册目标在环境初始化时明确报错（UnsupportedGoal），不会静默落进别的目标的奖励
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, fields

from tianlong.core import digest
from tianlong.core.goals import DEFAULT_GOALS, GOALS_VERSION, GoalRegistry
from tianlong.core.profiles import GoalKind
from tianlong.scenarios import Scenario
from tianlong.scenarios.procedural import random_scenario

TASK_VERSION = "task-v1"
ALL_GOALS = tuple(k.value for k in GoalKind)


@dataclass(frozen=True)
class TaskConfig:
    jianghu: float = 0.5          # 江湖化世界比例（身手、兵刃与毒、解药、秘籍、单向通道、寻仇与护人）
    max_places: int = 5
    max_items: int = 4
    max_persons: int = 3
    scroll_rate: float = 0.5      # 江湖世界里有秘籍的概率
    scroll_held: float = 0.0      # 秘籍一开始就在某人手上的概率（提高修习机制的数据覆盖）
    hide_goal_items: float = 0.0  # “先探查、再决策”任务：获取/递送目标的物品被藏起来的概率（预测器有无的对照用）
    roles: float = 0.3            # 江湖世界里出现“守地 + 约时潜逃”角色的概率：七类目标与时间闸门都进入训练分布
    goals: tuple[str, ...] = ALL_GOALS   # 启用的目标族：奖励与评测只认这些
    horizon: int = 30             # 一局的 tick 数

    def __post_init__(self) -> None:
        unknown = set(self.goals) - set(ALL_GOALS)
        if unknown:
            raise ValueError(f"未知目标族 {sorted(unknown)}")
        if any(not 0.0 <= v <= 1.0
               for v in (self.jianghu, self.scroll_rate, self.scroll_held, self.hide_goal_items, self.roles)):
            raise ValueError("概率旋钮必须在 [0, 1]")
        if not (2 <= self.max_persons <= 6 and 3 <= self.max_places <= 8 and 2 <= self.max_items <= 8
                and self.horizon >= 1):
            raise ValueError("规模超出程序化世界的范围：2≤人数≤6、3≤地点≤8、2≤物品≤8、时限≥1")

    def scenario(self, seed: int) -> Scenario:
        subset = None if set(self.goals) == set(ALL_GOALS) else tuple(self.goals)
        return random_scenario(seed, self.max_places, self.max_items, self.max_persons, self.jianghu,
                               self.scroll_rate, self.scroll_held, self.hide_goal_items, self.roles, subset)

    def registry(self) -> GoalRegistry:
        enabled = set(self.goals)
        return GoalRegistry(e for e in DEFAULT_GOALS if e.kind.value in enabled)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["goals"] = list(self.goals)
        return d

    @classmethod
    def from_dict(cls, d: dict | None) -> TaskConfig:
        if not d:
            return cls()
        names = {f.name for f in fields(cls)}
        kw = {k: (tuple(v) if k == "goals" else v) for k, v in d.items() if k in names}
        return cls(**kw)

    def fingerprint(self) -> str:
        return digest(TASK_VERSION, GOALS_VERSION, repr(sorted(self.to_dict().items())))


def _strict_bool(v: str) -> bool:
    low = str(v).strip().lower()
    if low in ("1", "true", "yes"):
        return True
    if low in ("0", "false", "no"):
        return False
    raise argparse.ArgumentTypeError(f"布尔参数只认 true/false/1/0/yes/no，收到 {v!r}")


def arg_type(default):
    """配置字段的 CLI 解析器：argparse 的 type=bool 会把字符串 "false" 当成真——布尔字段必须经这里。"""
    return _strict_bool if isinstance(default, bool) else type(default)
