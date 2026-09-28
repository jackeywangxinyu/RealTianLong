"""
[INPUT]: 依赖 cognition 的 BeliefStore，core 的 Intent / Op
[OUTPUT]: 对外提供 Scheduler（谁在本 tick 需要完整决策）
[POS]: agents 的节流阀；调度依据是游戏时间与事件：有新经历、手头有事、闲置太久才完整决策，其余人执行低成本例行动作（原地等待）。
       不在玩家眼前的角色也照常推进，但不必每分钟都跑一遍完整流程
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

from __future__ import annotations

from dataclasses import dataclass

from tianlong.cognition import BeliefStore
from tianlong.core import Intent, Op


@dataclass
class _Mark:
    tick: int
    busy: bool


class Scheduler:
    def __init__(self, idle_interval: int = 15) -> None:
        self.idle_interval = idle_interval
        self._marks: dict[str, _Mark] = {}

    def due(self, agent: str, store: BeliefStore, now: int) -> bool:
        mark = self._marks.get(agent)
        if mark is None or mark.busy or now - mark.tick >= self.idle_interval:
            return True
        return any(ep.tick >= mark.tick for ep in store.episodes)  # 上次决策之后有了新经历

    def record(self, agent: str, now: int, intent: Intent) -> None:
        self._marks[agent] = _Mark(now, busy=intent.op != Op.WAIT)
