"""
[INPUT]: 依赖 core 的 Percept / Modality / Op / Outcome，language/templates 的 render_percept / Names，language/llm 的 LLMClient
[OUTPUT]: 对外提供 Narrator（把玩家本回合的感知写成叙述）
[POS]: language 的输出层；输入只有玩家自己的感知（不是世界真相），模板先把它们写成事实清单，LLM 只负责润色，
       被要求不得添加清单外的任何人物、物品、事件或结论；模型不可用时直接输出清单
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

from tianlong.core import Modality, Op, Outcome, Percept
from tianlong.language.llm import LLMClient, LLMUnavailable
from tianlong.language.templates import Names, render_percept

log = logging.getLogger(__name__)

_SYSTEM = (
    "你是一部中文文字冒险游戏的叙述者，用第二人称“你”写 2~4 句简洁的叙述。"
    "只能使用给定清单里的事实；不得添加清单之外的人物、物品、事件、动机或推断；可以适度描写氛围与动作细节。"
)


def fact_lines(viewer: str, percepts: Sequence[Percept], names: Names, show_scene: bool = False) -> list[str]:
    """本回合值得讲的事：事件感知全部讲；环顾只在移动/查看之后（或被要求时）讲。"""
    moved = any(
        p.modality == Modality.SELF and p.event and p.event.kind in (Op.MOVE.value, Op.INSPECT.value)
        and p.event.outcome == Outcome.SUCCESS
        for p in percepts
    )
    lines = []
    for p in percepts:
        if p.modality == Modality.SCENE:
            if show_scene or moved:
                lines.append("你看到：" + render_percept(p, names, viewer, me="你"))
        else:
            lines.append(render_percept(p, names, viewer, me="你"))
    return lines


class Narrator:
    def __init__(self, llm: LLMClient | None = None) -> None:
        self.llm = llm

    def narrate(self, viewer: str, percepts: Sequence[Percept], names: Names, show_scene: bool = False) -> str:
        lines = fact_lines(viewer, percepts, names, show_scene)
        if not lines:
            return "时间悄悄过去，什么也没有发生。"
        plain = "\n".join(lines)
        if self.llm is None:
            return plain
        try:
            prose = self.llm.generate("本回合玩家感知到的事实：\n" + plain, system=_SYSTEM, temperature=0.6)
            return prose.strip() or plain
        except LLMUnavailable as e:
            log.warning("叙述润色失败，回退事实清单: %s", e)
            return plain
