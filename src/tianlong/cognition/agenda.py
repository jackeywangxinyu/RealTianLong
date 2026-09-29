"""
[INPUT]: 依赖 core 的 Fact / Modality / Op / Outcome / Percept
[OUTPUT]: 对外提供 Obligation（欠着别人的：被问到的问题）、Said（对谁说过什么）、fold_agenda()、MAX_OBLIGATIONS / MAX_SAID
[POS]: cognition 的持久任务状态：短期经历缓冲（episodes，容量 12）会被环顾、响动挤掉，“有人问过我”“我已经告诉过他”不能跟着消失。
       这里把它们从感知折叠成独立的、有界的记录：被人问到 → 记一笔待答；自己把答案说给了他 → 这一笔勾销，并记下“说过”。
       与信念一样只来自感知，不读真相；容量有界且溢出时丢最旧的一条（写明，不静默增长）
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

from __future__ import annotations

from dataclasses import dataclass

from tianlong.core import Fact, Modality, Op, Outcome, Percept

MAX_OBLIGATIONS = 16
MAX_SAID = 64


@dataclass(frozen=True, slots=True)
class Obligation:
    kind: str           # "answer"：有人问了我一个问题
    counterpart: str    # 问话的人
    topic: Fact         # 问的命题（宾语为 None 的提问）
    since: int


@dataclass(frozen=True, slots=True)
class Said:
    listener: str
    fact: Fact
    tick: int


def _answers(told: Fact, asked: Fact) -> bool:
    return told.prop.subject == asked.prop.subject and told.prop.predicate == asked.prop.predicate


def fold_agenda(owner: str, obligations: tuple[Obligation, ...], said: tuple[Said, ...], p: Percept
                ) -> tuple[tuple[Obligation, ...], tuple[Said, ...]]:
    ev = p.event
    if ev is None or ev.topic is None:
        return obligations, said
    if p.modality == Modality.SPEECH and ev.kind == Op.ASK.value and ev.target == owner and ev.actor:
        ob = Obligation("answer", ev.actor, ev.topic, p.tick)
        if not any(o.counterpart == ob.counterpart and o.topic == ob.topic for o in obligations):
            obligations = (*obligations, ob)[-MAX_OBLIGATIONS:]
    elif p.modality == Modality.SELF and ev.kind == Op.TELL.value and ev.outcome == Outcome.SUCCESS and ev.target:
        said = (*(s for s in said if not (s.listener == ev.target and s.fact == ev.topic)),
                Said(ev.target, ev.topic, p.tick))[-MAX_SAID:]
        obligations = tuple(o for o in obligations
                            if not (o.counterpart == ev.target and _answers(ev.topic, o.topic)))
    return obligations, said
