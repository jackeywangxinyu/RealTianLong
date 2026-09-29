"""
[INPUT]: 依赖 core 的 Observation / Fact / Modality / Op / Rel / make_id / MemoryRecord，cognition 的 BeliefChange，language/templates 的渲染函数
[OUTPUT]: 对外提供 records_for()：从一条观察及其引起的信念变化中提炼“值得记住的经历”；claim_verdicts()：传闻被亲眼证实/证伪的判定
[POS]: memory 的记忆写入策略；被权威写入器与 RL 训练环境在同一处调用（训练与上线的长期记忆是同一个定义）。
       只记事件、意外（原以为在的东西不见了）与“谁的说法被亲眼证实/证伪”，不记每分钟一次的“一切如常”；
       证伪只认亲眼所见——被另一个更可信的传闻盖过不算谁撒了谎
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

from __future__ import annotations

from collections.abc import Sequence

from tianlong.cognition import FIRSTHAND, BeliefChange
from tianlong.core import Fact, Modality, Observation, Op, Rel, make_id
from tianlong.core.memories import MemoryRecord
from tianlong.language.templates import Names, render_fact, render_percept

_AT = Rel.AT.value


def claim_verdicts(modality: Modality, changes: Sequence[BeliefChange]) -> list[tuple[str, str, BeliefChange]]:
    """(说话者, "confirmed"/"refuted", 变化)：只有亲眼所见才能证实或证伪一条传闻。"""
    if modality not in FIRSTHAND:
        return []
    out = []
    for c in changes:
        b, a = c.before, c.after
        if b is None or b.informant is None or not b.hearsay or not b.holds:
            continue
        if a is not None and not a.hearsay and a.holds == b.holds and a.prop == b.prop:
            out.append((b.informant, "confirmed", c))
        elif a is None or not a.holds:
            out.append((b.informant, "refuted", c))
    return out


def records_for(
    world_id: str, branch_id: str, obs: Observation, changes: Sequence[BeliefChange], names: Names
) -> list[MemoryRecord]:
    p = obs.percept
    me = obs.observer
    out: list[MemoryRecord] = []

    def emit(kind: str, text: str, subjects: tuple[str, ...], informant: str | None = None,
             verdict: str | None = None) -> None:
        rid = make_id("mem", world_id, branch_id, obs.id, len(out))  # 跨世界/分支全局唯一：向量点 ID 由它派生
        out.append(MemoryRecord(rid, world_id, branch_id, me, kind, text, p.tick, p.tick, obs.id, subjects,
                                informant, verdict))

    # ---- 事件：看到的、听到的、自己做的、别人说的 ----
    if p.event is not None and not (p.modality == Modality.SELF and p.event.kind == Op.WAIT.value):
        kind = "speech" if p.modality == Modality.SPEECH else "event"
        ids = tuple(i for i in (p.event.actor, p.event.target, p.event.obj, p.event.place) if i)
        emit(kind, render_percept(p, names, me), ids)

    # ---- 意外：某物不在原以为的位置了（以及随之发现的新位置）----
    lost: dict[str, BeliefChange] = {}
    for c in changes:
        b = c.before
        refuted = b is not None and b.holds and (c.after is None or not c.after.holds)
        if refuted and b.prop.predicate == _AT and b.prop.subject != me:
            lost[b.prop.subject] = c
    for subject, c in sorted(lost.items()):
        assert c.before is not None
        emit("discovery", "发现" + render_fact(Fact(c.before.prop, False), names, me), (subject,))
    for c in changes:
        a = c.after
        if a is not None and a.holds and a.prop.predicate == _AT and a.prop.subject in lost:
            emit("discovery", "发现" + render_fact(Fact(a.prop, True), names, me), (a.prop.subject,))

    # ---- 说法的验证：谁的话被亲眼证实、谁的话落了空 ----
    for who, verdict, c in claim_verdicts(p.modality, changes):
        assert c.before is not None
        claim = render_fact(Fact(c.before.prop, True), names, me)
        speaker = names[who].name if who in names else who
        text = f"{speaker}说过{claim}——{'果然不假' if verdict == 'confirmed' else '其实不然'}"
        emit("verdict", text, (who, c.before.prop.subject), who, verdict)
    return out
