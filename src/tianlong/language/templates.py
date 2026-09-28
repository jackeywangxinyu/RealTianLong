"""
[INPUT]: 依赖 core 的 Fact / PerceivedEvent / Percept / Modality / Op / Outcome / Rel
[OUTPUT]: 对外提供 Names 类型、render_fact()、render_event()、render_percept()、REASONS
[POS]: language 的确定性文本层（无 LLM）；memory 用它生成经历文本，narrator 在无模型时用它兜底——同一套措辞，两处复用
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

from __future__ import annotations

from collections.abc import Mapping

from tianlong.core import Fact, Modality, Op, Outcome, PerceivedEvent, Percept, Rel
from tianlong.core.schema import ATTR_PREFIX

Names = Mapping[str, str]

REASONS: dict[str, str] = {
    "door_locked": "门锁着",
    "not_found": "那里没有找到",
    "held_by_other": "东西在别人手里",
    "wrong_key": "钥匙对不上",
    "out_of_reach": "够不着",
    "not_holding": "手里并没有那样东西",
    "already_there": "已经在那里了",
    "already_held": "已经拿在手里了",
    "already_locked": "本来就锁着",
    "already_unlocked": "本来就没锁",
    "not_adjacent": "从这里过不去",
    "self_target": "不能对自己这么做",
    "stale": "时机已经过去了",
    "duplicate_actor": "同一时刻只能做一件事",
}


def _n(names: Names, eid: str | None, viewer: str | None = None) -> str:
    if eid is None:
        return "某处"
    if eid == viewer:
        return "我"
    return names.get(eid, eid)


# ============================================================
#  命题
# ============================================================


def render_fact(f: Fact, names: Names, viewer: str | None = None) -> str:
    p = f.prop
    subj = _n(names, p.subject, viewer)
    neg = not f.holds
    if p.predicate.startswith(ATTR_PREFIX):
        key = p.attr_key
        if key == "locked":
            return f"{subj}{'没锁' if neg else '锁着'}"
        if key == "hidden":
            return f"{subj}{'没有被藏起来' if neg else '被藏了起来'}"
        return f"{subj}的{key}{'不是' if neg else '是'}{p.value}"
    rel = Rel(p.predicate)
    if p.value is None:
        return f"{subj}在哪里？" if rel == Rel.AT else f"{subj}的{rel.value}是什么？"
    obj = _n(names, str(p.value), viewer)
    phrase = {
        Rel.AT: ("在", "不在"),
        Rel.OWNS: ("拥有", "并不拥有"),
        Rel.MATCHES: ("能打开", "打不开"),
        Rel.CONNECTS: ("通往", "不通往"),
    }[rel]
    return f"{subj}{phrase[1] if neg else phrase[0]}{obj}"


# ============================================================
#  事件
# ============================================================


def _verb(v: PerceivedEvent, names: Names, viewer: str | None) -> str:
    t, o = _n(names, v.target, viewer), _n(names, v.obj, viewer)
    topic = render_fact(v.topic, names, viewer) if v.topic else "一些话"
    op = Op(v.kind)
    table = {
        Op.MOVE: f"走向{t}",
        Op.TAKE: f"拿起{t}",
        Op.PUT: f"把{o}放在{t}",
        Op.GIVE: f"把{o}交给{t}",
        Op.UNLOCK: f"用{o}开{t}的锁",
        Op.LOCK: f"用{o}锁上{t}",
        Op.INSPECT: f"仔细查看{t}",
        Op.TELL: f"对{t}说：“{topic}”" if v.topic else f"对{t}低声说了些什么",
        Op.ASK: f"问{t}：“{topic}”" if v.topic else f"向{t}低声问了些什么",
        Op.WAIT: "静静等待",
    }
    return table[op]


def render_event(v: PerceivedEvent, names: Names, viewer: str | None = None) -> str:
    if v.kind == "noise":
        return f"{_n(names, v.place, viewer)}那边传来一阵响动"
    text = f"{_n(names, v.actor, viewer)}{_verb(v, names, viewer)}"
    if v.outcome == Outcome.FAILURE:
        text += f"，但没有成功（{REASONS.get(v.reason, v.reason)}）" if v.reason else "，但没有成功"
    elif v.outcome == Outcome.REJECTED:
        text = f"{_n(names, v.actor, viewer)}想要{_verb(v, names, viewer)}，但这行不通"
    return text


def render_percept(p: Percept, names: Names, viewer: str) -> str:
    """角色视角的一句话。SCENE 渲染为所见清单。"""
    if p.modality == Modality.SCENE:
        seen = [render_fact(f, names, viewer) for f in p.facts if f.holds and f.prop.predicate == Rel.AT.value
                and f.prop.subject != viewer]
        return "；".join(seen) if seen else "四下空无一物"
    if p.event is None:
        return "；".join(render_fact(f, names, viewer) for f in p.facts)
    prefix = {Modality.SOUND: "听到", Modality.SIGHT: "看见", Modality.SPEECH: "听见"}.get(p.modality, "")
    return prefix + render_event(p.event, names, viewer)

