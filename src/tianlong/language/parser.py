"""
[INPUT]: 依赖 cognition 的 BeliefStore / Candidate，core 的 Op / Manner / Kind / Rel / Fact / Proposition / signature_error，
         language/command 的 ACTION_WORDS / analyze / clarify / ParsedCommand / Mention，language/llm 的 LLMClient / LLMUnavailable / parse_json
[OUTPUT]: 对外提供 Parsed（含语态结构与等待时长）、IntentParser（语态闸门 → 规则快路径 → 受约束的 LLM 语义解析）、rule_parse()、normalize()
[POS]: language 的输入解析；把玩家自由文本变成结构化候选行动。先由 command.analyze() 判定语态：只有单一、肯定、即时的指令
       才走规则快路径；否定、条件、转述、复合、疑问交给 LLM（它也必须声明语态与主体），仍不确定就追问、不推进时间。
       LLM 失败时绝不回退到未经语义确认的候选。可引用的实体只来自玩家自己的认知图；解析结果仍要回到 kernel 结算
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass, replace

from tianlong.cognition import BeliefStore, Candidate
from tianlong.cognition.navigation import routes_between
from tianlong.core import Fact, Kind, Manner, Op, Proposition, Rel
from tianlong.core.grammar import signature_error
from tianlong.language.command import ACTION_WORDS, Mention, ParsedCommand, SpeechMode, analyze, clarify
from tianlong.language.llm import LLMClient, LLMUnavailable, parse_json

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class Parsed:
    candidate: Candidate | None
    utterance: str | None = None
    clarification: str | None = None   # 解析不了时给玩家的追问
    source: str = "rules"
    repeat: int = 1                    # 等待的分钟数（“等一炷香”= 30）
    until: str | None = None           # 等到某个时刻（"night"）：会话层按时钟换算
    command: ParsedCommand | None = None   # 语态结构：否定/条件/转述等非即时语态不会产生候选


# ============================================================
#  规则解析：语态闸门之后，关键词定操作，实体提及定角色
#  言语类最先判定（“告诉守卫我去港口”里的“去”不是移动）
# ============================================================

# 等待时长（分钟）；“等到天黑”交给会话层按时钟换算
_DURATIONS: tuple[tuple[str, int], ...] = (
    ("一个时辰", 120), ("半个时辰", 60), ("一炷香", 30), ("一盏茶", 15), ("一会", 10), ("片刻", 5),
)
_UNTIL_NIGHT = ("天黑", "入夜", "晚上", "夜里", "月亮")
_CAREFUL = ("悄悄", "小心", "轻轻", "偷偷", "藏")
_ROUGH = ("用力", "粗暴", "猛", "狠狠")
_ASKS = {
    Op.TAKE: "你想拿什么？", Op.PUT: "你想把什么放到哪里？", Op.GIVE: "你想把什么交给谁？",
    Op.UNLOCK: "你想用什么打开哪扇门？", Op.LOCK: "你想用什么锁上哪扇门？", Op.MOVE: "你想去哪里？你知道怎么走过去吗？",
    Op.TELL: "你想告诉谁什么？", Op.ASK: "你想问谁什么？", Op.INSPECT: "你想查看什么？",
    Op.ATTACK: "你想对谁出手？", Op.STUDY: "你想研读什么？", Op.USE: "你想把什么用在谁身上？",
}


def _aliases(name: str, kind: Kind) -> tuple[str, ...]:
    out = [name]
    if kind == Kind.SURFACE and len(name) >= 2 and name[-1] in "面子":
        out.append(name[:-1])  # 桌面 → 桌
    return tuple(out)


def _mentions(text: str, store: BeliefStore, extra: Mapping[str, tuple[str, ...]] | None = None) -> list[Mention]:
    """按出现位置排序的实体提及；同一位置取最长名字。“我/自己”指玩家自己。别称只对玩家认识的实体生效。"""
    found: dict[int, Mention] = {}
    for eid, sk in store.entities.items():
        for alias in (*_aliases(sk.name, sk.kind), *((extra or {}).get(eid, ()))):
            start = text.find(alias)
            while start != -1:
                prev = found.get(start)
                if prev is None or len(alias) > prev.length:
                    found[start] = Mention(start, len(alias), eid, sk.kind)
                start = text.find(alias, start + 1)
    for word in ("自己", "我"):
        if word in text:
            pos = text.find(word)
            found.setdefault(pos, Mention(pos, len(word), store.owner, Kind.PERSON))
    return [found[k] for k in sorted(found)]


def rule_parse(text: str, store: BeliefStore, aliases: Mapping[str, tuple[str, ...]] | None = None) -> Parsed:
    """语态闸门：非即时语态不产生候选。即时指令按优先级尝试每个命中关键词的操作，返回第一个角色齐全的解析
    （“揣进兜里”的“进”不该赢过“揣”）。"""
    t = text.strip().lower()
    ms = _mentions(t, store, aliases)
    command = analyze(text, ms, store.owner)
    ops = [o for o, words in ACTION_WORDS if any(w in t for w in words)]
    if not ops:
        return Parsed(None, clarification="没听懂。试试：去后院 / 查看玉璧 / 问马五爷… / 出手 / 研读… / 等到天黑",
                      command=command)
    if not command.immediate:
        return Parsed(None, clarification=clarify(command), command=command)
    attempts = [_parse_as(op, text, t, store, ms) for op in ops]
    chosen = next((p for p in attempts if p.candidate is not None), attempts[0])
    return replace(chosen, command=command)


def _wait_length(t: str) -> tuple[int, str | None]:
    if any(w in t for w in _UNTIL_NIGHT):
        return 1, "night"
    digits = "".join(ch for ch in t if ch.isdigit())
    if digits and "分" in t:
        return max(1, int(digits)), None
    return next((m for word, m in _DURATIONS if word in t), 1), None


def _parse_as(op: Op, text: str, t: str, store: BeliefStore, ms: list[Mention]) -> Parsed:
    manner = Manner.CAREFUL if any(w in t for w in _CAREFUL) else (
        Manner.ROUGH if any(w in t for w in _ROUGH) else Manner.NORMAL)
    me = store.owner
    here = store.location_of(me)

    def first(*kinds: Kind, after: int = -1, exclude: tuple[str, ...] = ()) -> tuple[int, str] | None:
        for m in ms:
            if m.pos > after and m.kind in kinds and m.eid not in exclude:
                return m.pos, m.eid
        return None

    def pick(*kinds: Kind, **kw) -> str | None:
        hit = first(*kinds, **kw)
        return hit[1] if hit else None

    held = [i for i, sk in sorted(store.entities.items()) if sk.kind == Kind.ITEM and store.location_of(i) == me]
    target = obj = None
    topic: Fact | None = None
    if op == Op.TAKE:
        target = pick(Kind.ITEM)
    elif op == Op.PUT:
        obj = pick(Kind.ITEM) or (held[0] if len(held) == 1 else None)
        target = pick(Kind.SURFACE, Kind.PLACE) or here
    elif op == Op.GIVE:
        target, obj = pick(Kind.PERSON, exclude=(me,)), pick(Kind.ITEM)
    elif op in (Op.UNLOCK, Op.LOCK):
        target = pick(Kind.DOOR)
        obj = pick(Kind.ITEM) or (held[0] if len(held) == 1 else None)
    elif op == Op.INSPECT:
        target = pick(Kind.PLACE, Kind.SURFACE, Kind.PERSON, exclude=(me,)) or here
    elif op == Op.MOVE:
        target = pick(Kind.PLACE, Kind.DOOR)
        obj = pick(Kind.DOOR) if target is not None and store.sketch(target).kind == Kind.PLACE else None
    elif op == Op.ATTACK:
        target = pick(Kind.PERSON, exclude=(me,))
    elif op == Op.STUDY:
        target = pick(Kind.ITEM) or (held[0] if len(held) == 1 else None)
    elif op == Op.USE:
        obj = pick(Kind.ITEM) or (held[0] if len(held) == 1 else None)
        target = pick(Kind.PERSON, exclude=(me,)) or me
    elif op in (Op.TELL, Op.ASK):
        listener = first(Kind.PERSON, exclude=(me,))
        if listener is not None:
            target = listener[1]
            subject = first(Kind.ITEM, Kind.PERSON, after=listener[0], exclude=(target,))
            if subject is not None:
                if op == Op.ASK:
                    topic = Fact(Proposition.rel(subject[1], Rel.AT, None), True)
                else:
                    value = pick(Kind.PLACE, Kind.SURFACE, Kind.PERSON, after=subject[0])
                    if value is not None:
                        topic = Fact(Proposition.rel(subject[1], Rel.AT, value), "不在" not in t)
    cand = normalize(Candidate(op, target, obj, manner, topic), store)
    if _invalid(cand, store):
        return Parsed(None, clarification=_ASKS.get(op, "请说得具体一些。"))
    utterance = text.strip() if op in (Op.TELL, Op.ASK) else None
    if op == Op.WAIT:
        repeat, until = _wait_length(t)
        return Parsed(cand, source="rules", repeat=repeat, until=until)
    return Parsed(cand, utterance, source="rules")


def normalize(c: Candidate, store: BeliefStore) -> Candidate:
    """MOVE 绑定一条玩家自己知道的路：“朝那扇门走” = 经这扇门去门那边的地点；“去某地” = 经玩家认为连通的门
    （认为没锁的优先）。玩家不知道怎么去，路线就留空——语法检查会追问，内核不会替他从真实地图里挑一条暗道。"""
    if c.op != Op.MOVE:
        return c
    here = store.location_of(store.owner)
    sk = store.sketch(c.target or "")
    if sk is not None and sk.kind == Kind.DOOR:
        others = [b.prop.value for b in store.positives(sk.id, Rel.CONNECTS.value) if b.prop.value != here]
        if len(others) == 1:
            return Candidate(Op.MOVE, str(others[0]), sk.id, c.manner, None)
        return c
    if c.target is not None and c.obj is None and here is not None:
        routes = routes_between(store, here, c.target)
        if routes:
            return Candidate(Op.MOVE, c.target, routes[0], c.manner, None)
    return c


def _invalid(c: Candidate, store: BeliefStore) -> str | None:
    def kind_of(eid: str) -> Kind | None:
        sk = store.sketch(eid)
        return sk.kind if sk else None

    return signature_error(c.op, kind_of, c.target, c.obj, c.topic)


# ============================================================
#  LLM 解析：只给它玩家认识的实体表，输出受 JSON Schema 约束，再用同一把语法尺子校验
# ============================================================

_SYSTEM = (
    "你是文字游戏的指令解析器。把玩家的中文输入解析成一个结构化行动。"
    "先判断语态 mode：immediate=玩家此刻要亲自做的一件事；negated=否定（不做）；conditional=带条件、计划或斟酌；"
    "narrative=叙述别人的举动或已经发生的事；quoted=行动只出现在引语里；compound=多件事；question=询问能否。"
    "actor=行动主体：player=玩家本人，other=别人，unknown=说不清。只有 immediate 且 actor=player 才会被执行。"
    "只能引用实体表里的 id；做不到或意图不明就把 op 设为 unknown 并给出 clarification。"
    "target=行动直接作用的对象（拿的物品、去的地点、开的门、说话的对象、查看的东西）；"
    "obj=工具或被递交/放置的物品（开锁的钥匙、放下或交出的东西）；move 的 obj 是走的那扇门（不确定就填 null）；"
    "不适用的字段填 null。"
    "manner: careful=小心/悄悄/藏，rough=粗暴/用力，否则 normal。"
    "tell/ask 的语义内容用 topic_subject/topic_value 表示“subject 在 value”（ask 时 topic_value 留空）。"
)


def _schema(ids: list[str]) -> dict:
    ref = {"type": "string", "enum": ids, "nullable": True}
    return {
        "type": "object",
        "properties": {
            "mode": {"type": "string", "enum": [m.value for m in SpeechMode]},
            "actor": {"type": "string", "enum": ["player", "other", "unknown"]},
            "op": {"type": "string", "enum": [o.value for o in Op] + ["unknown"]},
            "target": ref, "obj": ref, "topic_subject": ref, "topic_value": ref,
            "topic_holds": {"type": "boolean"},
            "manner": {"type": "string", "enum": [m.value for m in Manner]},
            "clarification": {"type": "string"},
        },
        # 角色字段全部 required（允许 null）：否则模型会干脆省略 target
        "required": ["mode", "actor", "op", "target", "obj", "manner", "topic_subject", "topic_value", "topic_holds",
                     "clarification"],
    }


def _table(store: BeliefStore) -> str:
    rows = []
    for eid, sk in sorted(store.entities.items()):
        loc = store.location_of(eid)
        where = f"，你认为在 {loc}" if loc else ""
        rows.append(f"- {eid}：{sk.name}（{sk.kind.value}{where}）")
    return "\n".join(rows)


class IntentParser:
    def __init__(self, llm: LLMClient | None = None, prefer_llm: bool = False,
                 aliases: Mapping[str, tuple[str, ...]] | None = None) -> None:
        self.llm = llm
        self.prefer_llm = prefer_llm
        self.aliases = dict(aliases or {})

    def parse(self, text: str, store: BeliefStore) -> Parsed:
        ruled = rule_parse(text, store, self.aliases)
        command = ruled.command
        immediate = command is not None and command.immediate
        if self.llm is None or (ruled.candidate is not None and not self.prefer_llm):
            return ruled
        try:
            got = self._llm_parse(text, store, command)
        except LLMUnavailable as e:
            log.warning("LLM 解析不可用: %s", e)
            got = None
        if got is not None:
            return got
        # 回退只允许经语态闸门确认过的即时解析；否定/条件/转述绝不因模型失败而被执行
        return ruled if immediate else Parsed(None, clarification=ruled.clarification, source="rules",
                                              command=command)

    def _llm_parse(self, text: str, store: BeliefStore, command: ParsedCommand | None) -> Parsed | None:
        assert self.llm is not None
        ids = sorted(store.entities)
        prompt = f"你是 {store.owner}。你认识的实体：\n{_table(store)}\n\n玩家输入：{text}"
        data = parse_json(self.llm.generate(prompt, system=_SYSTEM, schema=_schema(ids), temperature=0.0))
        mode = data.get("mode", SpeechMode.UNCLEAR.value)
        if data.get("op") in (None, "unknown") or mode != SpeechMode.IMMEDIATE.value or data.get("actor") != "player":
            fallback = clarify(command) if command is not None and not command.immediate else "请说得具体一些。"
            return Parsed(None, clarification=data.get("clarification") or fallback, source="llm", command=command)
        known = set(ids)

        def ref(key: str) -> str | None:
            v = data.get(key) or None
            return v if v in known else None

        topic = None
        subject = ref("topic_subject")
        if subject:
            topic = Fact(Proposition.rel(subject, Rel.AT, ref("topic_value")), bool(data.get("topic_holds", True)))
        cand = normalize(Candidate(Op(data["op"]), ref("target"), ref("obj"), Manner(data.get("manner", "normal")), topic),
                         store)
        if command is not None and cand.op in command.negated:
            # 规则层看见了对这个行动的否定：模型说“照做”也不行
            return Parsed(None, clarification=clarify(command), source="llm", command=command)
        if _invalid(cand, store):
            return None
        utterance = text.strip() if cand.op in (Op.TELL, Op.ASK) else None
        return Parsed(cand, utterance, source="llm", command=command)
