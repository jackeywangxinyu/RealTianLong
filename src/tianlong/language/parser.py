"""
[INPUT]: 依赖 cognition 的 BeliefStore / Candidate，core 的 Op / Manner / Kind / Rel / Fact / Proposition / signature_error，
         language/llm 的 LLMClient / LLMUnavailable / parse_json
[OUTPUT]: 对外提供 Parsed、IntentParser（规则优先、LLM 兜底）、rule_parse()、normalize()
[POS]: language 的输入解析；把玩家自由文本变成结构化候选行动。可引用的实体只来自玩家自己的认知图——
       LLM 看不到、也无法指向玩家不认识的东西；解析结果仍要回到 kernel 结算，失败本身也是游戏内容
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from tianlong.cognition import BeliefStore, Candidate
from tianlong.core import Fact, Kind, Manner, Op, Proposition, Rel
from tianlong.core.grammar import signature_error
from tianlong.language.llm import LLMClient, LLMUnavailable, parse_json

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class Parsed:
    candidate: Candidate | None
    utterance: str | None = None
    clarification: str | None = None   # 解析不了时给玩家的追问
    source: str = "rules"


# ============================================================
#  规则解析：关键词定操作，实体提及定角色
#  言语类最先判定（“告诉守卫我去港口”里的“去”不是移动）
# ============================================================

_OPS: list[tuple[Op, tuple[str, ...]]] = [
    (Op.ASK, ("问", "打听", "ask")),
    (Op.TELL, ("告诉", "说", "tell")),
    (Op.UNLOCK, ("开锁", "解锁", "打开", "unlock")),
    (Op.LOCK, ("锁上", "上锁", "lock")),
    (Op.GIVE, ("交给", "递给", "给", "give")),
    (Op.PUT, ("放", "藏", "put", "hide")),
    (Op.TAKE, ("拿", "取", "捡", "偷", "揣", "拾", "抓", "take", "grab")),
    (Op.INSPECT, ("查看", "检查", "搜", "看看", "观察", "找找", "inspect", "search", "look")),
    (Op.MOVE, ("去", "走", "前往", "进", "回", "到", "go", "move")),
    (Op.WAIT, ("等", "休息", "wait")),
]
_CAREFUL = ("悄悄", "小心", "轻轻", "偷偷", "藏")
_ROUGH = ("用力", "粗暴", "猛", "狠狠")
_ASKS = {
    Op.TAKE: "你想拿什么？", Op.PUT: "你想把什么放到哪里？", Op.GIVE: "你想把什么交给谁？",
    Op.UNLOCK: "你想用什么打开哪扇门？", Op.LOCK: "你想用什么锁上哪扇门？", Op.MOVE: "你想去哪里？",
    Op.TELL: "你想告诉谁什么？", Op.ASK: "你想问谁什么？", Op.INSPECT: "你想查看什么？",
}


def _aliases(name: str, kind: Kind) -> tuple[str, ...]:
    out = [name]
    if kind == Kind.SURFACE and len(name) >= 2 and name[-1] in "面子":
        out.append(name[:-1])  # 桌面 → 桌
    return tuple(out)


def _mentions(text: str, store: BeliefStore) -> list[tuple[int, str, Kind]]:
    """按出现位置排序的实体提及；同一位置取最长名字。“我”指玩家自己。"""
    found: dict[int, tuple[int, str, Kind]] = {}
    for eid, sk in store.entities.items():
        for alias in _aliases(sk.name, sk.kind):
            start = text.find(alias)
            while start != -1:
                prev = found.get(start)
                if prev is None or len(alias) > prev[0]:
                    found[start] = (len(alias), eid, sk.kind)
                start = text.find(alias, start + 1)
    if "我" in text:
        found.setdefault(text.find("我"), (1, store.owner, Kind.PERSON))
    return [(pos, eid, kind) for pos, (_, eid, kind) in sorted(found.items())]


def rule_parse(text: str, store: BeliefStore) -> Parsed:
    """按优先级尝试每个命中关键词的操作，返回第一个角色齐全的解析（“揣进兜里”的“进”不该赢过“揣”）。"""
    t = text.strip().lower()
    ops = [o for o, words in _OPS if any(w in t for w in words)]
    if not ops:
        return Parsed(None, clarification="没听懂。试试：拿钥匙 / 去仓库入口 / 查看桌面 / 问守卫钥匙在哪 / 等待")
    attempts = [_parse_as(op, text, t, store) for op in ops]
    return next((p for p in attempts if p.candidate is not None), attempts[0])


def _parse_as(op: Op, text: str, t: str, store: BeliefStore) -> Parsed:
    manner = Manner.CAREFUL if any(w in t for w in _CAREFUL) else (
        Manner.ROUGH if any(w in t for w in _ROUGH) else Manner.NORMAL)
    me = store.owner
    here = store.location_of(me)
    ms = _mentions(t, store)

    def first(*kinds: Kind, after: int = -1, exclude: tuple[str, ...] = ()) -> tuple[int, str] | None:
        for pos, eid, kind in ms:
            if pos > after and kind in kinds and eid not in exclude:
                return pos, eid
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
    return Parsed(cand, utterance, source="rules")


def normalize(c: Candidate, store: BeliefStore) -> Candidate:
    """“朝那扇门走” = 去门那边的地点（按玩家以为的门连接关系换算）。"""
    sk = store.sketch(c.target or "")
    if c.op == Op.MOVE and sk is not None and sk.kind == Kind.DOOR:
        here = store.location_of(store.owner)
        others = [b.prop.value for b in store.positives(sk.id, Rel.CONNECTS.value) if b.prop.value != here]
        if len(others) == 1:
            return Candidate(Op.MOVE, str(others[0]), None, c.manner, None)
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
    "只能引用实体表里的 id；做不到或意图不明就把 op 设为 unknown 并给出 clarification。"
    "target=行动直接作用的对象（拿的物品、去的地点、开的门、说话的对象、查看的东西）；"
    "obj=工具或被递交/放置的物品（开锁的钥匙、放下或交出的东西）；不适用的字段填 null。"
    "manner: careful=小心/悄悄/藏，rough=粗暴/用力，否则 normal。"
    "tell/ask 的语义内容用 topic_subject/topic_value 表示“subject 在 value”（ask 时 topic_value 留空）。"
)


def _schema(ids: list[str]) -> dict:
    ref = {"type": "string", "enum": ids, "nullable": True}
    return {
        "type": "object",
        "properties": {
            "op": {"type": "string", "enum": [o.value for o in Op] + ["unknown"]},
            "target": ref, "obj": ref, "topic_subject": ref, "topic_value": ref,
            "topic_holds": {"type": "boolean"},
            "manner": {"type": "string", "enum": [m.value for m in Manner]},
            "clarification": {"type": "string"},
        },
        # 角色字段全部 required（允许 null）：否则模型会干脆省略 target
        "required": ["op", "target", "obj", "manner", "topic_subject", "topic_value", "topic_holds", "clarification"],
    }


def _table(store: BeliefStore) -> str:
    rows = []
    for eid, sk in sorted(store.entities.items()):
        loc = store.location_of(eid)
        where = f"，你认为在 {loc}" if loc else ""
        rows.append(f"- {eid}：{sk.name}（{sk.kind.value}{where}）")
    return "\n".join(rows)


class IntentParser:
    def __init__(self, llm: LLMClient | None = None, prefer_llm: bool = False) -> None:
        self.llm = llm
        self.prefer_llm = prefer_llm

    def parse(self, text: str, store: BeliefStore) -> Parsed:
        ruled = rule_parse(text, store)
        if self.llm is None or (ruled.candidate is not None and not self.prefer_llm):
            return ruled
        try:
            return self._llm_parse(text, store) or ruled
        except LLMUnavailable as e:
            log.warning("LLM 解析不可用，回退规则: %s", e)
            return ruled

    def _llm_parse(self, text: str, store: BeliefStore) -> Parsed | None:
        assert self.llm is not None
        ids = sorted(store.entities)
        prompt = f"你是 {store.owner}。你认识的实体：\n{_table(store)}\n\n玩家输入：{text}"
        data = parse_json(self.llm.generate(prompt, system=_SYSTEM, schema=_schema(ids), temperature=0.0))
        if data.get("op") in (None, "unknown"):
            return Parsed(None, clarification=data.get("clarification") or "请说得具体一些。", source="llm")
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
        if _invalid(cand, store):
            return None
        utterance = text.strip() if cand.op in (Op.TELL, Op.ASK) else None
        return Parsed(cand, utterance, source="llm")
