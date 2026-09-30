"""
[INPUT]: 依赖 cognition 的 BeliefStore，core 的 Kind / Op / Social，language/command 的 GESTURE_WORDS / Mention / action_hits，
         language/render 的 STATUS_LEXICON / STATUS_EXCLUSIONS
[OUTPUT]: 对外提供 pose_of()（不带主语的动作短语）、witness()（姿态闸门：只留看得见的那一截，或给出场内的否定）、
          own_words()（一句话是不是玩家原文里的一段）、MIN_NAME
[POS]: language 的姿态闸门。带姿态的 WAIT 会被在场的人亲眼看见、写进他们的记忆、再被叙述者当作有出处的原文——
       所以姿态只能是玩家自己身上的动作：不许夹带结果（倒地不起、刺穿）、状态（点穴、流血、学成）、别人的举动
       （别人只能是“向/对/朝/打量……”的宾语）、不在手里的东西与玩家原文里没有的陌生名字（不回显）。
       动作只取第一个分句，后面紧跟的“道：……”只在那番话是玩家自己打出来的字时才留下；越界的动作退回句中的姿态词
       或按言语行为给出的中性姿态，都没有就不推进、给场内说法。parser 与解释器共用；本模块不依赖 parser（实体提及由调用方交来）。
       局限（如实）：只做词法近似，拦的是常见说法；漏网的仍由叙述闸门与内核兜底——姿态从来不改变任何事实
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence

from tianlong.cognition import BeliefStore
from tianlong.core import Kind, Op, Social
from tianlong.language.command import GESTURE_WORDS, Mention, action_hits
from tianlong.language.render import STATUS_EXCLUSIONS, STATUS_LEXICON

# ============================================================
#  词表：分句、宾语位置、越界的动作与说法、退回的中性姿态
# ============================================================

_STOPS = "，,；;。．.！!？?：:“”「」『』\"\n"
_CLAUSE = re.compile(f"[^{_STOPS}]+")
_SAY = "说道|笑道|喝道|问道|叫道|喊道|说|道"
_SAY_TAIL = re.compile(f"(?:{_SAY})$")
# 动作之后紧跟的一番话：“拱手道：……”“叹了口气，自言自语道：……”（引子至多六个字，以言说动词收尾）
_SPOKEN = re.compile(f"(?:[，,]\\s*[^{_STOPS}]{{0,6}}?(?:{_SAY})|(?:{_SAY}))?\\s*[：:“「『\"]")
# 别人（与不在手里的东西）只能是玩家动作的宾语：同一分句里前头有介词、看/指一类的动词、走到某人跟前
_OBJECT_MARK = re.compile(r"向|对|朝|冲|跟|和|同|与|替|为|打量|看|瞧|望|瞪|盯|瞥|瞅|指|凝视|注视|端详|观察|查看"
                          r"|走到|来到|走近|凑到|凑近|靠近|站到|退到|躲到|挨着|绕到|站在|坐在|躲在|跟在")
_PRONOUNS = ("他们", "她们", "你们", "他", "她", "你", "它")
# 这些操作带着宾语就是行动而不是姿态（拿、打、读、用、交、放、开锁）；“给他磕头”的“给”只是介词
_ACTING = frozenset({Op.ATTACK, Op.TAKE, Op.STUDY, Op.USE, Op.GIVE, Op.PUT, Op.UNLOCK, Op.LOCK})
_SEARCH = frozenset({"搜", "检查", "找找", "search"})
# 结果与施展：姿态说不出这些（成败只由内核裁定）
_CLAIMS = ("施展", "使出", "运起", "运功", "催动", "倒地", "倒下", "不起", "刺穿", "刺中", "击中", "打中", "砍中", "劈中",
           "正中", "打倒", "击倒", "打翻", "打飞", "踢飞", "震飞", "吐血", "丧命", "毙命", "杀死", "杀了", "死了", "昏倒",
           "昏了过去", "晕倒", "晕了过去", "夺下", "夺过", "抢过", "抢走", "拿下", "得手", "到手", "断了", "折断", "碎了", "脱手")
_KOWTOW = ("磕头", "叩首", "跪拜")
_POSE_WORDS = frozenset((*(w for w, _ in GESTURE_WORDS), *_KOWTOW))
# 言语行为 → 中性姿态：取姿态词表里第一个带这种言语行为的词（拱手、点头、冷笑、跪下……）
_BY_SOCIAL: dict[Social, str] = {}
for _word, _social in GESTURE_WORDS:
    if _social is not None:
        _BY_SOCIAL.setdefault(_social, _word)
_NO_POSE = "那可不是摆个姿势就能办到的。"
_UNKNOWN = "你不知道那是什么。"
_SQUASH = re.compile(r"[\s，,。．.；;！!？?、…~～：:“”「」『』\"'‘’（）()]+")
MIN_NAME = 2          # 别称至少两个字才作拒绝依据（与叙述闸门同一尺度）


def pose_of(text: str, names: Sequence[str] = ()) -> str:
    """姿态 = 不带主语的动作短语（“坐下来喝了口茶”）：去掉开头的“我/你/自己的名字”与句末标点，至多 40 字。"""
    s = text.strip()
    for lead in (*sorted(names, key=len, reverse=True), "我们", "我", "你"):
        if lead and s.startswith(lead):
            s = s[len(lead):]
            break
    return s.strip().rstrip("，,。．.；;！!？?、：:\"'“”‘’「」『』（）() 　…~～")[:40]


def own_words(line: str, text: str) -> bool:
    """这句话是不是玩家原文里的一段（去掉标点与空白、不分大小写再比）：模型只是指认了玩家自己打出来的字。"""
    core = _SQUASH.sub("", line).lower()
    return bool(core) and core in _SQUASH.sub("", text).lower()


# ============================================================
#  闸门
# ============================================================


def _first_clause(pose: str) -> tuple[int, int]:
    """第一个非空分句的跨度，去掉结尾的“说道/道”：“拱手道：幸会”的动作只是拱手。"""
    for m in _CLAUSE.finditer(pose):
        a, b = m.start(), m.end()
        while a < b and pose[a].isspace():
            a += 1
        b = a + len(_SAY_TAIL.sub("", pose[a:b]).rstrip())
        if b > a:
            return a, b
    return 0, 0


def _as_object(pose: str, a: int, m: Mention) -> bool:
    """同一分句里前头有介词或看/指/走到一类的动词（“看看梁上的钟灵”），或后接“的”（“钟灵的肩膀”）。"""
    return bool(_OBJECT_MARK.search(pose[a:m.pos])) or pose.startswith("的", m.pos + m.length)


def _unknown(clause: str, store: BeliefStore, aliases: Mapping[str, Sequence[str]], universe: Iterable[str]) -> list[str]:
    """点了哪些玩家不认识的实体（场景别称两字以上，或调用方交来的名字全集）：先抹掉认识的名字，免得被子串误伤。"""
    known = {n for eid, sk in store.entities.items() for n in (sk.name, *aliases.get(eid, ())) if n}
    hidden = {n for eid, names in aliases.items() if eid not in store.entities for n in names if len(n) >= MIN_NAME}
    hidden = (hidden | {n for n in universe if len(n) >= MIN_NAME}) - known
    rest = clause
    for n in sorted(known, key=len, reverse=True):
        rest = rest.replace(n, "\0")
    return [n for n in hidden if n in rest]


def _status(clause: str) -> bool:
    masked = {i for x in STATUS_EXCLUSIONS for m in re.finditer(x, clause) for i in range(m.start(), m.end())}
    return any(not set(range(m.start(), m.end())) <= masked
               for words in STATUS_LEXICON.values() for w in words for m in re.finditer(w, clause))


def _dative(clause: str, e: int, found: Sequence[Mention], a: int, allowed: set[int]) -> bool:
    """“给他磕头”“给钟灵作揖”：给 + 人 + 姿态词，“给”只是介词。"""
    person = next((m for m in found if m.pos == a + e and m.kind == Kind.PERSON), None)
    j = e + (person.length if person else next((len(p) for p in _PRONOUNS if clause.startswith(p, e)), 0))
    return j > e and j in allowed


def _governs(clause: str, e: int, at: list[int]) -> bool:
    """行动词之后最近的那个名字，中间没隔着介词或看/指一类的动词，就是它的宾语（“拿出勇气向龚光杰拱手”不是拿龚光杰）。"""
    nearest = min((i for i in at if i >= e), default=None)
    return nearest is not None and not _OBJECT_MARK.search(clause[e:nearest])


def _oversteps(pose: str, a: int, b: int, found: Sequence[Mention], strange: Sequence[str]) -> bool:
    """这一截越界了吗：状态、结果、带着宾语（认识的实体，或玩家自己写出的陌生名字）的行动、别人作主语。"""
    clause = pose[a:b]
    found = [m for m in found if a <= m.pos and m.pos + m.length <= b]
    if _status(clause) or any(w in clause for w in _CLAIMS):
        return True
    hits = action_hits(clause)
    acting = [(s, e, op) for s, e, op in hits if op in _ACTING or (op == Op.INSPECT and clause[s:e] in _SEARCH)]
    allowed = {s for s, e, op in hits if (s, e, op) not in acting}
    names = [m.pos - a for m in found] + [i for n in strange for i in range(len(clause)) if clause.startswith(n, i)]
    for s, e, op in acting:
        if clause[s:e] == "给" and _dative(clause, e, found, a, allowed):
            continue
        if op == Op.ATTACK and (clause[s:e] != "打" or any(i >= e for i in names)):
            return True                          # “一掌拍出”“出招”：动手只能是行动（单字“打”另有打坐、打个哈欠）
        if _governs(clause, e, names):
            return True                          # “打开宫门”“捡起北冥神功帛卷”：带着宾语的行动
    return any(m.kind == Kind.PERSON and not _as_object(pose, a, m) for m in found)


def witness(pose: str, social: Social | None, store: BeliefStore, found: Sequence[Mention],
            aliases: Mapping[str, Sequence[str]] | None = None, universe: Iterable[str] = (),
            said: str = "") -> tuple[str | None, str | None]:
    """(看得见的姿态, 场内的否定)。found 是 pose 里的实体提及（与 pose 同一套下标，由调用方的 mentions() 给出），
    said 是玩家的原文：原文里本来就有的陌生名字不算模型编的（“坐在垫子上”不因“垫子”是某件没见过的东西的别称而被拒）。
    - 动作只取第一个分句：“哈哈大笑，一剑刺穿了龚光杰”只看得见哈哈大笑；紧跟的“道：……”是玩家原话才留下；
    - 原文里没有的陌生名字、不在手里又不是宾语的东西：不推进，给场内否定（不回显名字）；
    - 状态、结果、带宾语的行动、别人作主语：退回句中的姿态词，其次按言语行为给中性姿态，都没有才不推进。
    两项都是 None = 没有可做的姿态（调用方追问）。"""
    a, b = _first_clause(pose)
    if b <= a:
        return None, None
    clause = pose[a:b]
    owner = store.owner
    found = [m for m in found if m.eid != owner]
    strange = _unknown(clause, store, aliases or {}, universe)
    if any(n not in said for n in strange):
        return None, _UNKNOWN
    for m in found:
        if a <= m.pos < b and m.kind == Kind.ITEM and store.location_of(m.eid) != owner and not _as_object(pose, a, m):
            sk = store.sketch(m.eid)
            return None, f"你身上并没有{sk.name if sk else '那样东西'}。"
    if _oversteps(pose, a, b, found, strange):
        word = next((clause[s:e] for s, e, _ in action_hits(clause) if clause[s:e] in _POSE_WORDS), None)
        word = word or (_BY_SOCIAL.get(social) if social is not None else None)
        return (word, None) if word else (None, _NO_POSE)
    spoken = _SPOKEN.match(pose, b)
    if spoken is not None:
        q = spoken.end() - 1                                  # 冒号或左引号
        words = pose[q + 1:].strip().strip("“”「」『』\"").strip()
        lead_ok = not _oversteps(pose, b, q, found, strange)
        if words and lead_ok and own_words(words, said):
            return f"{pose[a:q].rstrip()}：“{words}”", None
    return clause, None
