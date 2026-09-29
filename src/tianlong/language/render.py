"""
[INPUT]: 依赖 core 的 Percept / Modality / Op / Outcome / Kind / SKILLS / STATUS_ATTRS，language/templates 的 Names / render_percept
[OUTPUT]: 对外提供 fact_lines()（本回合允许讲的事实清单）、RenderPlan / build_plan()（渲染计划）、Violation / check()（叙述闸门）、
          check_utterance()（对白闸门）、RenderStatus / Rendered（渲染结果与来源）、
          词表 STATUS_LEXICON / COMMITMENT_WORDS / ARRIVAL_VERBS / ATTRIBUTION_VERBS / NEGATIONS / QUANTIFIERS
[POS]: language 的“文字 ≠ 事实”闸门。提示词约束拦不住一次成功调用返回的错误非空文本，这里用确定性的词法检查拦：
       点名清单外的人与物、状态升级（受伤→被制、略有所得→学成）、瞬移、物品复制、编造承诺、把传闻说成叙述者确认的事实——
       命中任何一条即回退确定模板。宁可错杀：误报只让这一回合的文字退回模板，世界结算不受任何影响。
       局限（如实）：只做词法比对，不做命题级语义理解——别称与代词不参与点名检查（别称只用于瞬移检查）；
       传闻只检查“说话者名字之后有言说动词”，LLM 若在带归属的句子之外再以叙述口吻复述同一命题，本闸门查不出；
       否定只看词前同一分句内四字以内的否定词；数量只认“数词 + 量词 + 物品名”的直接说法。
       出处优先：清单、外观描写与原话里本来就有的词、名字、数量与“抵达”说法，照搬不算违规
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum

from tianlong.core import SKILLS, STATUS_ATTRS, Kind, Modality, Op, Outcome, Percept
from tianlong.language.templates import Names, render_percept

# ============================================================
#  词表：闸门的全部语言知识集中在这里，改词表不改逻辑
# ============================================================

# 状态 → 断言该状态的说法（受伤不等于被制住，略有所得不等于学成）
STATUS_LEXICON: dict[str, tuple[str, ...]] = {
    "subdued": ("点了穴", "点穴", "穴道被制", "被制", "制住", "动弹不得"),
    "poisoned": ("中了毒", "中毒", "毒发"),
    "wounded": ("受了伤", "受伤", "负伤", "挂彩", "重伤", "伤口", "流血"),
    "mastered": ("融会贯通", "贯通", "学成", "练成", "大成"),
}
# 承诺与誓言：引擎没有“承诺”这种行动，叙述与对白都不能凭空多出一句保证
COMMITMENT_WORDS: tuple[str, ...] = ("答应", "保证", "承诺", "发誓", "起誓", "许诺", "应允", "一定会", "誓要")
# 抵达动词：其后同一分句内出现的地点必须是本回合真的到达的地方
ARRIVAL_VERBS: tuple[str, ...] = ("来到", "走进", "进入", "抵达", "到达", "回到", "走到", "踏入", "踏进", "钻进", "跳进", "到了")
_NOT_ARRIVAL_BEFORE_DAOLE = frozenset("看听见想找得遇碰感觉闻受收做办等迟猜料意顾说识注察")   # “听到了”“想到了”不是抵达
ARRIVAL_WINDOW = 10
# 言语归属：复述传闻时，说话者名字之后应当跟着这些动词之一
ATTRIBUTION_VERBS: tuple[str, ...] = ("说", "道", "告诉", "称", "问", "讲", "言", "答", "提起", "透露", "低语", "喊")
ATTRIBUTION_WINDOW = 10
# 否定：词前同一分句四字以内出现即视为未断言（“未能融会贯通”不是学成）；“不禁”“不料”之类不是否定
NEGATIONS = frozenset("未没不无非别莫")
_NOT_NEGATION = frozenset({"不但", "不仅", "不料", "不禁", "不由", "不觉", "不久", "不时", "无意", "无端"})
NEGATION_WINDOW = 4
# 数量：数词 + 量词 + 修饰（至多两字，或以“的”收尾的六字以内）+ 物品名；“又一/另一”意味着比现有的多一件
QUANTIFIERS: dict[str, int] = {"好几": 2, "两": 2, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9,
                               "十": 10, "几": 2, "数": 2}
ANOTHER: tuple[str, ...] = ("又一", "另一")
CLASSIFIERS = "把个只件枚支柄瓶卷本块串颗粒张根份对双"
_PUNCT = "，。；！？、,.;!?：:“”\"'（）()\n"


# ============================================================
#  事实清单：本回合值得讲、也只允许讲的事
# ============================================================


def fact_lines(viewer: str, percepts: Sequence[Percept], names: Names, show_scene: bool = False) -> list[str]:
    """本回合值得讲的事：事件感知全部讲；环顾只在移动/查看之后（或被要求时）讲。"""
    moved = any(
        p.modality == Modality.SELF and p.event and p.event.kind in (Op.MOVE.value, Op.INSPECT.value)
        and p.event.outcome == Outcome.SUCCESS
        for p in percepts
    )
    lines: list[str] = []
    for p in percepts:
        if p.modality == Modality.SCENE:
            if show_scene or moved:
                lines.append("你看到：" + render_percept(p, names, viewer, me="你"))
        else:
            lines.append(render_percept(p, names, viewer, me="你"))
    return list(dict.fromkeys(lines))      # 同一分钟里的三声响动，只说一次


# ============================================================
#  渲染计划：LLM 能说什么，由玩家本回合的感知决定
# ============================================================


@dataclass(frozen=True, slots=True)
class RenderPlan:
    viewer: str
    lines: tuple[str, ...]                    # 允许的事实清单（也是回退模板的正文）
    names: frozenset[str]                     # 允许点名的实体：本回合感知里出现的人、物、地点、通道 + 观察者自己
    statuses: frozenset[str]                  # 本回合感知里确实出现的状态：wounded / poisoned / subdued / mastered
    arrivals: frozenset[str]                  # 允许作为“抵达”宾语的地点（名与别称）：本回合真的有人走到了那里
    places: frozenset[str]                    # 观察者知道的全部地点（名与别称）：抵达动词之后只查这些
    items: tuple[tuple[str, int], ...]        # 本回合出现的物品名 → 件数
    hearsay: tuple[str, ...]                  # 传闻的说话者名：复述其说法必须带言语归属
    source: str = ""                          # 清单 + 外观描写 + 时辰原文：出现在其中的词算有出处
    commitment: bool = False                  # 本回合是否存在承诺；引擎没有承诺行动，恒为 False


def build_plan(viewer: str, percepts: Sequence[Percept], names: Names, show_scene: bool = False,
               looks: Sequence[str] = (), lapse: str = "",
               aliases: Mapping[str, Sequence[str]] | None = None) -> RenderPlan:
    """与 fact_lines() 同样的输入：观察者、本回合感知、观察者的名称表。looks/lapse 是一并交给 LLM 的外观描写与时辰。"""
    aliases = aliases or {}
    table: dict[str, tuple[str, Kind]] = {sk.id: (sk.name, sk.kind) for p in percepts for sk in p.sketches}
    table.update({eid: (sk.name, sk.kind) for eid, sk in names.items()})
    ids: set[str] = {viewer}
    statuses: set[str] = set()
    arrived: set[str] = set()
    hearsay: list[str] = []
    for p in percepts:
        ev = p.event
        if ev is not None:
            ids.update(x for x in (ev.actor, ev.target, ev.obj, ev.place) if x)
            if ev.kind == Op.MOVE.value and ev.outcome == Outcome.SUCCESS and ev.target:
                arrived.add(ev.target)
            if ev.reason == "subdued":
                statuses.add("subdued")          # “穴道被制，动弹不得”是失败原因，也是此刻的状态
            if ev.outcome == Outcome.SUCCESS and ev.reason == "mastered":
                statuses.add("mastered")
        for f in p.facts:
            ids.add(f.prop.subject)
            if not f.prop.is_attr and isinstance(f.prop.value, str):
                ids.add(f.prop.value)
            if f.holds and f.prop.is_attr:
                key = f.prop.attr_key
                if key in STATUS_ATTRS:
                    statuses.add(key)
                elif key in SKILLS:
                    statuses.add("mastered")
        ids.update(sk.id for sk in p.sketches)
        if p.modality == Modality.SPEECH and p.facts and p.informant and p.informant != viewer:
            hearsay.append(p.informant)

    def name(eid: str) -> str | None:
        return table[eid][0] if eid in table else None

    def with_aliases(eids: Iterable[str]) -> frozenset[str]:
        return frozenset(n for e in eids for n in (name(e), *aliases.get(e, ())) if n)

    items = Counter(n for e in ids if (n := name(e)) and table[e][1] == Kind.ITEM)
    lines = fact_lines(viewer, percepts, names, show_scene)
    return RenderPlan(
        viewer=viewer,
        lines=tuple(lines),
        names=frozenset(n for e in ids if (n := name(e))),
        statuses=frozenset(statuses),
        arrivals=with_aliases(arrived),
        places=with_aliases(e for e, (_, k) in table.items() if k == Kind.PLACE),
        items=tuple(sorted(items.items())),
        hearsay=tuple(dict.fromkeys(n for i in hearsay if (n := name(i)))),
        source="\n".join([*lines, *looks, lapse]),
    )


# ============================================================
#  词法积木
# ============================================================


@dataclass(frozen=True, slots=True)
class Violation:
    kind: str      # entity / status / teleport / duplicate / commitment / hearsay / subject / empty
    detail: str


def _mentions(text: str, words: Iterable[str]) -> list[tuple[int, str]]:
    """最长匹配扫描：“后山崖顶”不会再被数成“后山”，“剑湖宫大殿”不会再被数成别的短名。"""
    by_first: dict[str, list[str]] = {}
    for w in sorted({w for w in words if w}, key=lambda w: (-len(w), w)):
        by_first.setdefault(w[0], []).append(w)
    out: list[tuple[int, str]] = []
    i = 0
    while i < len(text):
        hit = next((w for w in by_first.get(text[i], ()) if text.startswith(w, i)), None)
        if hit is None:
            i += 1
            continue
        out.append((i, hit))
        i += len(hit)
    return out


def _clause_before(text: str, i: int, width: int) -> tuple[int, str]:
    start = max(0, i - width)
    for j in range(i - 1, start - 1, -1):
        if text[j] in _PUNCT:
            start = j + 1
            break
    return start, text[start:i]


def _negated(text: str, i: int) -> bool:
    start, window = _clause_before(text, i, NEGATION_WINDOW)
    return any(ch in NEGATIONS and text[start + k:start + k + 2] not in _NOT_NEGATION for k, ch in enumerate(window))


def _asserted(text: str, words: Iterable[str]) -> list[tuple[int, str]]:
    """文本里被肯定说出的词（去掉被否定的）。"""
    return [(i, w) for i, w in _mentions(text, words) if not _negated(text, i)]


def _arrivals(text: str) -> list[tuple[int, str]]:
    return [(i, v) for i, v in _mentions(text, ARRIVAL_VERBS)
            if not (v == "到了" and i > 0 and text[i - 1] in _NOT_ARRIVAL_BEFORE_DAOLE)]


def _clause_after(text: str, i: int, width: int) -> str:
    window = text[i:i + width]
    for k, ch in enumerate(window):
        if ch in _PUNCT:
            return window[:k]
    return window


def _quantified(text: str, item: str, count: int) -> list[tuple[int, str]]:
    """“两把钥匙”“另一把铜钥匙”“两把一模一样的钥匙”：返回 (暗示的件数, 原文)。“又一个人拿起钥匙”不算。"""
    nums = "|".join(sorted([*QUANTIFIERS, *ANOTHER], key=lambda q: -len(q)))
    free = f"[^{re.escape(_PUNCT)}]"
    pat = re.compile(f"({nums})[{CLASSIFIERS}](?:{free}{{0,5}}的|{free}{{0,2}}){re.escape(item)}")
    return [(count + 1 if m.group(1) in ANOTHER else QUANTIFIERS[m.group(1)], m.group(0)) for m in pat.finditer(text)]


def _attributed(text: str, speaker: str) -> bool:
    """说话者名字之后不远处有言说动词：“守卫告诉你……”“钟灵笑道……”。"""
    for i, n in _mentions(text, (speaker,)):
        after = text[i + len(n):i + len(n) + ATTRIBUTION_WINDOW]
        if any(v in after for v in ATTRIBUTION_VERBS):
            return True
    return False


def _unsourced(text: str, source: str, words: Iterable[str]) -> list[str]:
    """text 里肯定说出、而 source 里并未肯定说出的词。"""
    words = tuple(words)
    sourced = {w for _, w in _asserted(source, words)}
    return [w for _, w in _asserted(text, words) if w not in sourced]


# ============================================================
#  叙述闸门
# ============================================================


def check(text: str, plan: RenderPlan, known_names: Iterable[str] = ()) -> tuple[Violation, ...]:
    """known_names 是“可能被点名”的全集（玩家认识的 + 场景里所有实体），只用于拒绝：不在计划里的名字出现即违规。"""
    out: list[Violation] = []
    universe = set(known_names) | plan.names
    sourced = {n for _, n in _mentions(plan.source, universe)}

    # ---- 1. 点名：清单外的人与物 ----
    for _, n in _mentions(text, universe):
        if n not in plan.names and n not in sourced:
            out.append(Violation("entity", n))

    # ---- 2. 状态升级：本回合没有的状态不许出现 ----
    for status, words in STATUS_LEXICON.items():
        if status not in plan.statuses:
            out += [Violation("status", f"{status}:{w}") for w in _unsourced(text, plan.source, words)]

    # ---- 3. 编造承诺 ----
    if not plan.commitment:
        out += [Violation("commitment", w) for w in _unsourced(text, plan.source, COMMITMENT_WORDS)]

    # ---- 4. 瞬移：抵达动词之后的地点必须是本回合真的到达的（原话里转述的“回到港口”照搬不算）----
    for i, verb in _arrivals(text):
        hits = _mentions(_clause_after(text, i + len(verb), ARRIVAL_WINDOW), plan.places)
        if hits and hits[0][1] not in plan.arrivals and verb + hits[0][1] not in plan.source:
            out.append(Violation("teleport", verb + hits[0][1]))

    # ---- 5. 物品复制（外观描写本就写着“插着几柄长剑”的，照此说不算）----
    for item, count in plan.items:
        ceiling = max([count, *(implied for implied, _ in _quantified(plan.source, item, count))])
        out += [Violation("duplicate", phrase) for implied, phrase in _quantified(text, item, count) if implied > ceiling]

    # ---- 6. 传闻不得变成叙述者确认的事实 ----
    out += [Violation("hearsay", who) for who in plan.hearsay if not _attributed(text, who)]
    return tuple(dict.fromkeys(out))


# ============================================================
#  对白闸门：润色只能换说法，不能换内容
# ============================================================


def check_utterance(line: str, allowed: Iterable[str], universe: Iterable[str], plain: str,
                    subject_forms: Iterable[str]) -> tuple[Violation, ...]:
    """allowed：说话者、听者、话题主语与宾语的名字；plain：模板说法（意图原文）；subject_forms：话题主语的可接受称呼。"""
    allowed = set(allowed)
    universe = set(universe) | allowed
    sourced = {n for _, n in _mentions(plain, universe)}
    out = [Violation("entity", n) for _, n in _mentions(line, universe) if n not in allowed and n not in sourced]
    out += [Violation("commitment", w) for w in _unsourced(line, plain, COMMITMENT_WORDS)]
    for status, words in STATUS_LEXICON.items():
        out += [Violation("status", f"{status}:{w}") for w in _unsourced(line, plain, words)]
    forms = [f for f in subject_forms if f]
    if forms and not any(f in line for f in forms):
        out.append(Violation("subject", "/".join(forms)))
    return tuple(dict.fromkeys(out))


# ============================================================
#  渲染结果：世界结算成功与文字生成成功分开记录
# ============================================================


class RenderStatus(StrEnum):
    TEMPLATE = "template"                # 没有模型（或无事可润色）：确定模板
    LLM = "llm"                          # 模型润色且通过闸门
    GATED_FALLBACK = "gated_fallback"    # 模型返回了文字，但被闸门拦下，回退模板
    LLM_UNAVAILABLE = "llm_unavailable"  # 模型调用失败，回退模板


@dataclass(frozen=True, slots=True)
class Rendered:
    text: str
    status: RenderStatus
    violations: tuple[Violation, ...] = field(default=())
