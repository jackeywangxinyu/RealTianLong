"""
[INPUT]: 依赖 tianlong.language 的 narrator / render / scene / llm（ScriptedLLM），tianlong.runtime.authority（真实内核产出的玩家感知），
         tianlong.scenarios 的 build_wuliang
[OUTPUT]: 主持人之声验收：流式逐句过闸门且按序、尽早交付；点名清单外实体的句子被丢而其余照常流出；替玩家开口（引语与念头）、
          NPC 台词点名许可之外的人、凭空多出的说话者、台词里的状态升级各被拦下；丢满两句或一句未过即补模板；模型不可用（含中途失败）
          保留已交付的并补模板；模板把 NPC 言语写成带言语行为的台词；提示词带最近正文与台词要素且没有钟点数字；首句交付早于整段完成；
          分句器处理引号、省略号与流的边界；合法的道谢、挑衅与如实的位置说法不被误伤
[POS]: tests 的主持层叙述；证伪“流式叙述会把没过闸门的句子交给玩家”“主持人替玩家说话”“NPC 说出他不该知道的名字”
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

from __future__ import annotations

import itertools
import re
import time

import pytest

from tianlong.core import Intent, Manner, Op, Social
from tianlong.language.llm import LLMUnavailable, ScriptedLLM
from tianlong.language.narrator import SOCIAL_PHRASES, Narrator, render_voice
from tianlong.language.render import RenderStatus, build_plan, check, check_quotes, sentence_ends
from tianlong.language.scene import SceneBrief, VoiceLine
from tianlong.persistence import InMemoryWorldStore
from tianlong.runtime.authority import WorldAuthority
from tianlong.scenarios import build_wuliang

_ids = itertools.count()
SC = build_wuliang()
KNOWN = frozenset(e.name for e in SC.state.entities.values()) | {a for al in SC.aliases.values() for a in al}
CLOCK = re.compile(r"\d{1,2}\s*[:：]\s*\d{2}|第\s*\d+\s*[日天]")

TAUNT = "你笑什么？下场比划比划！"
GONG = VoiceLine("gongguangjie", "龚光杰", "你", Op.TELL.value, Social.CHALLENGE, None, TAUNT,
                 voice="骄横好胜，说话夹枪带棒", knows="无量剑东宗每五年与西宗比剑一次",
                 may_name=frozenset({"龚光杰", "段誉", "左子穆", "左掌门", "剑湖宫大殿", "大殿"}),
                 answering="段誉方才忍不住笑出声来")
LING = VoiceLine("zhongling", "钟灵", "你", Op.TELL.value, Social.GREET, None, None,
                 voice="娇憨顽皮", may_name=frozenset({"钟灵", "段誉", "龚光杰"}))

G1 = "大殿里的喧笑声忽地一静。"
G2 = "龚光杰霍地站起，冷笑道：“你笑什么？有胆便下场来比划比划！”"
G3 = "满堂目光都落在你身上，等你答话。"
B_ENTITY = "琅嬛福地的深处隐隐传来水声。"
B_PUPPET = "你连忙摆手道：“我这就走。”"
B_NAME = "龚光杰又喝道：“钟灵，你少管闲事！”"
BRIEF = SceneBrief(lines=(GONG,))


@pytest.fixture(scope="module")
def view():
    """龚光杰当众向段誉叫阵（不带命题的自由言语）：段誉本回合的感知与名称表。"""
    auth = WorldAuthority.found(InMemoryWorldStore(), SC)
    v = auth.head().version
    s = auth.settle([Intent(f"gm{next(_ids)}", "gongguangjie", Op.TELL, "duanyu", None, Manner.NORMAL, None, v, TAUNT,
                            Social.CHALLENGE)])
    return [o.percept for o in s.observations_of("duanyu")], auth.store.beliefs(auth.ref, "duanyu").entities


def _narrator(llm=None) -> Narrator:
    return Narrator(llm, SC.setting, SC.lore, SC.style, SC.aliases)


def _run(view, llm, brief=BRIEF, **kw):
    percepts, names = view
    got: list[str] = []
    r = _narrator(llm).narrate_scene("duanyu", percepts, names, brief=brief, show_scene=True, known=KNOWN,
                                     on_text=got.append, **kw)
    assert r.text == "".join(got), "Rendered.text 恒等于交付给玩家的全部文字"
    return r, got


def _script(text: str, **kw) -> ScriptedLLM:
    return ScriptedLLM(lambda prompt, system, schema: text, **kw)


def _template(view, brief=BRIEF, **kw) -> str:
    percepts, names = view
    return _narrator().narrate_scene("duanyu", percepts, names, brief=brief, show_scene=True, known=KNOWN, **kw).text


# ============================================================
#  模板：NPC 的言语写成带言语行为的台词
# ============================================================


def test_template_voices_npc_lines_with_social_phrasing(view):
    text = _template(view, SceneBrief(lines=(GONG, LING)))
    challenge = {f"龚光杰{p.format(to='你')}：“{TAUNT}”" for p in SOCIAL_PHRASES[Social.CHALLENGE]}
    greet = {f"钟灵{p.format(to='你')}。" for p in SOCIAL_PHRASES[Social.GREET]}
    rows = text.splitlines()
    assert challenge & set(rows) and greet & set(rows), text
    assert "听见龚光杰对你说" not in text, "清单里的那句言语被台词取代，不重复"
    assert "你看到：" in text, "其余事实照旧"
    assert text == _template(view, SceneBrief(lines=(GONG, LING))), "相同输入 → 相同文字"
    variants = {render_voice(LING, salt=f"上一回合{i}") for i in range(12)}
    assert len(variants) >= 2 and variants <= greet, "措辞随上下文轮换，但只在言语行为的措辞表里选"


def test_template_without_llm_is_delivered_through_the_sink(view):
    r, got = _run(view, None)
    assert r.status == RenderStatus.TEMPLATE and got == [r.text]


# ============================================================
#  流式：逐句过闸门，通过即交付
# ============================================================


def test_streaming_delivers_passing_sentences_in_order(view):
    r, got = _run(view, _script(G1 + G2 + G3, chunk=4))
    assert got == [G1, G2, G3]
    assert r.status == RenderStatus.LLM and r.violations == ()


def test_sentence_naming_unlisted_entity_is_dropped_and_rest_streams(view):
    r, got = _run(view, _script(G1 + B_ENTITY + G3, chunk=5))
    assert got == [G1, G3]
    assert r.status == RenderStatus.LLM and {v.kind for v in r.violations} == {"entity"}


def test_puppeting_quote_is_dropped(view):
    r, got = _run(view, _script(G1 + B_PUPPET + G2))
    assert got == [G1, G2] and {v.kind for v in r.violations} == {"puppet"}, "玩家什么也没说，主持人不能替他开口"
    said = SceneBrief(lines=(GONG,), player_line="我这就走")
    r, got = _run(view, _script(G1 + B_PUPPET + G2), said)
    assert got == [G1, B_PUPPET, G2] and r.violations == (), "玩家自己的原话可以照引"


def test_npc_quote_naming_outside_may_name_is_dropped(view):
    assert check(B_NAME, build_plan("duanyu", view[0], view[1], True, aliases=SC.aliases), KNOWN) == (), \
        "钟灵就在殿上，叙述闸门放行——拦下它的是台词闸门"
    r, got = _run(view, _script(G1 + B_NAME + G3))
    assert got == [G1, G3]
    assert [v for v in r.violations if v.kind == "quote_entity"][0].detail == "龚光杰:钟灵"


def test_two_drops_fall_back_to_template_after_streamed_prose(view):
    r, got = _run(view, _script(G1 + B_ENTITY + B_PUPPET + G3))
    assert r.status == RenderStatus.GATED_FALLBACK
    assert got[0] == G1 and G3 not in r.text, "丢满两句即停，不再读流"
    tail = _template(view)
    assert r.text == G1 + "\n" + tail, "已交付过正文的，换行后只补清单与台词"
    assert {v.kind for v in r.violations} == {"entity", "puppet"}


def test_nothing_accepted_falls_back_to_full_template(view):
    r, got = _run(view, _script(B_ENTITY))
    assert r.status == RenderStatus.GATED_FALLBACK and r.text == _template(view) and got == [r.text]
    r, _ = _run(view, _script("   "))
    assert r.status == RenderStatus.GATED_FALLBACK and [v.kind for v in r.violations] == ["empty"]


def test_llm_unavailable_falls_back_to_template(view):
    def boom(prompt, system, schema):
        raise LLMUnavailable("offline")
    r, got = _run(view, ScriptedLLM(boom))
    assert r.status == RenderStatus.LLM_UNAVAILABLE and r.text == _template(view) and got == [r.text]


class _Flaky:
    """流到一半断线：先吐出完整的一句和半句，然后失败。"""
    model = "flaky"

    def stream(self, prompt, *, system=None, max_tokens=None):
        yield G1
        yield G2[:6]
        raise LLMUnavailable("connection reset")


def test_llm_failing_mid_stream_keeps_delivered_and_appends_template(view):
    r, got = _run(view, _Flaky())
    assert r.status == RenderStatus.LLM_UNAVAILABLE
    assert got[0] == G1 and r.text == G1 + "\n" + _template(view), "半句话不交付"


class _GenerateOnly:
    model = "plain"

    def __init__(self, text: str) -> None:
        self.text = text

    def generate(self, prompt, *, system=None, schema=None, temperature=None):
        return self.text


def test_generate_only_client_is_gated_sentence_by_sentence(view):
    r, got = _run(view, _GenerateOnly(G1 + B_ENTITY + G3))
    assert got == [G1, G3] and r.status == RenderStatus.LLM


def test_voice_lines_are_sourced_even_without_a_matching_percept(view):
    """会话交来的台词本身就是出处：说话者即便不在本回合的感知里，也可以被点名、被写成对白。"""
    percepts, names = view
    comfort = VoiceLine("zhongling", "钟灵", "你", Op.TELL.value, Social.COMFORT, None, "书呆子，别怕他！",
                        may_name=frozenset({"钟灵", "段誉", "龚光杰"}))
    brief = SceneBrief(lines=(GONG, comfort))
    got: list[str] = []
    line = "钟灵在梁上笑道：“书呆子，别怕他！”"
    r = _narrator(_script(G2 + line)).narrate_scene("duanyu", percepts, names, brief=brief, known=KNOWN,
                                                     on_text=got.append)
    assert got == [G2, line] and r.status == RenderStatus.LLM, r.violations
    plain = _narrator().narrate_scene("duanyu", percepts, names, brief=brief, known=KNOWN).text
    assert plain.splitlines()[-1] == render_voice(comfort), "清单里对不上的台词补在最后"


def test_clock_label_in_prose_is_dropped(view):
    r, got = _run(view, _script(G1 + "此时已是第1日 18:27。" + G3))
    assert got == [G1, G3] and {v.kind for v in r.violations} == {"clock"}


# ============================================================
#  提示词：最近正文、台词要素、没有钟点数字
# ============================================================


def test_prompt_carries_recent_passages_and_voice_lines(view):
    llm = _script(G1)
    old = "最早那一回合的正文，不该再出现。"
    long = "你沿着回廊慢慢走去，" + "檐下灯笼摇晃，" * 60 + "（不觉已是第1日 18:27）末尾一句。"
    brief = SceneBrief(lines=(GONG,), recent=(old, "第二段正文。", "第三段正文。", long), player_line="在下只是觉得好笑",
                       stakes="龚光杰等着你答话")
    _run(view, llm, brief, command="拱手赔笑", lapse="第1日 19:00")
    system, prompt = llm.prompts[0]
    assert old not in prompt and "第二段正文。" in prompt and "第三段正文。" in prompt and "末尾一句。" in prompt
    passage = next(line for line in prompt.splitlines() if line.startswith("【3】"))
    assert len(passage) <= 310, "每段正文只留末尾约 300 字"
    for part in (TAUNT, "叫阵", GONG.voice, GONG.knows, GONG.answering, "左子穆", "闲话", "在下只是觉得好笑",
                 "拱手赔笑", "龚光杰等着你答话"):
        assert part in prompt, part
    assert "听见龚光杰对你说" not in prompt, "要说的话不在事实清单里重复一遍"
    assert "戌时" in prompt, "时辰换成文字"
    assert not CLOCK.search(system) and not CLOCK.search(prompt), "提示词里没有钟点数字"
    for rule in ("你", "80~250", "某某道：“……”", "可点名", "不替玩家", "钩子", "钟点", "按兵不动"):
        assert rule in system, rule
    assert SC.style in system


def test_legacy_prompt_still_starts_with_player_input(view):
    llm = _script(G1)
    percepts, names = view
    _narrator(llm).narrate_rendered("duanyu", percepts, names, command="飞上房梁", known=KNOWN)
    assert llm.prompts[0][1].startswith("玩家的输入：飞上房梁") and "听见龚光杰对你说" in llm.prompts[0][1]


# ============================================================
#  延迟：第一句一完整就交付
# ============================================================


def test_first_sentence_is_delivered_before_the_stream_ends(view):
    llm = _script(G1 + G2 + G3 + G3.replace("答话", "开口"), first_delay=0.05, chars_per_second=150, chunk=3)
    stamps: list[float] = []
    percepts, names = view
    t0 = time.perf_counter()
    r = _narrator(llm).narrate_scene("duanyu", percepts, names, brief=SceneBrief(lines=(GONG,)), show_scene=True,
                                     known=KNOWN, on_text=lambda s: stamps.append(time.perf_counter() - t0))
    total = time.perf_counter() - t0
    assert r.status == RenderStatus.LLM and len(stamps) == 4
    assert stamps[0] < 0.6 * total, (stamps, total)
    assert stamps == sorted(stamps)


# ============================================================
#  分句器
# ============================================================


@pytest.mark.parametrize("text, ends", [
    ("你好。他来了！", [3, 7]),
    ("龚光杰道：“你笑什么？下场来！”说着拔出长剑。", [23]),
    ("“你笑什么？”龚光杰喝道。满堂哗然。", [13, 18]),
    ("龚光杰道：“下场来！”满堂目光都落在你身上。", [11, 22]),
    ("满堂哗然……“好！”", [6, 10]),
    ("他问：“真的？！”\n你一怔", [9]),
])
def test_sentence_ends(text, ends):
    assert sentence_ends(text) == ends


def test_sentence_ends_waits_for_lookahead_in_a_stream():
    assert sentence_ends("你好。", final=False) == [], "后面也许还有省略号或收引号"
    assert sentence_ends("你好。", final=True) == [3]
    assert sentence_ends("龚光杰道：“好！”", final=False) == [], "收引号之后也许还接着“龚光杰喝道”"
    assert sentence_ends("龚光杰道：“好！”你", final=False) == [], "后面这一小句还没写完"
    assert sentence_ends("龚光杰道：“好！”满堂哗然，", final=False) == [9], "那一小句不是归属：引语自成一句"
    assert sentence_ends("“好！”钟灵笑道，", final=False) == [], "带言说动词的小句是归属，句子继续"
    assert sentence_ends("“好！”钟灵笑道：“", final=False) == [4], "以冒号收尾的是下一段引语的引子"
    assert sentence_ends("龚光杰道：“好！”\n", final=False) == [9]


# ============================================================
#  台词闸门：合法对白放行，越界的拦下
# ============================================================


@pytest.fixture(scope="module")
def plan(view):
    return build_plan("duanyu", view[0], view[1], True, aliases=SC.aliases)


MA = VoiceLine("mawude", "马五德", "你", Op.TELL.value, Social.EXPLAIN, "长剑在兵器架上", None,
               may_name=frozenset({"马五德", "段誉"}))
THANKS = VoiceLine("zhongling", "钟灵", "你", Op.TELL.value, Social.THANK, None, "多谢你啦",
                   may_name=frozenset({"钟灵", "段誉"}))


@pytest.mark.parametrize("text, lines", [
    ("钟灵拍手笑道：“书呆子，多谢你啦！”", (THANKS,)),
    ("龚光杰斜眼冷笑道：“就凭你，也配在剑湖宫大殿撒野？”", (GONG,)),
    ("“你笑什么？”龚光杰喝道。", (GONG,)),
    ("马五德捋须道：“长剑就搁在兵器架上。”", (MA,)),
    ("钟灵拨了拨兵器架上的长剑，冲龚光杰做了个鬼脸，笑道：“多谢你啦。”", (THANKS, GONG)),
    ("龚光杰喝道：“下场来！”“你笑什么？”", (GONG,)),
    ("龚光杰收剑入鞘，目光直直落在你脸上，冷笑道：“你笑什么？”", (GONG,)),
    ("龚光杰见你不答，喝道：“你笑什么？”", (GONG,)),
    ("你听见梁上的钟灵笑道：“多谢你啦。”", (THANKS,)),
    ("你身旁的钟灵笑道：“多谢你啦。”", (THANKS,)),
    ("是去是留，由你决定。", (GONG,)),
    ("你打算如何应对？此事须你自己拿定主意。", (GONG,)),
])
def test_legitimate_npc_dialogue_passes(plan, text, lines):
    brief = SceneBrief(lines=lines)
    assert check_quotes(text, brief, plan, KNOWN) == ()
    assert check(text, plan, KNOWN) == ()


@pytest.mark.parametrize("text, kind, detail", [
    ("龚光杰指着你道：“钟灵护不了你。”", "quote_entity", "龚光杰:钟灵"),
    ("你指着龚光杰道：“钟灵护不了你。”", "puppet", "钟灵护不了你。"),
    ("你心中一动。“我这就走。”", "puppet", "我这就走。"),
    ("片刻之后你开口道：“我这就走。”", "puppet", "我这就走。"),
    ("龚光杰的师父左子穆道：“够了。”", "voice", "左子穆"),
    ("你当即打定主意，转身便走。", "puppet", "你当即打定主意"),
    ("左子穆沉声道：“够了。”", "voice", "左子穆"),
    ("他冷笑道：“左子穆也保不住你。”", "quote_entity", "?:左子穆"),
    ("龚光杰冷笑道：“你已中了毒。”", "quote_status", "龚光杰:poisoned:中了毒"),
    ("龚光杰喝道：马五德也救不了你。", "quote_entity", "龚光杰:马五德"),
])
def test_quote_gate_rejects(plan, text, kind, detail):
    found = check_quotes(text, SceneBrief(lines=(GONG, LING)), plan, KNOWN)
    assert (kind, detail) in {(v.kind, v.detail) for v in found}, found


def test_players_own_words_may_be_quoted(plan):
    brief = SceneBrief(lines=(GONG,), player_line="在下段誉，只是觉得好笑")
    for text in ("你拱手道：“在下段誉，只是觉得好笑。”", "“只是觉得好笑，”你说。"):
        assert check_quotes(text, brief, plan, KNOWN) == (), text
    assert check_quotes("你道：“我这就走。”", SceneBrief(lines=(GONG,)), plan, KNOWN, command="说我这就走") == ()


def test_sourced_inscription_is_not_anyones_line(view):
    looks = [SC.lore["d_stonedoor"]]
    plan = build_plan("duanyu", view[0], view[1], True, looks, aliases=SC.aliases)
    text = "你抬头望去，石门上刻着“琅嬛福地”四个大字。"
    assert check_quotes(text, SceneBrief(lines=(GONG,)), plan, KNOWN) == ()


def test_since_only_checks_new_quotes(plan):
    text = B_PUPPET + G2
    assert check_quotes(text, SceneBrief(lines=(GONG,)), plan, KNOWN)
    assert check_quotes(text, SceneBrief(lines=(GONG,)), plan, KNOWN, since=len(B_PUPPET)) == ()
