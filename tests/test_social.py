"""
[INPUT]: 依赖 tianlong.core 的 Intent / Op / Manner / Social / Fact / Proposition / Rel，tianlong.persistence 的 codec 与 InMemoryWorldStore，
         tianlong.runtime.authority 的 WorldAuthority，tianlong.scenarios 的 build_wuliang，tianlong.language.templates 的 render_percept
[OUTPUT]: 主持层的内核地基验收：自由言语（不带命题的 TELL/ASK）合法且只传原话与言语行为、不产生任何事实或信念；
          带姿态的等待让在场的人亲眼看见、不带姿态的等待照旧无声无息；耳语时旁人听不到原话也不知道言语行为；
          言语行为随意图与感知落库往返；模板把自由言语与姿态写成人话
[POS]: tests 的主持层地基：证伪“闲话会改变世界”“姿态会被隔墙看见”“耳语泄露了是赔罪还是威胁”
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

from __future__ import annotations

import itertools

import pytest

from tianlong.core import Fact, Intent, Manner, Modality, Op, Outcome, Proposition, Rel, Social
from tianlong.language.templates import render_percept
from tianlong.persistence import InMemoryWorldStore
from tianlong.persistence.codec import intent_from, intent_to, pevent_from, pevent_to
from tianlong.runtime.authority import WorldAuthority
from tianlong.scenarios import build_wuliang

_ids = itertools.count()


@pytest.fixture
def auth():
    return WorldAuthority.found(InMemoryWorldStore(), build_wuliang())


def _settle(auth, *specs):
    v = auth.head().version
    intents = [Intent(f"s{next(_ids)}", actor, op, target, None, manner, topic, v, utterance, social)
               for actor, op, target, manner, topic, utterance, social in specs]
    return auth.settle(intents)


def _percepts(settlement, who):
    return [o.percept for o in settlement.observations_of(who)]


def _beliefs(auth, who):
    return {(b.prop, b.holds) for b in auth.store.beliefs(auth.ref, who).sorted_beliefs()}


# ============================================================
#  自由言语：原话与言语行为传给在场的人，事实与信念纹丝不动
# ============================================================


def test_free_speech_is_legal_and_carries_only_rhetoric(auth):
    before = _beliefs(auth, "zhongling")
    s = _settle(auth, ("duanyu", Op.TELL, "zhongling", Manner.NORMAL, None, "多谢姑娘仗义执言", Social.THANK))
    ev = next(e for e in s.events if e.actor == "duanyu")
    assert ev.outcome == Outcome.SUCCESS and ev.changes == ()
    heard = [p for p in _percepts(s, "zhongling") if p.modality == Modality.SPEECH]
    assert heard and heard[0].facts == () and heard[0].event.utterance == "多谢姑娘仗义执言"
    assert heard[0].event.social == Social.THANK
    after = _beliefs(auth, "zhongling")
    assert {x for x in after - before if x[0].predicate != Rel.AT.value} == set(), "闲话不产生新的命题信念"
    bystander = [p for p in _percepts(s, "mawude") if p.modality == Modality.SPEECH]
    assert bystander and bystander[0].event.social == Social.THANK, "当众说话，旁人也听见了"


def test_free_question_is_legal(auth):
    s = _settle(auth, ("duanyu", Op.ASK, "mawude", Manner.NORMAL, None, "马五爷，这无量剑派为何要分东西二宗？",
                       Social.EXPLAIN))
    ev = next(e for e in s.events if e.actor == "duanyu")
    assert ev.outcome == Outcome.SUCCESS


def test_whisper_hides_words_and_social_act_from_bystanders(auth):
    s = _settle(auth, ("duanyu", Op.TELL, "zhongling", Manner.CAREFUL, None, "咱们快走", Social.COMMAND))
    target = [p for p in _percepts(s, "zhongling") if p.modality == Modality.SPEECH]
    assert target[0].event.utterance == "咱们快走" and target[0].event.social == Social.COMMAND
    seen = [p for p in _percepts(s, "mawude") if p.event is not None and p.event.kind == Op.TELL.value]
    assert seen and seen[0].event.utterance is None and seen[0].event.social is None


def test_speech_with_topic_still_conveys_the_claim(auth):
    claim = Fact(Proposition.rel("yijing", Rel.AT, "duanyu"), True)
    s = _settle(auth, ("duanyu", Op.TELL, "zhongling", Manner.NORMAL, claim, "我身上只有一卷易经", Social.EXPLAIN))
    heard = [p for p in _percepts(s, "zhongling") if p.modality == Modality.SPEECH]
    assert heard[0].facts == (claim,)


# ============================================================
#  姿态：带姿态的等待看得见；不带姿态的等待无声无息
# ============================================================


def test_posed_wait_is_seen_by_those_present_only(auth):
    s = _settle(auth, ("duanyu", Op.WAIT, None, Manner.NORMAL, None, "拔出长剑横在胸前", Social.THREATEN))
    mine = [p for p in _percepts(s, "duanyu") if p.event is not None and p.event.kind == Op.WAIT.value]
    assert mine and mine[0].modality == Modality.SELF
    seen = [p for p in _percepts(s, "gongguangjie") if p.event is not None and p.event.kind == Op.WAIT.value]
    assert seen and seen[0].modality == Modality.SIGHT and seen[0].facts == ()
    assert seen[0].event.utterance == "拔出长剑横在胸前" and seen[0].event.social == Social.THREATEN
    far = [p for p in _percepts(s, "sikongxuan") if p.event is not None and p.event.kind == Op.WAIT.value]
    assert far == [], "不在场的人看不见姿态"


def test_plain_wait_stays_silent(auth):
    s = _settle(auth, ("duanyu", Op.WAIT, None, Manner.NORMAL, None, None, None))
    assert not [p for p in _percepts(s, "zhongling") if p.event is not None and p.event.kind == Op.WAIT.value]


def test_templates_render_free_speech_and_gestures(auth):
    s = _settle(auth, ("duanyu", Op.TELL, "zhongling", Manner.NORMAL, None, "多谢姑娘", Social.THANK),
                ("gongguangjie", Op.WAIT, None, Manner.NORMAL, None, "冷笑一声", Social.TAUNT))
    names = auth.store.beliefs(auth.ref, "duanyu").entities
    texts = [render_percept(p, names, "duanyu", "你") for p in _percepts(s, "duanyu") if p.event is not None]
    assert any("对钟灵说：“多谢姑娘”" in t for t in texts)
    assert any("龚光杰冷笑一声" in t for t in texts)


# ============================================================
#  落库往返
# ============================================================


def test_social_round_trips_through_the_codec(auth):
    it = Intent("x", "duanyu", Op.TELL, "zhongling", None, Manner.NORMAL, None, 3, "姑娘好", Social.GREET)
    assert intent_from(intent_to(it)) == it
    s = _settle(auth, ("duanyu", Op.TELL, "zhongling", Manner.NORMAL, None, "姑娘好", Social.GREET))
    heard = [p for p in _percepts(s, "zhongling") if p.modality == Modality.SPEECH][0].event
    assert pevent_from(pevent_to(heard)) == heard
    legacy = {k: v for k, v in intent_to(it).items() if k != "social"}
    assert intent_from(legacy).social is None, "旧记录没有 social 键也能读"
