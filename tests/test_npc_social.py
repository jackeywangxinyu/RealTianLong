"""
[INPUT]: 依赖 tianlong.agents 的 ScriptedPolicy / Situation / Choice / reply_act / hush_chatter / SPEAK / CHATTER，tianlong.cognition 的
         BeliefStore / candidates / agenda（SocialCue / Obligation / Said / REPLY_TTL），tianlong.kernel 的 Kernel / scene_percept，
         tianlong.persistence 的 codec 与双后端夹具，tianlong.runtime 的 WorldAuthority / GameSession，tianlong.scenarios 的 build_wuliang
[OUTPUT]: NPC 社交层验收：当面的闲话生成社交线索与回话义务、下一 tick 按性情 × 态度 × 言语行为回话（火爆且积怨者动手）、
          不知道的结构化提问回一句“不知道”并勾销、寻仇先叫阵（嘴硬/想走/不应才动手，服软则冷静的人饶过、火爆的人照打）、
          话多的人见礼一次后按冷却说笑、两个 NPC 不会没完没了地互相回话、见义出声、态度按言语与动手/救治/赠物确定性增减并
          经编解码与两个后端往返、建档把自己人写进心里、混战里挤掉了“谁动的手”仍去搜出解药救治同伴、守卫一次闯入只动一次手；
          一切闲话的 index 都指向 WAIT（只认下标的学习层看到的是等待）；
          评审回归：候选话题之外的问题知道就照实答（free 只带自己相信的命题）且不堵后面的问题、饶过服软者不随线索缓冲滚掉、
          气已出了的仇家照样回话、每处每 tick 至多一句闲谈（Stage 与编排器同样经 hush_chatter 裁决）、见礼刷不出好感
[POS]: tests 的主持层 NPC 社交：证伪“NPC 只会回答某某在哪”“一见面就动手”“不知道就永远沉默”“把中毒的同伴丢在一边”“守卫把人点住不放”
       “知道也不答”“饶过了又平白动手”“仇报完了就装聋”“满堂人一齐开口”
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from tianlong.agents.policies import ScriptedPolicy, Situation, hush_chatter
from tianlong.agents.policy_kit import CHATTER, SPEAK, Choice
from tianlong.agents.tactics import reply_act
from tianlong.cognition import BeliefStore, Candidate, candidates
from tianlong.cognition.agenda import REPLY_TTL, Obligation, Said, SocialCue
from tianlong.core import (
    Entity,
    Fact,
    Intent,
    Kind,
    Modality,
    Op,
    Outcome,
    PerceivedEvent,
    Percept,
    Proposition,
    Rel,
    Relation,
    Social,
    WorldState,
    at,
    make_id,
)
from tianlong.core.profiles import Goal, GoalKind, Profile
from tianlong.kernel import Kernel
from tianlong.kernel.perception import scene_percept
from tianlong.persistence import codec

from .test_store_contract import found, store  # noqa: F401  （复用双后端夹具）

T0 = at(1, 9, 0)
ASK_JADE = Fact(Proposition.rel("jade", Rel.AT, None), True)


# ============================================================
#  小舞台：内核结算 + 各人按感知修正认知；NPC 由脚本策略决策，主角由测试指定（默认原地等待）
# ============================================================


def _world(*extra: Entity, rels: tuple[Relation, ...] = (), **attrs: dict) -> WorldState:
    ents = [Entity.make("hall", Kind.PLACE, "大厅"), Entity.make("yard", Kind.PLACE, "后院"),
            Entity.make("cell", Kind.PLACE, "密室"), Entity.make("d0", Kind.DOOR, "侧门"),
            Entity.make("d1", Kind.DOOR, "铁门", locked=True),
            Entity.make("hero", Kind.PERSON, "段誉", **attrs.get("hero", {})),
            Entity.make("npc", Kind.PERSON, "龚光杰", martial=0.9, **attrs.get("npc", {})),
            Entity.make("maid", Kind.PERSON, "马五德", **attrs.get("maid", {})),
            Entity.make("cup", Kind.ITEM, "茶碗"), Entity.make("jade", Kind.ITEM, "玉佩", small=True), *extra]
    base = [Relation("d0", Rel.CONNECTS, "hall"), Relation("d0", Rel.CONNECTS, "yard"),
            Relation("d1", Rel.CONNECTS, "hall"), Relation("d1", Rel.CONNECTS, "cell"),
            Relation("hero", Rel.AT, "hall"), Relation("npc", Rel.AT, "hall"), Relation("maid", Rel.AT, "hall"),
            Relation("cup", Rel.AT, "hall"), Relation("jade", Rel.AT, "yard")]
    return WorldState.build(7, T0, ents, base + list(rels))


class Stage:
    def __init__(self, state: WorldState, profiles: dict[str, Profile], player: str | None = "hero") -> None:
        self.state, self.profiles, self.player = state, profiles, player
        self.policy, self.kernel = ScriptedPolicy(), Kernel()
        self.stores = {a: BeliefStore(a, allies=p.allies).revise(replace(scene_percept(state, a), tick=T0 - 1))[0]
                       for a, p in profiles.items()}
        self.log: list[tuple[int, str, Candidate]] = []

    @property
    def now(self) -> int:
        return self.state.clock

    def situation(self, agent: str) -> Situation:
        b, prof = self.stores[agent], self.profiles[agent]
        return Situation(agent, prof, b, self.now, candidates(b, prof.interests()), (), player=self.player)

    def decide(self, agent: str) -> tuple[Choice, Candidate, Situation]:
        sit = self.situation(agent)
        choice = self.policy.choose(sit)
        return choice, choice.chosen(sit.candidates, sit.beliefs), sit

    def step(self, **acts: Candidate | tuple[Candidate, str]):
        """acts：指定角色本 tick 的行动（可附原话/姿态）；没指定的 NPC 按脚本策略，主角原地等待。
        与编排器一样，每处每 tick 至多留一句闲谈（hush_chatter），其余按捺住、原地等待。"""
        decided = {a: self.decide(a) for a, prof in sorted(self.profiles.items()) if a not in acts and not prof.is_player}
        hushed = hush_chatter(self.now, {a: (self.stores[a].location_of(a), c.tag, cand)
                                         for a, (c, cand, _) in decided.items()})
        intents = []
        for a in sorted(self.profiles):
            npc = Candidate(Op.WAIT) if a not in decided or a in hushed else decided[a][1]
            spec = acts.get(a) or npc
            cand, words = spec if isinstance(spec, tuple) else (spec, None)
            self.log.append((self.now, a, cand))
            intents.append(cand.to_intent(make_id("int", a, self.state.version), a, self.state.version, words))
        r = self.kernel.step(self.state, intents)
        for o in r.observations:
            self.stores[o.observer] = self.stores[o.observer].revise(o.percept)[0]
        self.state = r.state
        return r

    def acts_of(self, agent: str) -> list[tuple[Op, str | None, Social | None]]:
        return [(c.op, c.target, c.social) for _, a, c in self.log if a == agent]


def _cast(npc: Profile | None = None, maid: Profile | None = None) -> dict[str, Profile]:
    return {"hero": Profile("hero", "书生", "x", is_player=True),
            "npc": npc or Profile("npc", "弟子", "x"), "maid": maid or Profile("maid", "宾客", "x")}


def _say(target: str, social: Social, words: str = "……", op: Op = Op.TELL, topic: Fact | None = None):
    return Candidate(op, target, topic=topic, social=social), words


def _free(choice: Choice, sit: Situation, target: str, social: Social, tag: str = SPEAK) -> None:
    """闲话：候选之外、不带命题，index 指向 WAIT——只认下标的学习层看到的是等待。自己找话说的标 CHATTER，其余标 SPEAK。"""
    assert choice.free == Candidate(Op.TELL, target, social=social), choice
    assert sit.candidates[choice.index].op == Op.WAIT and choice.tag == tag
    assert choice.chosen(sit.candidates) == choice.free


# ============================================================
#  社交线索与回话义务
# ============================================================


def test_free_speech_leaves_a_cue_and_a_reply_obligation():
    st = Stage(_world(), _cast())
    st.step(hero=_say("npc", Social.INSULT, "你这蠢材"))
    npc = st.stores["npc"]
    assert npc.cues[-1] == SocialCue("hero", Op.TELL, Social.INSULT, T0, "你这蠢材", "npc")
    assert npc.obligations == (Obligation("reply", "hero", None, T0, Social.INSULT),)
    assert npc.attitude("hero") == -2, "当面辱骂：态度 -2"
    maid = st.stores["maid"]
    assert maid.cues[-1].to == "npc", "当众说给别人听：旁人记下线索"
    assert not maid.obligations and maid.attitude("hero") == 0, "不是冲着我：不欠回话、态度不变"
    st.step(hero=(Candidate(Op.WAIT, social=Social.SUBMIT), "扑通一声跪下"), npc=Candidate(Op.WAIT))
    gesture = st.stores["npc"].cues[-1]
    assert (gesture.op, gesture.social, gesture.to, gesture.utterance) == (Op.WAIT, Social.SUBMIT, None, "扑通一声跪下")
    assert st.stores["npc"].attitude("hero") == -2, "当众的姿态不是冲着我：态度不变"


def test_cues_are_bounded():
    st = Stage(_world(), _cast())
    for i in range(12):
        st.step(hero=_say("maid", Social.REMARK, f"第{i}句"), maid=Candidate(Op.WAIT))
    assert len(st.stores["npc"].cues) == 8 and st.stores["npc"].cues[-1].utterance == "第11句"


@pytest.mark.parametrize(("temper", "incoming", "expected"), [
    (0.0, Social.INSULT, Social.THREATEN),
    (-0.6, Social.APOLOGIZE, Social.AGREE),      # 冷静的人：赔罪就算了
    (0.0, Social.THANK, Social.AGREE),
    (0.0, Social.GREET, Social.GREET),
    (0.0, Social.REMARK, Social.REMARK),
])
def test_npc_replies_next_tick_with_the_expected_social_act(temper, incoming, expected):
    st = Stage(_world(), _cast(npc=Profile("npc", "弟子", "x", temper=temper)))
    st.step(hero=_say("npc", incoming))
    choice, _, sit = st.decide("npc")
    _free(choice, sit, "hero", expected)
    st.step()
    npc = st.stores["npc"]
    assert not npc.obligations, "回了话，义务勾销"
    assert npc.said[-1] == Said("hero", None, T0 + 1, expected)
    heard = [ep for ep in st.stores["hero"].episodes if ep.modality == Modality.SPEECH and ep.event.actor == "npc"]
    assert heard and heard[-1].event.social == expected and heard[-1].event.topic is None


def test_hot_tempered_npc_mocks_an_apology_from_someone_it_resents():
    st = Stage(_world(), _cast(npc=Profile("npc", "弟子", "x", temper=0.8)))
    st.step(hero=_say("npc", Social.INSULT))
    st.step(hero=_say("npc", Social.APOLOGIZE), npc=Candidate(Op.WAIT))
    assert st.stores["npc"].attitude("hero") == -1
    choice, _, sit = st.decide("npc")
    _free(choice, sit, "hero", Social.TAUNT)


def test_hot_temper_with_a_grudge_escalates_to_a_real_attack():
    st = Stage(_world(), _cast(npc=Profile("npc", "弟子", "x", temper=1.0)))
    st.step(hero=_say("npc", Social.INSULT))
    choice, cand, sit = st.decide("npc")
    assert choice.free is None and cand == Candidate(Op.ATTACK, "hero"), "出言不逊、脾气火爆、积怨已深：动手（普通候选）"
    st.step()
    assert not st.stores["npc"].obligations, "以拳脚作答，回话义务勾销"


def test_reply_table_is_temper_times_attitude_times_act():
    assert reply_act(Social.APOLOGIZE, temper=0.8, attitude=-1) == Social.TAUNT
    assert reply_act(Social.APOLOGIZE, temper=0.8, attitude=0) == Social.AGREE
    assert reply_act(Social.APOLOGIZE, temper=-0.5, attitude=-2) == Social.REMARK
    assert reply_act(Social.INSULT, temper=0.8) == Social.INSULT
    assert reply_act(Social.GREET, attitude=-3) == Social.REMARK
    assert reply_act(Social.COMMAND, temper=0.2, chatty=0.7) == Social.TAUNT, "嘴快的小姑娘被生人呵斥：挖苦一句，不会乖乖应承"
    assert reply_act(Social.COMMAND, temper=0.2, attitude=1) == Social.AGREE
    assert reply_act(Social.COMMAND, temper=-0.5) == Social.AGREE and reply_act(Social.COMMAND) == Social.REFUSE
    assert reply_act(None, question=True) == Social.EXPLAIN


def test_two_npcs_do_not_reply_to_each_other_forever():
    st = Stage(_world(), _cast(npc=Profile("npc", "弟子", "x", chatty=1.0)), player=None)
    for _ in range(10):
        st.step()
    talk = [(t, a, c.target, c.social) for t, a, c in st.log if c.op == Op.TELL]
    assert talk == [(T0, "npc", "hero", Social.GREET), (T0 + 6, "npc", "maid", Social.GREET),
                    (T0 + 7, "maid", "npc", Social.GREET)], "马五德回了礼，龚光杰不必再回：一来一往即止"


# ============================================================
#  结构化提问：知道就答，不知道就说不知道
# ============================================================


def test_unknown_structured_question_yields_explain_and_clears_the_obligation():
    st = Stage(_world(), _cast())
    st.step(hero=_say("npc", Social.EXPLAIN, "玉佩在哪？", Op.ASK, ASK_JADE))
    assert [o.kind for o in st.stores["npc"].obligations] == ["answer"]
    choice, _, sit = st.decide("npc")
    _free(choice, sit, "hero", Social.EXPLAIN)
    st.step()
    assert not st.stores["npc"].obligations, "说了“不知道”：不再干欠着"
    for _ in range(3):
        st.step()
    assert st.acts_of("npc").count((Op.TELL, "hero", Social.EXPLAIN)) == 1, "只说一次"


def test_known_structured_question_keeps_the_factual_answer():
    keeper = Profile("npc", "弟子", "x", (Goal(GoalKind.PROTECT, "cup", home="hall"),))   # 言语话题只谈关心的物品与人
    st = Stage(_world(), _cast(npc=keeper))
    st.step(hero=_say("npc", Social.EXPLAIN, "茶碗在哪？", Op.ASK, Fact(Proposition.rel("cup", Rel.AT, None), True)))
    choice, cand, _ = st.decide("npc")
    assert choice.free is None and cand.op == Op.TELL and cand.topic == Fact(Proposition.rel("cup", Rel.AT, "hall"))


# ============================================================
#  先礼后兵
# ============================================================


def _foe(temper: float = 0.0) -> Profile:
    return Profile("npc", "弟子", "x", (Goal(GoalKind.HOSTILE, person="hero", until="wounded"),), temper=temper)


def test_hostile_npc_challenges_first_and_strikes_only_when_ignored():
    st = Stage(_world(), _cast(npc=_foe()))
    choice, _, sit = st.decide("npc")
    _free(choice, sit, "hero", Social.CHALLENGE)
    for _ in range(3):
        st.step()
    assert st.acts_of("npc") == [(Op.TELL, "hero", Social.CHALLENGE), (Op.TELL, "hero", Social.TAUNT),
                                 (Op.ATTACK, "hero", None)], "叫阵 → 再激一激 → 不应才动手"


@pytest.mark.parametrize("answer", [Social.REFUSE, Social.INSULT])
def test_hostile_npc_strikes_after_a_defiant_answer(answer):
    st = Stage(_world(), _cast(npc=_foe()))
    st.step(hero=_say("npc", answer, "偏不"))
    st.step()
    assert st.acts_of("npc") == [(Op.TELL, "hero", Social.CHALLENGE), (Op.ATTACK, "hero", None)], "嘴硬：立刻动手"


def test_hostile_npc_strikes_when_the_target_tries_to_leave():
    st = Stage(_world(), _cast(npc=_foe()))
    st.step(hero=Candidate(Op.MOVE, "cell", "d1"))           # 铁门锁着：走不成，但想走被看见了
    st.step()
    assert st.acts_of("npc")[-1] == (Op.ATTACK, "hero", None)


def test_submissive_target_is_spared_by_a_calm_npc():
    st = Stage(_world(), _cast(npc=_foe(temper=-0.5)))
    st.step(hero=(Candidate(Op.WAIT, social=Social.SUBMIT), "跪地求饶"))
    for _ in range(8):
        st.step()
    acts = st.acts_of("npc")
    assert acts[:2] == [(Op.TELL, "hero", Social.CHALLENGE), (Op.TELL, "hero", Social.TAUNT)], "挖苦一句"
    assert (Op.ATTACK, "hero", None) not in acts, "服软了：冷静的人且饶他"
    assert not st.state.attr("hero", "wounded")


def test_hot_tempered_npc_may_strike_a_submissive_target_anyway():
    st = Stage(_world(), _cast(npc=_foe(temper=1.0)))
    st.step(hero=_say("npc", Social.APOLOGIZE, "在下知错"))
    st.step()
    assert st.acts_of("npc")[-1] == (Op.ATTACK, "hero", None)


# ============================================================
#  闲谈与见义出声
# ============================================================


def test_chatty_npc_greets_once_then_remarks_with_a_cooldown():
    st = Stage(_world(), _cast(npc=Profile("npc", "弟子", "x", chatty=1.0), maid=Profile("maid", "宾客", "x")))
    choice, _, sit = st.decide("npc")
    _free(choice, sit, "hero", Social.GREET, CHATTER)
    for _ in range(13):
        st.step()
    talk = [(t, c.social) for t, a, c in st.log if a == "npc" and c.op == Op.TELL]
    assert talk[0] == (T0, Social.GREET) and len(talk) == 3, talk
    assert [t for t, _ in talk] == [T0, T0 + 6, T0 + 12], "冷却 6 个 tick"
    assert all(s in (Social.REMARK, Social.JOKE) for _, s in talk[1:]), "见礼只一次"
    assert all(c.target == "hero" for t, a, c in st.log if a == "npc" and c.op == Op.TELL), "只找主角说话"
    assert not [c for _, a, c in st.log if a == "maid" and c.op == Op.TELL], "不话多的人不找话说"


def test_chatter_gate_is_deterministic_and_never_for_the_player():
    runs = []
    for _ in range(2):
        st = Stage(_world(), _cast(npc=Profile("npc", "弟子", "x", chatty=0.4)))
        for _ in range(30):
            st.step()
        runs.append(st.acts_of("npc"))
    assert runs[0] == runs[1] and runs[0].count((Op.TELL, "hero", Social.GREET)) == 1
    st = Stage(_world(), {**_cast(), "hero": Profile("hero", "书生", "x", is_player=True, chatty=1.0)})
    sit = st.situation("hero")
    assert ScriptedPolicy().choose(sit).free is None, "主角从不自己找话说"


def test_without_a_known_player_chatter_goes_round_the_room():
    st = Stage(_world(), _cast(npc=Profile("npc", "弟子", "x", chatty=1.0)), player=None)
    for _ in range(7):
        st.step(maid=Candidate(Op.WAIT))
    talk = [(t, c.target, c.social) for t, a, c in st.log if a == "npc" and c.op == Op.TELL]
    assert talk == [(T0, "hero", Social.GREET), (T0 + 6, "maid", Social.GREET)]


def test_witness_shouts_only_for_a_player_they_are_fond_of():
    st = Stage(_world(), _cast())
    st.step(npc=Candidate(Op.ATTACK, "hero"))
    choice, _, sit = st.decide("maid")
    assert choice.free is None, "素不相识的书生挨打，满堂宾客不会人人替他出头"
    st = Stage(_world(), _cast())
    st.step(hero=Candidate(Op.TELL, "maid", social=Social.GREET))       # 先见过礼：心里有了好感
    st.step(npc=Candidate(Op.ATTACK, "hero"))
    choice, _, sit = st.decide("maid")
    _free(choice, sit, "npc", Social.COMMAND)


# ============================================================
#  态度：只在自己心里，确定性增减，随认知落库
# ============================================================


def _attitude_stage() -> Stage:
    extra = (Entity.make("pal", Kind.PERSON, "干光豪"), Entity.make("antidote", Kind.ITEM, "解药", cures="poisoned"),
             Entity.make("cake", Kind.ITEM, "糕点"))
    state = _world(*extra, rels=(Relation("pal", Rel.AT, "hall"), Relation("antidote", Rel.AT, "hero"),
                                 Relation("cake", Rel.AT, "hero")), npc={"poisoned": True})
    cast = {**_cast(npc=Profile("npc", "弟子", "x", allies=("pal",))), "pal": Profile("pal", "弟子", "x")}
    return Stage(state, cast)


def test_attitudes_follow_words_blows_and_kindness():
    st = _attitude_stage()
    hold = {"npc": Candidate(Op.WAIT), "pal": Candidate(Op.WAIT), "maid": Candidate(Op.WAIT)}
    mind = lambda: st.stores["npc"].attitude("hero")  # noqa: E731
    st.step(hero=Candidate(Op.ATTACK, "pal"), **hold)
    assert mind() == -1, "打我的自己人：-1"
    st.step(hero=Candidate(Op.ATTACK, "npc"), **hold)
    assert mind() == -3, "打我：-2"
    st.step(hero=_say("npc", Social.INSULT), **hold)
    assert mind() == -3, "截在 -3"
    st.step(hero=Candidate(Op.USE, "npc", "antidote"), **hold)
    assert mind() == -1 and not st.state.attr("npc", "poisoned"), "救治我：+2"
    st.step(hero=Candidate(Op.GIVE, "npc", "cake"), **hold)
    assert mind() == 1, "送我东西：+2"
    st.step(hero=_say("npc", Social.THANK), **hold)
    assert mind() == 2
    assert st.stores["pal"].attitude("hero") == -2 and st.stores["maid"].attitude("hero") == 0, "态度只在各自心里"


def test_social_state_round_trips_through_the_codec():
    st = _attitude_stage()
    st.step(hero=_say("npc", Social.INSULT, "蠢材"))
    st.step(hero=(Candidate(Op.WAIT, social=Social.SUBMIT), "磕头"))
    npc = st.stores["npc"]
    assert npc.cues and npc.said and npc.attitudes and npc.company and npc.allies == ("pal",)
    assert npc.yielded == {"hero": T0 + 1}
    wire = lambda x: json.loads(json.dumps(x, ensure_ascii=False))  # noqa: E731
    assert all(codec.cue_from(wire(codec.cue_to(c))) == c for c in npc.cues)
    assert all(codec.said_from(wire(codec.said_to(s))) == s for s in npc.said)
    reply = Obligation("reply", "hero", None, 3, Social.INSULT)
    assert codec.obligation_from(wire(codec.obligation_to(reply))) == reply
    assert codec.attitudes_from(wire(codec.attitudes_to(npc.attitudes))) == dict(npc.attitudes)
    # 旧记录：没有 social 键、没有态度
    old_ob = {"kind": "answer", "counterpart": "hero", "topic": codec.fact_to(ASK_JADE), "since": 1}
    assert codec.obligation_from(old_ob) == Obligation("answer", "hero", ASK_JADE, 1)
    old_said = {"listener": "hero", "fact": codec.fact_to(ASK_JADE), "tick": 2}
    assert codec.said_from(old_said) == Said("hero", ASK_JADE, 2)
    assert codec.attitudes_from(None) == {}


def test_social_state_persists_on_both_backends(store):  # noqa: F811
    from tianlong.persistence import CommitBatch
    _, auth = found(store)
    head = auth.head()
    guard = replace(auth.store.beliefs(auth.ref, "guard"), allies=("captain",))
    insult = PerceivedEvent(Op.TELL.value, "entrance", "player", "guard", None, Outcome.SUCCESS,
                            utterance="看门狗", social=Social.INSULT)
    kneel = PerceivedEvent(Op.WAIT.value, "entrance", "player", None, None, Outcome.SUCCESS,
                           utterance="作揖", social=Social.SUBMIT)
    look = scene_percept(head, "guard")
    guard = guard.revise_all([Percept(head.clock, Modality.SPEECH, insult, (), (), (), "player"),
                              Percept(head.clock, Modality.SIGHT, kneel), look])[0]
    assert guard.cues and guard.attitude("player") == -2 and guard.obligations[0].kind == "reply"
    assert guard.yielded == {"player": head.clock}, "当众作揖服软：记进 yielded"
    store.commit(CommitBatch(auth.ref, head.version, head.stamp(head.version + 1, head.clock + 1), (), (),
                             {"guard": guard}, ()))
    assert store.beliefs(auth.ref, "guard") == guard


def test_founding_writes_allies_into_each_mind():
    from tianlong.persistence import InMemoryWorldStore
    from tianlong.runtime.authority import WorldAuthority
    from tianlong.scenarios import build_wuliang
    sc = build_wuliang()
    auth = WorldAuthority.found(InMemoryWorldStore(), sc)
    for a, p in sc.profiles.items():
        mind = auth.store.beliefs(auth.ref, a)
        assert mind.allies == tuple(sorted(p.allies)) and not mind.attitudes
    assert "gongguangjie" in auth.store.beliefs(auth.ref, "zuozimu").company, "开场就知道谁在眼前"


# ============================================================
#  摸底时发现的行为问题
# ============================================================


def test_poisoned_ally_is_healed_even_after_the_attack_scrolled_out_of_memory():
    extra = (Entity.make("patient", Kind.PERSON, "龚光杰", poisoned=True),
             Entity.make("culprit", Kind.PERSON, "钟灵", subdued_until=T0 + 30),
             Entity.make("antidote", Kind.ITEM, "解药", small=True, cures="poisoned"))
    state = _world(*extra, rels=(Relation("patient", Rel.AT, "hall"), Relation("culprit", Rel.AT, "hall"),
                                 Relation("antidote", Rel.AT, "culprit")))
    cast = {**_cast(npc=Profile("npc", "掌门", "x", allies=("patient",))),
            "patient": Profile("patient", "弟子", "x", allies=("npc",)), "culprit": Profile("culprit", "少女", "x")}
    st = Stage(state, cast)
    bite = PerceivedEvent(Op.ATTACK.value, "hall", "culprit", "patient", None, Outcome.SUCCESS)
    noise = [Percept(T0 - 1, Modality.SOUND, PerceivedEvent("noise", "yard")) for _ in range(13)]
    st.stores["npc"] = st.stores["npc"].revise_all([Percept(T0 - 2, Modality.SIGHT, bite), *noise])[0]
    assert not any(ep.event.kind == Op.ATTACK.value for ep in st.stores["npc"].episodes), "前提：谁动的手已挤出经历缓冲"
    for _ in range(4):
        st.step()
    acts = st.acts_of("npc")
    assert (Op.INSPECT, "culprit", None) in acts and (Op.TAKE, "antidote", None) in acts
    assert (Op.USE, "patient", None) in acts and not st.state.attr("patient", "poisoned"), acts


def _guard_post(**hero: object) -> Stage:
    cast = {"hero": Profile("hero", "书生", "x", is_player=True),
            "npc": Profile("npc", "帮众", "x", (Goal(GoalKind.GUARD, home="hall"),)),
            "maid": Profile("maid", "帮众", "x", allies=("npc",))}
    cast["npc"] = replace(cast["npc"], allies=("maid",))
    return Stage(_world(hero=hero), cast)


def test_guard_strikes_once_per_intrusion_and_never_stun_locks():
    st = _guard_post()
    for _ in range(45):
        st.step()
        assert not st.state.attr("hero", "subdued_until") or st.state.attr("hero", "subdued_until") <= st.now
    acts = st.acts_of("npc")
    assert acts.count((Op.ATTACK, "hero", None)) == 1, "一次闯入只动一次手"
    assert st.state.attr("hero", "wounded") and (Op.TELL, "hero", Social.COMMAND) in acts, "之后喝令离开"
    st.step(hero=_say("npc", Social.REFUSE, "偏不走"))
    st.step()
    assert st.acts_of("npc")[-1] == (Op.ATTACK, "hero", None), "顶撞喝令即是硬闯"


def test_guard_does_not_strike_a_wounded_intruder_who_stays_put():
    st = _guard_post(wounded=True)
    for _ in range(10):
        st.step()
    acts = st.acts_of("npc")
    assert (Op.ATTACK, "hero", None) not in acts and acts[0] == (Op.TELL, "hero", Social.COMMAND)


def test_free_speech_is_only_rhetoric_for_index_consumers():
    """学习层只认下标：闲话的 index 是 WAIT，照 index 执行就是原地等待，不产生任何言语事件。"""
    st = Stage(_world(), _cast(npc=_foe()))
    choice, _, sit = st.decide("npc")
    assert choice.free is not None and sit.candidates[choice.index] == Candidate(Op.WAIT)
    it = sit.candidates[choice.index].to_intent("i", "npc", st.state.version)
    r = st.kernel.step(st.state, [it])
    assert all(e.op == Op.WAIT for e in r.events) and isinstance(it, Intent)


def test_decision_graph_passes_the_player_and_checkpoints_the_social_act():
    pytest.importorskip("langgraph")
    from tianlong.agents.npc_graph import NpcContext
    from tianlong.agents.orchestrator import Orchestrator
    from tianlong.agents.port import AgentPort
    from tianlong.persistence import InMemoryWorldStore
    from tianlong.runtime.authority import WorldAuthority
    from tianlong.scenarios.base import Scenario
    state = _world()
    cast = _cast(npc=Profile("npc", "弟子", "x", chatty=1.0))
    sc = Scenario("social", state, cast, {a: (replace(scene_percept(state, a), tick=T0 - 1),) for a in cast})
    auth = WorldAuthority.found(InMemoryWorldStore(), sc)
    port = AgentPort("npc", cast["npc"], auth.ref.world_id, auth.ref.branch_id, 0, T0,
                     beliefs=lambda: auth.store.beliefs(auth.ref, "npc"))
    orch = Orchestrator(checkpoint=True)
    [d] = orch.decide({"npc": NpcContext(port, player="hero")})
    assert (d.intent.op, d.intent.target, d.intent.topic, d.intent.social) == (Op.TELL, "hero", None, Social.GREET)
    snap = orch.npc_graph.get_state({"configurable": {"thread_id": Orchestrator.thread_id(port)}})
    assert snap.values["chosen"].social is Social.GREET, "检查点读回来的言语行为仍是 Social，而不是被拦下的字符串"
    assert auth.settle([d.intent]).events[0].outcome == Outcome.SUCCESS


# ============================================================
#  评审回归：答话不在候选里、服软被挤出线索、气已出了就不理人、闲谈扎堆、见礼刷好感
# ============================================================

ASK_CUP = Fact(Proposition.rel("cup", Rel.AT, None), True)
CUP_IN_HALL = Fact(Proposition.rel("cup", Rel.AT, "hall"), True)


def test_known_answer_outside_the_candidate_topics_is_told_and_later_questions_still_get_answers():
    """没有目标的人，候选话题里没有茶碗：知道就照实答（free 带上自己相信的命题，index 仍指向 WAIT），也不堵住后面的问题。"""
    st = Stage(_world(), _cast())
    st.step(hero=_say("npc", Social.EXPLAIN, "茶碗在哪？", Op.ASK, ASK_CUP))
    choice, cand, sit = st.decide("npc")
    assert not any(c.op == Op.TELL and c.topic == CUP_IN_HALL for c in sit.candidates), "前提：这句答话不在候选里"
    assert cand == Candidate(Op.TELL, "hero", topic=CUP_IN_HALL) and choice.free == cand and choice.tag == SPEAK
    assert sit.candidates[choice.index].op == Op.WAIT, "只认下标的学习层照旧看到等待"
    st.step(hero=_say("npc", Social.EXPLAIN, "马五德在哪？", Op.ASK, Fact(Proposition.rel("maid", Rel.AT, None), True)))
    st.step()
    npc = st.stores["npc"]
    assert not npc.obligations, "两个问题都答了"
    told = {s.fact for s in npc.said if s.listener == "hero"}
    assert {CUP_IN_HALL, Fact(Proposition.rel("maid", Rel.AT, "hall"), True)} <= told
    heard = [ep.event.topic for ep in st.stores["hero"].episodes if ep.modality == Modality.SPEECH]
    assert CUP_IN_HALL in heard, "答话是真正说出口的言语：问话的人听到了这个说法"


def test_free_answer_is_tried_only_while_the_question_is_fresh():
    """只认下标的消费者把 free 答话当作等待（话没说出口，问题也没勾销）：只在 REPLY_TTL 内尝试，示范者不会一直干等。"""
    st = Stage(_world(), _cast())
    st.step(hero=_say("npc", Social.EXPLAIN, "茶碗在哪？", Op.ASK, ASK_CUP))
    tries = 0
    for _ in range(REPLY_TTL + 3):
        choice, _, sit = st.decide("npc")
        tries += choice.free is not None
        st.step(npc=sit.candidates[choice.index])
    assert tries == REPLY_TTL


def test_free_line_may_carry_only_a_claim_the_speaker_believes():
    sit = Stage(_world(), _cast()).situation("npc")
    b = sit.beliefs
    ok = Choice(0, "如实相告", SPEAK, free=Candidate(Op.TELL, "hero", topic=CUP_IN_HALL))
    assert ok.chosen(sit.candidates, b) == ok.free
    for bad in (Candidate(Op.TELL, "hero", topic=Fact(Proposition.rel("cup", Rel.AT, "yard"), True)),   # 自己不信的
                Candidate(Op.TELL, "hero", topic=Fact(CUP_IN_HALL.prop, False)),                       # 否定的说法
                Candidate(Op.TELL, "hero", topic=ASK_CUP),                                             # 没有答案
                Candidate(Op.ASK, "hero", topic=ASK_JADE)):                                            # 问话永不带命题
        with pytest.raises(ValueError):
            Choice(0, "", SPEAK, free=bad).chosen(sit.candidates, b)
    with pytest.raises(ValueError):
        ok.chosen(sit.candidates)        # 没有说话者的认知可对照：不放行


@pytest.mark.parametrize("flush", ["bystander_talk", "public_gestures"])
def test_a_spared_target_stays_spared_when_the_cue_buffer_rolls_over(flush):
    """饶过服软的人是记在心里的事：闲谈（别人之间）或他自己的一串姿态把线索缓冲挤满，也不会平白动手。"""
    st = Stage(_world(), _cast(npc=_foe(temper=-0.5)))
    st.step(hero=(Candidate(Op.WAIT, social=Social.SUBMIT), "跪地求饶"))
    for _ in range(3):
        st.step()
    assert (Op.TELL, "hero", Social.TAUNT) in st.acts_of("npc"), "服了软：挖苦一句便罢手"
    for i in range(6):
        st.step(hero=_say("maid", Social.PRAISE, f"第{i}句") if flush == "bystander_talk"
                else (Candidate(Op.WAIT), f"端起茶碗喝了第{i}口"))
        st.step(hero=(Candidate(Op.WAIT), f"又喝了第{i}口") if flush == "public_gestures" else Candidate(Op.WAIT))
    assert flush == "bystander_talk" or not any(c.social == Social.SUBMIT for c in st.stores["npc"].cues), \
        "前提：服软的那条线索已被挤出缓冲"
    assert (Op.ATTACK, "hero", None) not in st.acts_of("npc") and not st.state.attr("hero", "wounded")
    assert st.stores["npc"].yielded == {"hero": T0}, "谁在何时服过软：按人记下，不随线索缓冲滚掉"


def test_bystander_talk_is_evicted_before_cues_aimed_at_me():
    st = Stage(_world(), _cast())
    st.step(hero=_say("npc", Social.APOLOGIZE, "得罪了"), npc=Candidate(Op.WAIT))
    for i in range(10):
        st.step(hero=_say("maid", Social.REMARK, f"第{i}句"), npc=Candidate(Op.WAIT), maid=Candidate(Op.WAIT))
    cues = st.stores["npc"].cues
    assert len(cues) == 8 and cues[0].social == Social.APOLOGIZE and cues[-1].utterance == "第9句"


def test_foe_whose_grudge_is_settled_still_answers():
    """寻仇已了（仇人已受伤）：先礼后兵不再接话，回话照旧——被骂回嘴，被骂急了的火爆脾气照样动手。"""
    st = Stage(_world(hero={"wounded": True}), _cast(npc=_foe()))
    st.step(hero=_say("npc", Social.INSULT, "蠢材"))
    choice, _, sit = st.decide("npc")
    _free(choice, sit, "hero", Social.THREATEN)
    st.step()
    assert not st.stores["npc"].obligations, "回了话，义务勾销"
    st.step(hero=_say("npc", Social.GREET, "兄台好"))
    choice, _, sit = st.decide("npc")
    _free(choice, sit, "hero", Social.REMARK)       # 仇家的见礼：不冷不热地应一句，不回礼
    st = Stage(_world(hero={"wounded": True}), _cast(npc=_foe(temper=1.0)))
    st.step(hero=_say("npc", Social.INSULT, "蠢材"))
    assert st.stores["npc"].attitude("hero") <= -2
    _, cand, _ = st.decide("npc")
    assert cand == Candidate(Op.ATTACK, "hero"), "火爆且积怨已深：被骂即动手"


def test_chatty_npcs_take_turns_instead_of_talking_at_once():
    """两个话多的人同处一室：每 tick 至多一句闲谈（确定性地选一人），别人刚开过口也不接着插嘴——但两人都轮得到。"""
    cast = _cast(npc=Profile("npc", "弟子", "x", chatty=1.0), maid=Profile("maid", "宾客", "x", chatty=1.0))
    runs = []
    for _ in range(2):
        st = Stage(_world(), cast)
        for _ in range(15):
            st.step()
        runs.append([(t, a, c.social) for t, a, c in st.log if a != "hero" and c.op == Op.TELL])
    talk = runs[0]
    assert runs[0] == runs[1], "闸门确定"
    ticks = [t for t, _, _ in talk]
    assert len(ticks) == len(set(ticks)), f"每 tick 至多一句：{talk}"
    assert all(b - a > 1 for a, b in zip(ticks, ticks[1:], strict=False)), f"不一串接一串：{talk}"
    assert {a for _, a, _ in talk} == {"npc", "maid"} and [s for _, _, s in talk].count(Social.GREET) == 2


def test_chatter_yields_the_floor_to_a_prompted_line():
    st = Stage(_world(), _cast(npc=_foe(), maid=Profile("maid", "宾客", "x", chatty=1.0)))
    st.step()
    assert st.acts_of("npc") == [(Op.TELL, "hero", Social.CHALLENGE)]
    assert st.acts_of("maid") == [(Op.WAIT, None, None)], "有人叫阵：旁人不在这时找话寒暄"


def test_hush_chatter_keeps_one_line_per_place_regardless_of_order():
    chat = lambda who: Candidate(Op.TELL, who, social=Social.REMARK)  # noqa: E731
    decided = {"a": ("hall", CHATTER, chat("p")), "b": ("hall", CHATTER, chat("p")), "c": ("hall", CHATTER, chat("p")),
               "d": ("yard", CHATTER, chat("p")), "e": ("yard", "", Candidate(Op.WAIT))}
    hushed = hush_chatter(T0, decided)
    assert len(hushed & {"a", "b", "c"}) == 2 and "d" not in hushed and "e" not in hushed
    assert hush_chatter(T0, dict(reversed(list(decided.items())))) == hushed, "与扇出顺序无关"
    decided["e"] = ("yard", SPEAK, Candidate(Op.TELL, "p", social=Social.CHALLENGE))
    assert "d" in hush_chatter(T0, decided), "同处有人正经开口：闲谈让出话头"


def test_repeated_greetings_get_small_talk_and_do_not_farm_goodwill():
    st = Stage(_world(), _cast())
    for _ in range(3):
        st.step(hero=_say("npc", Social.GREET, "兄台好"))
        st.step()
    replies = [s for op, _, s in st.acts_of("npc") if op == Op.TELL]
    assert replies == [Social.GREET, Social.REMARK, Social.REMARK], "见礼只回一次，此后寒暄"
    assert st.stores["npc"].attitude("hero") == 1, "见礼只把生分暖成点头之交"
    st = Stage(_world(), _cast())
    st.step(hero=_say("npc", Social.INSULT), npc=Candidate(Op.WAIT))
    for _ in range(2):
        st.step(hero=_say("npc", Social.GREET), npc=Candidate(Op.WAIT))
    assert st.stores["npc"].attitude("hero") == -2, "见礼抹不掉积怨：要赔罪才行"


def test_wuliang_opening_has_at_most_one_spontaneous_remark_per_tick():
    """实测：开场第二个 tick 左子穆、马五德一齐见礼，龚光杰同时讥讽——三句话挤在一起。"""
    pytest.importorskip("langgraph")
    pytest.importorskip("qdrant_client")
    from tianlong.runtime.session import GameSession
    from tianlong.scenarios import build_wuliang
    chat = {Social.GREET, Social.REMARK, Social.JOKE}
    for seed in (1, 7):
        s = GameSession(build_wuliang(seed))
        s.intro()
        events = [e for _ in range(12) for e in s.turn("等待").events]
        lines = [e for e in events if e.actor != "duanyu" and e.op in (Op.TELL, Op.ASK)]
        for tick, place in sorted({(e.tick, e.place or "") for e in lines}):
            now = [e for e in lines if e.tick == tick and (e.place or "") == place]
            said = [(e.actor, e.intent.social) for e in now]
            small = [e for e in now if e.intent.target == "duanyu" and e.intent.topic is None and e.intent.social in chat]
            assert len(small) <= 1, (seed, tick, said)
            assert not small or len(now) == 1, f"有人正经开口时不寒暄：{seed, tick, said}"
