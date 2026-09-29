"""
[INPUT]: 依赖 tianlong.cognition 的 BeliefStore / candidates，tianlong.agents 的 ScriptedPolicy / Situation / HeuristicPredictor，
         tianlong.memory 的 records_for / MemoryView，tianlong.persistence 的 codec 与两种后端，tianlong.learning 的 datagen / predictor / rl.observation
[OUTPUT]: 验收 M01（长期记忆进入决策：同样的眼前认知，谁撒过谎的经历不同，选择就不同；也进入策略观测）、
          M02（被问到的问题跨越经历缓冲仍待回答，答过即勾销，说过的话不重复；两后端持久化）、
          G01（位移不等于获知：确定地走到刚看过的地方几乎没有新观察，原地翻查没翻过的地方才有；GNN 的预期获知只来自有效新观察头）、
          E04（“先探查”任务旋钮只藏目标物品且默认逐字节不变）；Prediction v2 的进展与风险来自假想分支
[POS]: tests 的记忆、承诺与预测层；证伪“短期缓冲当长期状态”“记忆检索了却没人用”“把位移当信息”三类错误
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

from __future__ import annotations

import pytest

from tianlong.agents.policies import ScriptedPolicy, Situation
from tianlong.agents.predictors import HeuristicPredictor, Prediction
from tianlong.cognition import BeliefStore, Candidate, candidates
from tianlong.core import (
    EntitySketch,
    Fact,
    Kind,
    Modality,
    Observation,
    Op,
    Outcome,
    PerceivedEvent,
    Percept,
    Proposition,
    Rel,
)
from tianlong.core.profiles import Goal, GoalKind, Profile
from tianlong.memory.records import records_for
from tianlong.memory.view import MemoryView

from .test_store_contract import found, store  # noqa: F401  （复用双后端夹具）


def sk(eid: str, kind: Kind, name: str) -> EntitySketch:
    return EntitySketch(eid, kind, name)


NAMES = (sk("hall", Kind.PLACE, "大堂"), sk("yard", Kind.PLACE, "后院"), sk("cellar", Kind.PLACE, "地窖"),
         sk("d1", Kind.DOOR, "边门"), sk("d2", Kind.DOOR, "地窖门"), sk("me", Kind.PERSON, "我"),
         sk("liar", Kind.PERSON, "张三"), sk("honest", Kind.PERSON, "李四"), sk("jade", Kind.ITEM, "玉佩"),
         sk("coin", Kind.ITEM, "铜钱"), sk("cup", Kind.ITEM, "茶杯"))


def at(a: str, b: str) -> Fact:
    return Fact(Proposition.rel(a, Rel.AT, b))


def base_store() -> BeliefStore:
    layout = Percept(0, Modality.SCENE, None, (
        Fact(Proposition.rel("d1", Rel.CONNECTS, "hall")), Fact(Proposition.rel("d1", Rel.CONNECTS, "yard")),
        Fact(Proposition.rel("d2", Rel.CONNECTS, "hall")), Fact(Proposition.rel("d2", Rel.CONNECTS, "cellar")),
        at("me", "hall"), at("liar", "hall"), at("honest", "hall")), ("hall",), NAMES)
    return BeliefStore("me", trust={"liar": 0.7, "honest": 0.6}).revise(layout)[0]


def said(t: int, who: str, fact: Fact) -> Percept:
    ev = PerceivedEvent(Op.TELL.value, "hall", who, "me", None, Outcome.SUCCESS, fact)
    return Percept(t, Modality.SPEECH, ev, (fact,), (), NAMES, who)


# ============================================================
#  M01：长期记忆进入决策
# ============================================================


def _history(liar_is: str) -> MemoryView:
    """早先的经历：某人说铜钱在后院，我亲眼一看——其实不在（或果然在）。"""
    mind = base_store().revise(said(1, liar_is, at("coin", "yard")))[0]
    look = Percept(3, Modality.SCENE, None, (at("me", "yard"),), ("yard",), NAMES)
    obs = Observation("o1", "me", None, look)
    store2, changes = mind.revise(look)
    recs = records_for("w", "b", obs, changes, store2.entities)
    assert any(r.verdict == "refuted" and r.informant == liar_is for r in recs), recs
    return MemoryView.from_records(recs)


def test_m01_memory_of_deception_changes_the_choice():
    # 眼前的认知完全相同：张三说玉佩在后院，李四说在地窖（张三略可信，两种说法并存）
    now = base_store().revise(said(10, "liar", at("jade", "yard")))[0].revise(said(11, "honest", at("jade", "cellar")))[0]
    prof = Profile("me", "寻宝", "x", (Goal(GoalKind.ACQUIRE, "jade"),))
    cands = candidates(now, prof.interests())
    preds = tuple(HeuristicPredictor().predict(now, 12, cands, prof.interests(), profile=prof))
    pol = ScriptedPolicy()

    def dest(memory: MemoryView | None) -> str | None:
        c = cands[pol.choose(Situation("me", prof, now, 12, cands, preds, memory=memory)).index]
        return c.target if c.op == Op.MOVE else None

    assert len(now.positives("jade", "AT")) == 2, "前提：矛盾的说法并存"
    assert dest(None) == "yard", "没有记忆：信略可信的张三"
    assert dest(_history("liar")) == "cellar", "张三骗过我：信李四，去地窖"
    assert dest(_history("honest")) == "yard", "李四骗过我：信张三，去后院"
    # 也进入学得的策略的观测
    pytest.importorskip("gymnasium")
    pytest.importorskip("torch_geometric")
    from tianlong.learning.rl.observation import ObsSpec, build_observation
    a = build_observation(now, 12, prof, cands, preds, ObsSpec(), _history("liar")).obs["memory"]
    b = build_observation(now, 12, prof, cands, preds, ObsSpec(), _history("honest")).obs["memory"]
    assert (a != b).any()
    assert MemoryView().reliability("liar") == 0.5 and _history("liar").reliability("liar") < 0.5


# ============================================================
#  M02：承诺跨越经历缓冲
# ============================================================


def _asked_then_noisy(noise: int = 20) -> BeliefStore:
    s = base_store().revise(Percept(1, Modality.SCENE, None, (at("cup", "hall"),), (), NAMES))[0]
    ask = PerceivedEvent(Op.ASK.value, "hall", "liar", "me", None, Outcome.SUCCESS,
                         Fact(Proposition.rel("cup", Rel.AT, None)))
    s = s.revise(Percept(2, Modality.SPEECH, ask, (), (), NAMES, "liar"))[0]
    for t in range(3, 3 + noise):            # 一串响动把经历缓冲挤满
        s = s.revise(Percept(t, Modality.SOUND, PerceivedEvent("noise", "yard"), (), (), NAMES))[0]
    return s


def test_m02_obligation_survives_the_episode_buffer_and_is_answered_once():
    s = _asked_then_noisy()
    assert not any(ep.event.kind == Op.ASK.value for ep in s.episodes), "前提：提问已被挤出经历缓冲"
    assert [o.counterpart for o in s.obligations] == ["liar"], "欠着的回答仍在"
    prof = Profile("me", "x", "x")
    cands = candidates(s, ["cup"])
    choice = ScriptedPolicy().choose(Situation("me", prof, s, 30, cands, tuple(HeuristicPredictor().predict(s, 30, cands))))
    c = cands[choice.index]
    assert c.op == Op.TELL and c.target == "liar" and c.topic == at("cup", "hall"), choice
    # 说了就勾销，并且记下“说过”——之后不再追着说
    told = PerceivedEvent(Op.TELL.value, "hall", "me", "liar", None, Outcome.SUCCESS, at("cup", "hall"))
    s2 = s.revise(Percept(31, Modality.SELF, told, (), (), NAMES))[0]
    assert not s2.obligations and s2.said and s2.said[-1].listener == "liar"
    again = ScriptedPolicy().choose(Situation("me", prof, s2, 32, candidates(s2, ["cup"]), ()))
    assert again.tag in ("idle", "goal_done") and again.index == 0


def test_m02_agenda_and_survey_memory_persist_on_both_backends(store):  # noqa: F811
    from tianlong.persistence import CommitBatch
    sc, auth = found(store)
    head = auth.head()
    guard = auth.store.beliefs(auth.ref, "guard")
    ask = PerceivedEvent(Op.ASK.value, "entrance", "player", "guard", None, Outcome.SUCCESS,
                         Fact(Proposition.rel("key", Rel.AT, None)))
    guard = guard.revise(Percept(head.clock, Modality.SPEECH, ask, (), (), (), "player"))[0]
    guard = guard.revise(Percept(head.clock, Modality.SCENE, None, (), ("entrance",), ()))[0]
    assert guard.obligations and guard.surveyed == {"entrance": head.clock}
    store.commit(CommitBatch(auth.ref, head.version, head.stamp(head.version + 1, head.clock + 1), (), (),
                             {"guard": guard}, ()))
    assert store.beliefs(auth.ref, "guard") == guard


# ============================================================
#  G01：位移 ≠ 获知
# ============================================================


def test_g01_displacement_is_not_information_gain():
    from tianlong.core import Entity, Relation, WorldState
    from tianlong.core import at as clock_at
    from tianlong.kernel import Kernel
    from tianlong.kernel.perception import scene_percept
    pytest.importorskip("numpy")
    from tianlong.learning.datagen import observation_gain, own_effect_slots

    ents = [Entity.make("hall", Kind.PLACE, "大堂"), Entity.make("yard", Kind.PLACE, "后院"),
            Entity.make("d1", Kind.DOOR, "边门"), Entity.make("me", Kind.PERSON, "我"),
            Entity.make("jade", Kind.ITEM, "玉佩", small=True, hidden=True)]
    rels = [Relation("d1", Rel.CONNECTS, "hall"), Relation("d1", Rel.CONNECTS, "yard"), Relation("me", Rel.AT, "hall"),
            Relation("jade", Rel.AT, "hall")]
    s = WorldState.build(1, clock_at(1, 9, 0), ents, rels)
    k = Kernel()
    # 先走一趟后院再回来：两处都看过了
    mind = BeliefStore("me").revise(Percept(0, Modality.SCENE, None, (
        Fact(Proposition.rel("d1", Rel.CONNECTS, "hall")), Fact(Proposition.rel("d1", Rel.CONNECTS, "yard"))),
        (), ()))[0].revise(scene_percept(s, "me"))[0]

    def gain(state, st, op, target, obj=None):
        r = k.step(state, [Candidate(op, target, obj).to_intent(f"g{state.version}", "me", state.version)])
        mine = next(e for e in r.events if e.actor == "me")
        changed = []
        for o in r.observations:
            if o.observer == "me":
                st, cs = st.revise(o.percept)
                changed += cs
        return observation_gain(changed, own_effect_slots(mine.changes)), r.state, st

    g_move, s1, st1 = gain(s, mind, Op.MOVE, "yard", "d1")
    g_back, s2, st2 = gain(s1, st1, Op.MOVE, "hall", "d1")
    g_look, _, _ = gain(s2, st2, Op.INSPECT, "hall")
    assert g_back == 0, "确定地回到刚看过的地方：没有有效新观察（自己的位置变化不算获知）"
    assert g_look >= 1, "原地仔细翻查发现藏着的玉佩：没有任何位移，却有新观察"
    # 启发式先验同样：刚看过的地方 < 没翻过的此处
    prof = Profile("me", "x", "x", (Goal(GoalKind.ACQUIRE, "jade"),))
    cands = [Candidate(Op.MOVE, "yard", "d1"), Candidate(Op.INSPECT, "hall")]
    preds = HeuristicPredictor().predict(st2, s2.clock, cands, ["jade"], profile=prof)
    assert preds[0].info_gain < preds[1].info_gain


def test_g01_gnn_info_gain_comes_from_the_observation_head_not_displacement():
    torch = pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")
    from tianlong.learning.model import DynamicsModel
    from tianlong.learning.predictor import GAIN_SCALE, GNNPredictor

    class Probe(DynamicsModel):
        def forward(self, data):
            out = super().forward(data)
            out.holder = torch.zeros_like(out.holder)
            out.holder[:, -1] = 10.0                    # 模型“认为”所有东西都变得不知去向
            out.obs_gain = torch.full_like(out.obs_gain, 0.5)
            return out

    torch.manual_seed(0)
    mind = base_store()
    cands = candidates(mind, ["jade"])
    preds = GNNPredictor(Probe(16)).predict(mind, 5, cands, ["jade"])
    assert all(abs(p.info_gain - 0.5 / GAIN_SCALE) < 1e-6 for p in preds), "预期获知只看有效新观察头"
    assert all(isinstance(p, Prediction) and 0 <= p.uncertainty <= 1 for p in preds)


def test_prediction_v2_progress_and_risk_come_from_imagined_branches():
    mind = base_store().revise(Percept(1, Modality.SCENE, None, (at("cup", "hall"),), ("hall",), NAMES))[0]
    prof = Profile("me", "x", "x", (Goal(GoalKind.ACQUIRE, "cup"),))
    take, wait = Candidate(Op.TAKE, "cup"), Candidate(Op.WAIT)
    fight = Candidate(Op.ATTACK, "liar")
    p_take, p_wait, p_fight = HeuristicPredictor().predict(mind, 2, [take, wait, fight], ["cup"], profile=prof)
    assert p_take.progress > 0.3 and p_wait.progress == 0.0, "拿到想要的东西是进展，等待不是"
    assert p_fight.risk > 0 and p_take.risk == 0.0
    assert mind.location_of("cup") == "hall", "假想分支不写回认知"


def test_e04_probe_knob_hides_only_goal_items_and_defaults_are_unchanged():
    from tianlong.scenarios.procedural import random_scenario
    for seed in range(30):
        base, probe = random_scenario(seed, jianghu=0.5), random_scenario(seed, jianghu=0.5, hide_goal_items=1.0)
        assert random_scenario(seed, jianghu=0.5, hide_goal_items=0.0).state.fingerprint() == base.state.fingerprint()
        changed = [e for e in probe.state.entities if probe.state.attr(e, "hidden") != base.state.attr(e, "hidden")]
        goal_items = {g.item for p in probe.profiles.values() for g in p.goals
                      if g.kind in (GoalKind.ACQUIRE, GoalKind.DELIVER)}
        assert set(changed) <= goal_items
