"""
[INPUT]: 依赖 tianlong.cognition 的 world_view / belief_view / candidates / BeliefStore，tianlong.learning 的 featurize / schema / samples /
         rl.observation，tianlong.kernel 的 Kernel / perception，tianlong.core 的类型，scenarios/procedural 的 random_scenario
[OUTPUT]: Schema v2 验收 F01–F08：机制变量进入环境输入而不进入无权知道的角色输入、兵刃的锋利与毒、修习的技能与进度是预测目标、
          言语行动可区分、目标人物/时间闸门/了结条件可区分、裁剪不留悬空引用、交流与观察不被物件组合挤掉；另有值域与不适用性质测试
[POS]: tests 的特征规格层；“表示丢信息”与“真实随机性”在这里被分开证伪——同一输入不能对应必败与必胜两种确定结局
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("torch_geometric")
pytest.importorskip("gymnasium")

from tianlong.agents.predictors import HeuristicPredictor  # noqa: E402
from tianlong.cognition import BeliefStore, Candidate, belief_view, candidates, world_view  # noqa: E402
from tianlong.core import (  # noqa: E402
    Entity,
    Fact,
    Kind,
    Modality,
    Op,
    Proposition,
    Rel,
    Relation,
    WorldState,
    at,
)
from tianlong.core.profiles import Goal, GoalKind, Profile  # noqa: E402
from tianlong.kernel import Kernel  # noqa: E402
from tianlong.kernel.perception import make_percept, scene_percept  # noqa: E402
from tianlong.learning.featurize import encode_action, featurize  # noqa: E402
from tianlong.learning.rl.observation import (  # noqa: E402
    GOAL_FIELDS,
    ROLES,
    ObsSpec,
    build_observation,
    observation_space,
)
from tianlong.learning.samples import env_sample  # noqa: E402
from tianlong.learning.schema import ATTR_BLOCK, DYN_BOOL, DYN_NUM  # noqa: E402

from .conftest import make_intent  # noqa: E402


def arena(a_martial: float = 0.0, edge: float = 0.3, venom: bool = False, blade_at: str = "b",
          extra_items: int = 0) -> WorldState:
    """甲、乙、丙同处演武厅；一把兵刃在乙手上（或别处）；隔壁有间库房。"""
    ents = [Entity.make("hall", Kind.PLACE, "演武厅"), Entity.make("store", Kind.PLACE, "库房"),
            Entity.make("door", Kind.DOOR, "侧门"),
            Entity.make("a", Kind.PERSON, "甲", martial=a_martial), Entity.make("b", Kind.PERSON, "乙", martial=0.5),
            Entity.make("c", Kind.PERSON, "丙", martial=0.2),
            Entity.make("blade", Kind.ITEM, "短剑", weapon=True, edge=edge, venom=True if venom else None)]
    rels = [Relation("door", Rel.CONNECTS, "hall"), Relation("door", Rel.CONNECTS, "store"),
            Relation("a", Rel.AT, "hall"), Relation("b", Rel.AT, "hall"), Relation("c", Rel.AT, "hall"),
            Relation("blade", Rel.AT, blade_at)]
    for i in range(extra_items):
        ents.append(Entity.make(f"junk{i}", Kind.ITEM, f"杂物{i}", small=True))
        rels.append(Relation(f"junk{i}", Rel.AT, "hall"))
    return WorldState.build(3, at(1, 9, 0), ents, rels)


def observer(s: WorldState, who: str) -> BeliefStore:
    """只凭一次环顾认识这个世界。"""
    layout = make_percept(s, Modality.SCENE, facts=tuple(Fact(Proposition.of(r)) for r in s.sorted_relations()
                                                         if r.type == Rel.CONNECTS))
    return BeliefStore(who).revise_all([layout, scene_percept(s, who)])[0]


def col(g, eid: str, key: str) -> tuple[float, float]:
    b = ATTR_BLOCK[key]
    row = g.x[g.index_of(eid)]
    return float(row[b.start]), float(row[b.known])


# ============================================================
#  F01 / F02：机制变量进入环境输入；角色只得到已知或自我感知的部分
# ============================================================


def test_f01_attacker_martial_distinguishes_env_input_but_not_bystander_input():
    weak, strong = arena(a_martial=0.0), arena(a_martial=1.0)
    ge, gs = featurize(world_view(weak, "a")), featurize(world_view(strong, "a"))
    assert col(ge, "a", "martial") != col(gs, "a", "martial"), "环境输入必须区分身手：同一输入不能对应必败与必胜"
    assert col(gs, "a", "martial")[1] == 1.0
    # 旁观者丙从未得知甲的内力：角色输入逐字节不变
    ce, cs = featurize(belief_view(observer(weak, "c"), 999)), featurize(belief_view(observer(strong, "c"), 999))
    assert np.array_equal(ce.x, cs.x) and np.array_equal(ce.edge_attr, cs.edge_attr)
    assert col(cs, "a", "martial") == (0.0, 0.0), "他人的内力：未知"
    # 甲自己知道自己的内力（自我感知）
    assert col(featurize(belief_view(observer(strong, "a"), 999)), "a", "martial")[1] == 1.0


def test_f02_weapon_edge_and_venom_reach_env_and_only_the_holder():
    dull, sharp = arena(edge=0.2), arena(edge=0.5, venom=True)
    gd, gs = featurize(world_view(dull)), featurize(world_view(sharp))
    assert col(gd, "blade", "edge") != col(gs, "blade", "edge") and col(gs, "blade", "venom") == (1.0, 1.0)
    # 丙只看见乙手里的短剑（外观相同）：锋利与毒是手感，丙不知道
    vd, vs = belief_view(observer(dull, "c"), 999), belief_view(observer(sharp, "c"), 999)
    assert vd == vs
    blade = next(n for n in vs.nodes if n.id == "blade")
    assert blade.value("weapon") is True and not blade.knows("edge") and not blade.knows("venom")
    # 乙握着它：手感是已知的
    held = next(n for n in belief_view(observer(sharp, "b"), 999).nodes if n.id == "blade")
    assert held.value("venom") is True and held.value("edge") == pytest.approx(0.5)


# ============================================================
#  F03：修习的技能与进度是预测目标
# ============================================================


def test_f03_study_to_mastery_changes_skill_and_progress_targets():
    ents = [Entity.make("hall", Kind.PLACE, "静室"), Entity.make("a", Kind.PERSON, "甲"),
            Entity.make("book", Kind.ITEM, "秘籍", small=True, teaches="evasion", difficulty=2)]
    s = WorldState.build(1, at(1, 9, 0), ents, [Relation("a", Rel.AT, "hall"), Relation("book", Rel.AT, "a")])
    k = Kernel()
    cand = Candidate(Op.STUDY, "book")
    s1 = k.step(s, [make_intent("a", Op.STUDY, "book", based_on=0)]).state
    r = k.step(s1, [make_intent("a", Op.STUDY, "book", based_on=1)])
    smp = env_sample(s1, r.state, "a", cand, True)
    i = smp.graph.index_of("a")
    skill = DYN_BOOL.index("evasion")
    assert smp.bool_now[i, skill] == -1 and smp.bool_next[i, skill] == 1, "学成：技能从否到是"
    prog = DYN_NUM.index("progress_evasion")
    assert smp.num_next[i, prog] > smp.num_now[i, prog] and smp.num_next_known[i, prog], "进度是数值目标，不只看位置"
    assert "evasion" in DYN_BOOL and "progress_evasion" in DYN_NUM and "martial" in DYN_NUM


# ============================================================
#  F04：言语行动可区分
# ============================================================


def test_f04_different_propositions_or_polarity_are_different_actions():
    s = arena()
    store = observer(s, "a")
    g = featurize(belief_view(store, 999))
    say_b = Candidate(Op.TELL, "c", topic=Fact(Proposition.rel("blade", Rel.AT, "b"), True))
    say_hall = Candidate(Op.TELL, "c", topic=Fact(Proposition.rel("blade", Rel.AT, "hall"), True))
    deny_b = Candidate(Op.TELL, "c", topic=Fact(Proposition.rel("blade", Rel.AT, "b"), False))
    ask = Candidate(Op.ASK, "c", topic=Fact(Proposition.rel("blade", Rel.AT, None), True))
    codes = [encode_action(g, "a", c) for c in (say_b, say_hall, deny_b, ask)]
    assert len(set(codes)) == 4, codes
    assert codes[0].topic_val != codes[1].topic_val and codes[0].topic_holds == -codes[2].topic_holds
    assert codes[3].topic_query == 1.0
    # 策略观测里同样可分
    prof = Profile("a", "x", "x")
    ob = build_observation(store, 999, prof, [Candidate(Op.WAIT), say_b, say_hall, deny_b, ask],
                           [HeuristicPredictor().predict(store, 999, [c])[0] for c in
                            (Candidate(Op.WAIT), say_b, say_hall, deny_b, ask)], ObsSpec())
    rows = {tuple(ob.obs["cand"][i]) + tuple(ob.obs["cand_flag"][i]) for i in range(5)}
    assert len(rows) == 5


# ============================================================
#  F05 / F06：目标人物、时间闸门、了结条件可区分；未激活目标不当作眼下要办的事
# ============================================================


def _goal_obs(goal: Goal, now: int = 999):
    s = arena()
    store = observer(s, "a")
    prof = Profile("a", "x", "x", (goal,))
    return build_observation(store, now, prof, [Candidate(Op.WAIT)], [HeuristicPredictor().predict(store, now,
                             [Candidate(Op.WAIT)])[0]], ObsSpec())


def test_f05_swapping_the_foe_changes_goal_encoding():
    ob_b = _goal_obs(Goal(GoalKind.HOSTILE, person="b"))
    ob_c = _goal_obs(Goal(GoalKind.HOSTILE, person="c"))
    assert not np.array_equal(ob_b.obs["goal_ptr"], ob_c.obs["goal_ptr"])
    person = ROLES.index("goal_person")
    assert not np.array_equal(ob_b.obs["role"][:, person], ob_c.obs["role"][:, person])


def test_f06_time_gate_and_until_are_encoded_and_inactive_goals_are_not_immediate():
    now = at(1, 9, 0)
    later = _goal_obs(Goal(GoalKind.HOSTILE, person="b", not_before=now + 600), now)
    soon = _goal_obs(Goal(GoalKind.HOSTILE, person="b", not_before=now + 30), now)
    active = _goal_obs(Goal(GoalKind.HOSTILE, person="b"), now)
    wounded = _goal_obs(Goal(GoalKind.HOSTILE, person="b", until="wounded"), now)
    c = {k: i for i, k in enumerate(GOAL_FIELDS)}
    assert later.obs["goal"][0, c["active"]] == 0 and soon.obs["goal"][0, c["active"]] == 0
    assert later.obs["goal"][0, c["wait"]] > soon.obs["goal"][0, c["wait"]] > 0, "距激活的时间可区分"
    assert active.obs["goal"][0, c["active"]] == 1
    assert wounded.obs["goal"][0, c["until:wounded"]] == 1 and active.obs["goal"][0, c["until:subdued"]] == 1
    person = ROLES.index("goal_person")
    assert later.obs["role"][:, person].sum() == 0, "未激活：对象不被标成眼下要办的事"
    assert active.obs["role"][:, person].sum() == 1


# ============================================================
#  F07 / F08：裁剪与候选预算
# ============================================================


def test_f07_crop_never_leaves_dangling_references():
    s = arena(extra_items=30)
    store = observer(s, "a")
    prof = Profile("a", "x", "x", (Goal(GoalKind.ACQUIRE, "blade"), Goal(GoalKind.HOSTILE, person="c")))
    cands = candidates(store, prof.interests(), 40)
    preds = HeuristicPredictor().predict(store, 999, cands, prof.interests())
    spec = ObsSpec(max_nodes=10, max_edges=24, max_cands=40)
    ob = build_observation(store, 999, prof, cands, preds, spec)
    assert ob.report.cropped and ob.report.nodes > spec.max_nodes
    o = ob.obs
    n = int(o["node_mask"].sum())
    assert n <= spec.max_nodes and o["role"][:, ROLES.index("self")].sum() == 1, "自身永远在图里"
    assert o["goal_ptr"][0, 0] >= 0 and o["goal_ptr"][1, 3] >= 0, "目标所指的实体（物品、仇人）都在图里"
    assert (o["goal_ptr"][0, 1:] == -1).all(), "目标没有的字段才是 -1"
    for i, c in enumerate(ob.candidates):
        row = o["cand"][i]
        for need, j in ((c.target, 2), (c.obj, 3)):
            assert need is None or 0 <= row[j] < n, (c, row)
        if c.topic is not None:
            assert 0 <= row[5] < n
    assert len(ob.candidates) == int(o["action_mask"].sum()) == ob.report.cands_kept
    em = o["edge_mask"] > 0
    assert (o["edge_index"][:, em] < n).all(), "边的两端都在保留的节点里"
    assert observation_space(spec).contains(o)


def test_f08_many_items_do_not_crowd_out_speech_inspection_or_waiting():
    s = arena(extra_items=25)
    store = observer(s, "a")
    full = candidates(store, ["blade"])
    few = candidates(store, ["blade"], max_count=12)
    assert len(full) > 40 and len(few) == 12
    ops = {c.op for c in few}
    assert few[0].op == Op.WAIT and {Op.TELL, Op.ASK} & ops and Op.INSPECT in ops and Op.ATTACK in ops
    assert any(c.op == Op.TAKE and c.target == "blade" for c in few) or any(c.op == Op.TELL for c in few)
    assert list(few) == [c for c in full if c in few], "入选者保持规范顺序（动作编号稳定）"


# ============================================================
#  性质：值域、不适用与未知可分
# ============================================================


def test_feature_values_stay_in_range_on_jianghu_worlds():
    from tianlong.scenarios.procedural import random_scenario
    for seed in range(12):
        sc = random_scenario(seed, jianghu=1.0)
        g = featurize(world_view(sc.state))
        assert np.abs(g.x).max() <= 1.0 + 1e-6
        for who in sc.profiles:
            store = BeliefStore(who).revise_all(sc.priors[who])[0]
            gb = featurize(belief_view(store, sc.state.clock))
            assert np.abs(gb.x).max() <= 1.0 + 1e-6
            place = gb.index_of(next(e for e, sk in store.entities.items() if sk.kind == Kind.PLACE))
            assert gb.x[place, ATTR_BLOCK["wounded"].known] == -1, "地点受没受伤：不适用"


def test_pure_id_renaming_keeps_dynamics_prediction():
    """实体 ID 只是名字：整体改名（节点顺序随之改变）后，同一行动的成败预测不变——模型没有记住 ID。"""
    import torch
    from torch_geometric.data import Batch

    from tianlong.learning.model import DynamicsModel
    from tianlong.learning.samples import agent_query, to_data

    def renamed(s: WorldState, prefix: str) -> WorldState:
        m = {e: (prefix + e) for e in s.entities}
        ents = [Entity(m[e.id], e.kind, e.name, e.attrs) for e in s.entities.values()]
        rels = [Relation(m[r.src], r.type, m[r.dst]) for r in s.relations]
        return WorldState.build(s.seed, s.clock, ents, rels)

    torch.manual_seed(0)
    model = DynamicsModel(32).eval()
    outs = []
    for prefix in ("", "zz_"):
        s = renamed(arena(extra_items=3), prefix)
        store = observer(s, prefix + "a")
        cand = Candidate(Op.ATTACK, prefix + "c")
        with torch.no_grad():
            outs.append(torch.sigmoid(model(Batch.from_data_list([to_data(agent_query(store, 999, prefix + "a",
                                                                                          cand))])).success))
    assert torch.allclose(outs[0], outs[1], atol=1e-5)
