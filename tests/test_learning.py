"""
[INPUT]: 依赖 tianlong.learning 的 featurize / samples / datagen / model / train / predictor，conftest 的 make_intent
[OUTPUT]: 学习层测试：特征语义（极性/方向/可信度）、批处理指针偏移、张量级认知隔离、小规模训练胜过“不变”基线、过期检查点被明确拒绝、GNN 预测器接入决策流程
[POS]: tests 的 GNN 层；验证“先隔离信息、再做消息传递”与“预测保留不确定性”
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")   # 学习层 extras 未装时整模块跳过（核心零依赖）

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from torch_geometric.data import Batch  # noqa: E402

from tianlong.cognition import BeliefStore, Candidate, belief_view, world_view  # noqa: E402
from tianlong.core import Op  # noqa: E402
from tianlong.kernel import Kernel  # noqa: E402
from tianlong.learning.featurize import F_EDGE, F_NODE, REL_VOCAB, featurize  # noqa: E402
from tianlong.learning.samples import GONE, UNKNOWN, agent_sample, env_sample, to_data  # noqa: E402
from tianlong.scenarios import build_warehouse  # noqa: E402

from .conftest import make_intent  # noqa: E402

pytestmark = pytest.mark.learn


def test_featurize_encodes_polarity_direction_and_confidence(warehouse):
    s = warehouse.state
    r = Kernel().step(s, [make_intent("player", Op.TAKE, "key", based_on=0)])
    r = Kernel().step(r.state, [make_intent("guard", Op.MOVE, "warehouse", "door_main", based_on=1)])
    store = BeliefStore("guard").revise_all(warehouse.priors["guard"])[0]
    # 守卫从未进入仓库之前与之后的认知差：负边“钥匙不在桌上”应当出现
    for o in r.observations:
        if o.observer == "guard":
            store, _ = store.revise(o.percept)
    g = featurize(belief_view(store, r.state.clock))
    assert g.x.shape[1] == F_NODE and g.edge_attr.shape[1] == F_EDGE
    k, t = g.index_of("key"), g.index_of("table")
    neg_fwd = REL_VOCAB.index(("AT", True, False))
    hit = [j for j in range(g.edge_index.shape[1]) if g.edge_index[0, j] == k and g.edge_index[1, j] == t]
    assert hit and g.edge_attr[hit[0], neg_fwd] == 1.0
    rev = [j for j in range(g.edge_index.shape[1]) if g.edge_index[0, j] == t and g.edge_index[1, j] == k]
    assert rev and g.edge_attr[rev[0], REL_VOCAB.index(("AT", True, True))] == 1.0
    assert g.edge_index.shape[1] % 2 == 0, "每条边都配一条反向边"


def test_agent_tensors_ignore_unobserved_truth():
    from tests.test_isolation import _guard_after, _perturbed
    base = featurize(belief_view(_guard_after(build_warehouse().state), 999))
    other = featurize(belief_view(_guard_after(_perturbed("door_unlocked")), 999))
    assert base.node_ids == other.node_ids
    assert np.array_equal(base.x, other.x) and np.array_equal(base.edge_attr, other.edge_attr)
    # 对照：全知视图当然会变
    assert not np.array_equal(featurize(world_view(build_warehouse().state)).x,
                              featurize(world_view(_perturbed("door_unlocked"))).x)


def test_batching_offsets_pointers(warehouse):
    s = warehouse.state
    cand = Candidate(Op.TAKE, "key")
    r = Kernel().step(s, [cand.to_intent("x", "player", 0)])
    a = to_data(env_sample(s, r.state, "player", cand, True))
    b = to_data(env_sample(s, r.state, "player", cand, True))
    batch = Batch.from_data_list([a, b])
    n = a.num_nodes
    assert int(batch.act_target[1]) == int(a.act_target[0]) + n
    assert torch.equal(batch.located[len(a.located):], a.located + n)
    key_row = int((a.located == a.act_target[0]).nonzero()[0])
    assert int(a.holder_next_idx[key_row]) == a.act_actor[0], "标签：钥匙的下一容纳者是玩家"


def test_agent_label_separates_gone_from_unknown(warehouse):
    s = warehouse.state
    guard = BeliefStore("guard").revise_all(warehouse.priors["guard"])[0]
    r = Kernel().step(s, [make_intent("player", Op.TAKE, "key", based_on=0)])
    r2 = Kernel().step(r.state, [make_intent("guard", Op.MOVE, "warehouse", "door_main", based_on=1)])
    before = guard.revise_all(o.percept for o in r.observations if o.observer == "guard")[0]
    after = before.revise_all(o.percept for o in r2.observations if o.observer == "guard")[0]
    smp = agent_sample(before, after, r.state.clock, "guard", Candidate(Op.MOVE, "warehouse", "door_main"), True)
    key_pos = list(smp.located).index(smp.graph.index_of("key"))
    assert smp.holder_now[key_pos] == smp.graph.index_of("table")
    assert smp.holder_next[key_pos] == GONE, "进门后：钥匙确知不在桌上、去向不明——不是“仍不知道”"
    player_pos = list(smp.located).index(smp.graph.index_of("captain"))
    assert smp.holder_next[player_pos] in (UNKNOWN, smp.holder_now[player_pos]), "没有负证据的就不是 GONE"


def test_tiny_training_beats_no_change_baseline():
    from tianlong.learning.train import TrainConfig, train_dynamics
    # 冒烟只验证流水线可学（底图分布）；江湖化分布的效果由全量训练报告给出（README“训练与结果”）
    _, m = train_dynamics(TrainConfig(view="env", worlds=50, epochs=6, seed=1, jianghu=0.0), log=lambda *_: None)
    assert m["holder_changed_recall"] > 0.3, m          # 基线为 0
    assert m["holder_unchanged_kept"] > 0.97, m          # 基线为 1，不能为了抓变化而乱改事实
    assert m["coverage_train"]["op:move"] > 0 and "baselines_fit_on_train" in m


def test_stale_or_wrong_view_checkpoint_is_refused_with_a_clear_error(tmp_path):
    """规格一变（属性/行动/预测目标增减），旧模型在加载时就被拒绝；全知（env）模型不能冒充角色预测器（C01）。"""
    from tianlong.learning.model import DynamicsModel
    from tianlong.learning.predictor import GNNPredictor
    from tianlong.learning.schema import SCHEMA, StaleModel

    ckpt = {"state_dict": DynamicsModel(16).state_dict(), "config": {"hidden": 16}}
    torch.save({**ckpt, "vocab": "old"}, tmp_path / "old.pt")
    with pytest.raises(StaleModel, match="重训"):
        GNNPredictor.load(tmp_path / "old.pt")
    torch.save({**ckpt, "schema": SCHEMA, "view": "env"}, tmp_path / "env.pt")
    with pytest.raises(StaleModel, match="视角"):
        GNNPredictor.load(tmp_path / "env.pt")
    torch.save({**ckpt, "schema": SCHEMA, "view": "agent"}, tmp_path / "agent.pt")
    assert GNNPredictor.load(tmp_path / "agent.pt").model is not None


def test_gnn_predictor_plugs_into_npc_pipeline():
    pytest.importorskip("langgraph")
    from tianlong.agents.npc_graph import NpcContext
    from tianlong.agents.orchestrator import Orchestrator
    from tianlong.agents.port import AgentPort
    from tianlong.learning.model import DynamicsModel
    from tianlong.learning.predictor import GNNPredictor
    from tianlong.persistence import InMemoryWorldStore
    from tianlong.runtime.authority import WorldAuthority

    sc = build_warehouse()
    auth = WorldAuthority.found(InMemoryWorldStore(), sc)
    predictor = GNNPredictor(DynamicsModel())
    s = auth.head()
    store = auth.store.beliefs(auth.ref, "guard")
    from tianlong.cognition import candidates
    cands = candidates(store, ["key"])
    preds = predictor.predict(store, s.clock, cands, ["key"])
    assert len(preds) == len(cands) and all(0 <= p.success <= 1 and 0 <= p.info_gain <= 1 for p in preds)
    ctx = {a: NpcContext(AgentPort(a, sc.profiles[a], auth.ref.world_id, auth.ref.branch_id, s.version, s.clock,
                                   beliefs=lambda a=a: auth.store.beliefs(auth.ref, a)), predictor=predictor)
           for a in sc.npcs}
    ds = Orchestrator().decide(ctx)
    assert {d.agent for d in ds} == set(sc.npcs)
