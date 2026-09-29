"""
[INPUT]: 依赖 tianlong.learning 的 parallel / datagen / samples / predictor / model / provenance / train，
         learning/rl 的 env / evaluation / imitation / module
[OUTPUT]: 放大训练的等价性验收：多进程与顺序执行逐项相同（GNN 数据、模仿示范、各策略评测与 coverage 报告，含随机基线）；
          推理快路径（一张认知图只构图、只编码一次，按块复制的批）与逐候选的原路径张量与预测逐项相同；
          训练期消融了预测的环境从不调用预测器、观测照样全零；并行度不进 run_id、不挡续训
[POS]: tests 的性能等价层：提速只许改变“怎么算”，不许改变“算出什么”——每一处提速都在这里对照原路径
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

from __future__ import annotations

from dataclasses import asdict

import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")
pytest.importorskip("ray.rllib")

from tianlong.learning.datagen import RolloutConfig, collect  # noqa: E402
from tianlong.learning.parallel import ordered_map, resolve_workers  # noqa: E402
from tianlong.learning.rl.env import TianlongEnv  # noqa: E402
from tianlong.learning.rl.evaluation import evaluate, net_policy, random_policy, scripted_policy  # noqa: E402
from tianlong.learning.rl.imitation import collect_demos  # noqa: E402
from tianlong.learning.rl.module import GraphPolicyNet  # noqa: E402


def _square(x: int) -> int:
    return x * x


def _sample_key(s) -> tuple:
    return (s.graph.x.tobytes(), s.graph.edge_index.tobytes(), s.graph.edge_attr.tobytes(), s.graph.node_ids,
            repr(s.action), s.success, s.located.tobytes(), s.holder_next.tobytes(), s.bool_next.tobytes(),
            s.num_next.tobytes(), s.discover, s.obs_gain)


def _log_key(r: dict) -> list:
    return [(lg.world, lg.returns, sorted(lg.counts.items())) for lg in r["_logs"]]


# ============================================================
#  多进程 = 顺序
# ============================================================


def test_ordered_map_keeps_input_order_and_zero_means_all_cores():
    assert ordered_map(_square, range(7), workers=3) == [x * x for x in range(7)]
    assert ordered_map(_square, [], workers=4) == []
    assert resolve_workers(0) >= 1 and resolve_workers(-3) == 1 and resolve_workers(5) == 5


def test_datagen_in_parallel_is_identical_to_sequential():
    cfg = RolloutConfig(worlds=5, steps=5, seed=11)
    a, b = collect(cfg, workers=1), collect(cfg, workers=2)
    assert a.world_of == b.world_of
    assert [_sample_key(s) for s in a.env + a.agent] == [_sample_key(s) for s in b.env + b.agent]


def test_demos_and_every_policy_evaluation_are_identical_in_parallel():
    cfg = {"task": {"jianghu": 0.5, "horizon": 8}}
    seq, par = TianlongEnv(cfg), TianlongEnv(cfg)
    da, db = collect_demos(seq, 3, 5), collect_demos(par, 3, 5, workers=2)
    assert [(d.action, d.tag, d.world, {k: v.tobytes() for k, v in d.obs.items()}) for d in da] == \
        [(d.action, d.tag, d.world, {k: v.tobytes() for k, v in d.obs.items()}) for d in db]
    torch.manual_seed(0)
    net = GraphPolicyNet(16)
    for pol in (scripted_policy, random_policy(3), net_policy(net), net_policy(net, ("predictions",))):
        ra = evaluate(seq, pol, 4, draws=20, keep_logs=True)
        rb = evaluate(par, pol, 4, draws=20, keep_logs=True, workers=2)
        assert _log_key(ra) == _log_key(rb), pol
        assert {k: v for k, v in ra.items() if k != "_logs"} == {k: v for k, v in rb.items() if k != "_logs"}
    assert seq.coverage == par.coverage, "子进程里累计的 coverage 并回父进程：报告逐字相同（含值为 0 的键）"


def test_random_baseline_is_reproducible_per_episode():
    env = TianlongEnv({"task": {"horizon": 6}})
    once = evaluate(env, random_policy(7), 3, draws=10, keep_logs=True)
    again = evaluate(env, random_policy(7), 3, draws=10, keep_logs=True)
    alone = evaluate(env, random_policy(7), 1, seed_base=900_002, draws=10, keep_logs=True)
    assert _log_key(once) == _log_key(again)
    assert _log_key(alone)[0] == _log_key(once)[2], "第三局单独评测与连着评测相同：随机流按局派生"


# ============================================================
#  推理快路径 = 原路径
# ============================================================


def test_shared_graph_inference_equals_per_candidate_path():
    from torch_geometric.data import Batch

    from tianlong.learning.model import DynamicsModel
    from tianlong.learning.predictor import GNNPredictor
    from tianlong.learning.samples import agent_queries, agent_query, shared_graph_batch, to_data
    torch.manual_seed(0)
    pred = GNNPredictor(DynamicsModel(32).eval())
    env = TianlongEnv({"task": {"jianghu": 1.0, "horizon": 6}, "predictor": pred})
    env.reset(seed=4)
    checked = 0
    for _ in range(4):
        for a in env.agents:
            sit = env.situation(a)
            store, cands = sit.beliefs, sit.candidates
            fast = shared_graph_batch(agent_queries(store, sit.now, store.owner, cands))
            slow = Batch.from_data_list([to_data(agent_query(store, sit.now, store.owner, c)) for c in cands])
            for key, v in slow:
                if isinstance(v, torch.Tensor):
                    assert torch.equal(v, getattr(fast, key)), key
            with torch.no_grad():
                ref = torch.sigmoid(pred.model(slow).success).tolist()
            got = [p.success for p in pred.predict(store, sit.now, cands, sit.profile.interests(), profile=sit.profile)]
            assert got == pytest.approx(ref, abs=1e-6)
            checked += len(cands)
        env.step(env.expert_actions())
    assert checked > 20


class _Forbidden:
    def predict(self, *a, **k):
        raise AssertionError("训练期消融了预测：不该调用预测器")


def test_prediction_ablated_env_never_calls_the_predictor():
    env = TianlongEnv({"task": {"jianghu": 1.0, "horizon": 5}, "predictor": _Forbidden(), "ablate": ["predictions"]})
    obs, _ = env.reset(seed=2)
    for _ in range(4):
        assert all(not o["cand_pred"].any() for o in obs.values())
        obs, *_ = env.step(env.expert_actions())


# ============================================================
#  并行度是资源旋钮：不进 run_id，不挡续训
# ============================================================


def test_workers_do_not_enter_run_id_or_block_resume(tmp_path):
    from tianlong.learning.provenance import run_manifest
    from tianlong.learning.train import TrainConfig, _load_state
    a = run_manifest("policy", {"seed": 0, "workers": 1}, seeds={"train": 0})
    b = run_manifest("policy", {"seed": 0, "workers": 12}, seeds={"train": 0})
    assert a["run_id"] == b["run_id"] and b["config"]["workers"] == 12
    cfg = TrainConfig(view="agent", workers=1)
    torch.save({"config": asdict(cfg)}, tmp_path / "s.pt")
    _load_state(tmp_path / "s.pt", TrainConfig(view="agent", workers=8))       # 换并行度可以续训
    with pytest.raises(ValueError, match="lr"):
        _load_state(tmp_path / "s.pt", TrainConfig(view="agent", workers=8, lr=1.0))


def test_gnn_predictor_runs_are_bitwise_identical_whatever_the_parent_thread_count(tmp_path):
    """GNN 的 CPU 运算随线程数差几个 ulp：顺序路径与子进程都在单线程下算，workers 才不改结果。"""
    from tianlong.learning.model import DynamicsModel
    from tianlong.learning.provenance import run_manifest
    from tianlong.learning.train import TrainConfig, save_checkpoint
    cfg = TrainConfig(view="agent", hidden=64)
    torch.manual_seed(1)
    save_checkpoint(DynamicsModel(64), cfg, {"manifest": run_manifest("dynamics_agent", asdict(cfg), cfg.task())},
                    tmp_path / "p.pt")
    env_cfg = {"task": {"jianghu": 1.0, "horizon": 6}, "predictor_path": str(tmp_path / "p.pt")}
    before = torch.get_num_threads()
    torch.set_num_threads(4)
    try:
        seq, par = TianlongEnv(env_cfg), TianlongEnv(env_cfg)
        da, db = collect_demos(seq, 3, 5), collect_demos(par, 3, 5, workers=2)
        assert torch.get_num_threads() == 4, "顺序路径只临时单线程，结束后恢复"
    finally:
        torch.set_num_threads(before)
    assert [{k: v.tobytes() for k, v in d.obs.items()} for d in da] == [{k: v.tobytes() for k, v in d.obs.items()} for d in db]


def test_random_policy_is_stateless_so_reuse_and_order_do_not_matter():
    env = TianlongEnv({"task": {"horizon": 6}})
    pol = random_policy(7)
    first = evaluate(env, pol, 2, seed_base=900_000, draws=10, keep_logs=True)
    overlap = evaluate(env, pol, 2, seed_base=900_001, draws=10, keep_logs=True)      # 同一个对象，世界 900001 再来一次
    twice = evaluate(env, pol, 1, seed_base=900_001, draws=10, keep_logs=True)
    assert _log_key(overlap)[0] == _log_key(first)[1] == _log_key(twice)[0]
    assert _log_key(evaluate(env, pol, 3, seed_base=899_999, draws=10, keep_logs=True, workers=3)) == \
        _log_key(evaluate(env, pol, 3, seed_base=899_999, draws=10, keep_logs=True))
