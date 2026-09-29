"""
[INPUT]: 依赖 tianlong.learning.rl 的 evaluation / env / rewards，tianlong.learning 的 provenance / results / train，
         tianlong.cognition 的 BeliefStore，tianlong.kernel 的 Kernel，tianlong.scenarios 的 build_warehouse
[OUTPUT]: 评测口径验收 E01–E03、E05、E06：被动损失不算搜身、行为计数与奖励权重无关、以世界为单位的区间与配对比较、
          “没差异 ≠ 等效”、永远等待基线把“保持初态”与“新达成”分开、结果带可溯源 manifest 且结果表写明 run 与提交；
          动态模型的温度只在校准世界上拟合；评审回归（动手缘由、无效循环与随机重掷、严格 JSON、run_id 含未提交改动、
          跨种子汇总的分组与去重、两次运行按世界配对）
[POS]: tests 的统计口径层；证伪“指标从奖励推算”“把同一局的角色当独立样本”“把初态当学会”这三类错误
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

from __future__ import annotations

from collections import Counter

import pytest

pytest.importorskip("torch")
pytest.importorskip("ray.rllib")

from tianlong.cognition import BeliefStore  # noqa: E402
from tianlong.core import Intent, Op  # noqa: E402
from tianlong.core.profiles import Goal, GoalKind, Profile  # noqa: E402
from tianlong.kernel import Kernel  # noqa: E402
from tianlong.learning.rl.env import TianlongEnv  # noqa: E402
from tianlong.learning.rl.evaluation import (  # noqa: E402
    EpisodeLog,
    cluster_ci,
    compare,
    evaluate,
    event_counts,
    ratio,
    run_episode,
    scripted_policy,
    wait_policy,
)
from tianlong.scenarios import build_warehouse  # noqa: E402


def _stores(sc):
    return {a: BeliefStore(a).revise_all(sc.priors.get(a, ()))[0] for a in sc.profiles}


def test_e01_passive_loss_is_not_a_search():
    sc = build_warehouse()
    profiles = {"guard": Profile("guard", "x", "x", (Goal(GoalKind.PROTECT, "key", home="table"),)),
                "player": Profile("player", "x", "x", (Goal(GoalKind.ACQUIRE, "key"),))}
    r = Kernel().step(sc.state, [Intent("t", "player", Op.TAKE, "key", based_on=0)])
    c = event_counts(profiles, sc.state, _stores(sc), r.events, ())
    assert c["searches"] == 0 and c["search_miss"] == 0 and c["search_no_evidence"] == 0, "守护物被拿走不是搜身"
    # 对照：守卫去搜一个身上没有钥匙、自己也没理由怀疑的人
    k = Kernel()
    s1 = k.step(sc.state, [Intent("m", "guard", Op.MOVE, "warehouse", "door_main", based_on=0)]).state
    r2 = k.step(s1, [Intent("i", "guard", Op.INSPECT, "player", based_on=1)])
    c2 = event_counts(profiles, s1, _stores(sc), r2.events, ())
    assert (c2["searches"], c2["search_miss"], c2["search_no_evidence"]) == (1, 1, 1)


def test_e02_counts_do_not_depend_on_reward_weights():
    a = TianlongEnv({"task": {"jianghu": 1.0, "horizon": 12}})
    b = TianlongEnv({"task": {"jianghu": 1.0, "horizon": 12},
                     "reward": {"search_penalty": 5.0, "step_cost": 0.5, "fail_cost": 1.0, "shaping": 0.0}})
    for seed in range(4):
        la, lb = run_episode(a, scripted_policy, seed), run_episode(b, scripted_policy, seed)
        assert la.counts == lb.counts, "同样的事件：计数一个不差"
        if la.counts["agent_ticks"]:
            assert la.returns != lb.returns, "只有奖励分项与总回报改变"


def _log(world: int, agents: int, achieved: int, ret: float) -> EpisodeLog:
    lg = EpisodeLog(world, [f"h{i}" for i in range(agents)], {f"h{i}": ret for i in range(agents)})
    lg.counts = Counter(agents=agents, achieved=achieved)
    return lg


def test_e03_world_is_the_resampling_unit_and_comparisons_are_paired():
    # 每个世界内部角色完全同命运：按世界重采样，区间只随世界间差异变化
    same = [_log(w, 3, 3 if w % 2 else 0, 1.0) for w in range(20)]
    lo, hi = cluster_ci(same, ratio("goal_rate"))
    assert 0.0 < lo < 0.5 < hi < 1.0
    flat = [_log(w, 3, 3, 1.0) for w in range(20)]
    assert cluster_ci(flat, ratio("goal_rate")) == [1.0, 1.0]
    # 配对：同一批世界；没有共同世界直接拒绝
    other = [_log(w, 3, 3 if w % 2 else 0, 1.0) for w in range(20)]
    c = compare(same, other, "goal_rate")
    assert c["diff"] == 0 and c["verdict"] == "inconclusive", "没发现差异 ≠ 等效"
    assert compare(same, other, "goal_rate", margin=0.05)["verdict"] == "equivalent_within_margin"
    better = [_log(w, 3, 3, 1.0) for w in range(20)]
    assert compare(better, same, "goal_rate")["verdict"] == "different"
    few = compare(same[:5], other[:5], "goal_rate", margin=0.05)
    assert few["verdict"] == "insufficient_worlds", "五个世界的区间会退化，不能据此宣称等效"
    with pytest.raises(ValueError, match="同一批世界"):
        compare(same, [_log(w + 100, 3, 3, 1.0) for w in range(5)], "goal_rate")


def test_e05_wait_only_baseline_separates_initial_from_new_achievement():
    env = TianlongEnv({"task": {"jianghu": 0.5, "horizon": 10}})
    waited = evaluate(env, wait_policy, 12, draws=200)
    assert waited["initial_goal_rate"]["den"] > 0 and waited["initial_goal_rate"]["value"] > 0, "有开局即满足的目标"
    assert waited["new_goal_achievement"]["num"] == 0, "谁都不动：没有任何“新达成”"
    assert waited["goal_rate"]["value"] > 0, "永远等待也有非零达成率——这正是必须报这条基线的原因"
    acted = evaluate(env, scripted_policy, 12, draws=200)
    assert acted["new_goal_achievement"]["num"] > 0
    for k in ("goal_rate", "new_goal_achievement", "maintenance_success", "search_miss_rate", "invalid_loop_rate"):
        cell = acted[k]
        assert {"value", "ci95_world", "num", "den", "definition"} <= set(cell), "每项指标带分母、区间与定义"
    assert acted["worlds"] == 12


def test_e06_results_are_traceable():
    from tianlong.learning.provenance import run_manifest
    from tianlong.learning.results import markdown
    from tianlong.learning.task import TaskConfig

    m = run_manifest("policy", {"seed": 3}, TaskConfig(jianghu=1.0), seeds={"train": 3, "eval": [900000, 900010]})
    assert m["git_sha"] and len(m["run_id"]) == 12 and m["task_fingerprint"] == TaskConfig(jianghu=1.0).fingerprint()
    assert {"features", "schema", "attributes", "goals", "reward", "deps"} <= set(m["versions"])
    env = TianlongEnv({"task": {"horizon": 4}})
    runs = {k: evaluate(env, p, 3, draws=50) for k, p in (("wait_only", wait_policy), ("scripted", scripted_policy),
                                                             ("ppo", scripted_policy))}
    md = markdown([{"manifest": m, "policies": runs, "paired_comparisons": {}}])
    assert m["run_id"] in md and m["git_sha"][:10] in md and "(" in md


def test_dynamics_temperature_is_fit_on_calibration_worlds_only():
    from tianlong.learning.train import TrainConfig, split3, train_dynamics
    samples = list(range(100))
    world_of = [i // 5 for i in range(100)]
    tr, ca, te = split3(samples, world_of, 0.2, 0.1, 0)
    wt = {world_of[s] for s in tr}
    assert not wt & {world_of[s] for s in te} and not wt & {world_of[s] for s in ca}
    assert not {world_of[s] for s in ca} & {world_of[s] for s in te}
    _, m = train_dynamics(TrainConfig(view="agent", worlds=24, epochs=2, seed=2), log=lambda *_: None)
    assert m["split"]["worlds"]["calib"] > 0 and m["split"]["samples"]["calib"] > 0
    assert m["success_temperature_fit_on_calib"] > 0
    assert "success_brier_calibrated" in m and m["manifest"]["kind"] == "dynamics_agent"


# ============================================================
#  评审回归：动手缘由、无效循环、严格 JSON、run_id、结果汇总与跨运行配对
# ============================================================


def _advance(k, state, stores, intents):
    r = k.step(state, intents)
    for o in r.observations:
        stores[o.observer] = stores[o.observer].revise(o.percept)[0]
    return r


def test_attack_on_the_believed_holder_of_a_goal_item_is_goal_driven():
    sc = build_warehouse()
    profiles = {"guard": Profile("guard", "x", "x", (Goal(GoalKind.PROTECT, "key", home="table"),)),
                "player": Profile("player", "x", "x", (Goal(GoalKind.ACQUIRE, "key"),))}
    k, stores = Kernel(), _stores(sc)
    s1 = _advance(k, sc.state, stores, [Intent("m", "guard", Op.MOVE, "warehouse", "door_main", based_on=0)]).state
    before_take = dict(stores)
    hit = [Intent("a0", "guard", Op.ATTACK, "player", based_on=1)]
    c0 = event_counts(profiles, s1, before_take, k.step(s1, hit).events, ())
    assert c0["attack_unprovoked"] == 1, "守卫并不认为钥匙在他身上：动手就是无端"
    s2 = _advance(k, s1, stores, [Intent("t", "player", Op.TAKE, "key", based_on=1)]).state   # 守卫亲眼看着他拿走
    c = event_counts(profiles, s2, stores, k.step(s2, [Intent("a1", "guard", Op.ATTACK, "player", based_on=2)]).events, ())
    assert (c["attacks"], c["attack_goal_driven"], c["attack_unprovoked"]) == (1, 1, 0), "制住再搜是为了守护物"


def test_escape_goal_does_not_excuse_attacking_an_ally():
    sc = build_warehouse()
    k, stores = Kernel(), _stores(sc)
    s1 = _advance(k, sc.state, stores, [Intent("m", "guard", Op.MOVE, "warehouse", "door_main", based_on=0)]).state
    esc = Goal(GoalKind.ESCAPE, home="yard") if hasattr(GoalKind, "ESCAPE") else None
    hit = k.step(s1, [Intent("a", "guard", Op.ATTACK, "player", based_on=1)]).events
    ally = {"guard": Profile("guard", "x", "x", (esc,), allies=("player",)), "player": Profile("player", "x", "x")}
    foe = {"guard": Profile("guard", "x", "x", (esc,)), "player": Profile("player", "x", "x")}
    assert event_counts(ally, s1, stores, hit, ())["attack_unprovoked"] == 1
    assert event_counts(foe, s1, stores, hit, ())["attack_goal_driven"] == 1


def test_only_deterministic_failures_count_as_invalid_loops():
    from tianlong.core import Event, Outcome
    from tianlong.learning.rl.evaluation import repeat_kind

    def ev(i, op, target, outcome, reason=None, obj=None):
        return Event(f"e{i}", i, Intent(f"i{i}", "guard", op, target, obj, based_on=i), "warehouse", outcome, reason)

    parried = ev(0, Op.ATTACK, "player", Outcome.FAILURE, "parried")
    assert repeat_kind(parried, ev(1, Op.ATTACK, "player", Outcome.SUCCESS)) == "retries_stochastic"
    locked = ev(0, Op.MOVE, "yard", Outcome.FAILURE, "door_locked", obj="door_main")
    assert repeat_kind(locked, ev(1, Op.MOVE, "yard", Outcome.FAILURE, "door_locked", obj="door_main")) == "loops"
    assert repeat_kind(locked, ev(1, Op.MOVE, "yard", Outcome.FAILURE, obj="door_back")) is None, "换了路线不是重复"
    assert repeat_kind(ev(0, Op.TAKE, "key", Outcome.SUCCESS), ev(1, Op.TAKE, "key", Outcome.FAILURE)) is None
    assert repeat_kind(None, locked) is None


def test_reports_are_strict_json_when_denominators_are_zero():
    import json

    from tianlong.learning.rl.evaluation import summarize
    s = summarize([_log(w, 2, 0, 0.0) for w in range(3)], draws=50)
    assert s["search_miss_rate"]["value"] is None and s["search_miss_rate"]["ci95_world"] == [None, None]
    json.dumps(s, allow_nan=False)
    assert summarize([], draws=10)["mean_return"] is None


def test_run_id_distinguishes_uncommitted_changes(monkeypatch):
    from tianlong.learning import provenance
    ids = []
    for diff in ("aaaa", "bbbb", None):
        monkeypatch.setattr(provenance, "git_state", lambda d=diff: {"git_sha": "c" * 40, "git_dirty": d is not None,
                                                                     "git_diff_sha256": d})
        ids.append(provenance.run_manifest("policy", {"seed": 0}, seeds={"train": 0})["run_id"])
    assert len(set(ids)) == 3, "同一提交上的不同改动、以及干净工作区，各是不同的实验"


def _report(seed, value, created, sha="c" * 40, **cfg):
    """结构与 rl.train 写出的一致：summarize 的策略表 + 逐世界记录；goal_rate 的值钉成 value 便于断言。"""
    from tianlong.learning.rl.evaluation import summarize
    logs = [_log(w, 2, 2 if (w + seed) % 3 else 0, value) for w in range(24)]
    ppo = summarize(logs, draws=20)
    ppo["goal_rate"]["value"] = value
    return {"manifest": {"run_id": f"r{seed}{created}", "kind": "policy", "git_sha": sha, "git_diff_sha256": None,
                         "created_at": created, "task_fingerprint": "t", "seeds": {"train": seed},
                         "config": {"seed": seed, "predictor_path": "", "ppo_iterations": 20, "env_runners": 1, **cfg}},
            "policies": {"ppo": ppo}, "paired_comparisons": {}, "worlds": {"ppo": [lg.to_dict() for lg in logs]}}


def test_seed_summary_only_pools_runs_that_differ_by_seed():
    from tianlong.learning.results import seed_summary
    reps = [_report(0, 0.2, "2026-01-01"), _report(1, 0.4, "2026-01-01"),
            _report(0, 0.6, "2026-01-02"),                              # 同一种子重跑：取最新
            _report(0, 0.9, "2026-01-01", predictor_path="gnn.pt"),     # 换了预测器：另一组
            _report(1, 0.9, "2026-01-01", sha="d" * 40),                # 换了提交：另一组
            _report(2, 0.5, "2026-01-01", env_runners=4)]               # 只换并行度：同一组
    out = seed_summary(reps)
    assert len(out) == 3
    main = next(v for v in out.values() if len(v["train_seeds"]) == 3)
    assert main["train_seeds"] == [0, 1, 2] and main["table"]["ppo"]["goal_rate"]["mean"] == 0.5


def test_pair_reports_compares_two_runs_on_the_same_worlds(tmp_path):
    import json

    from tianlong.learning.results import main, pair_reports
    a, b = _report(0, 1.0, "x"), _report(0, 0.0, "y", ablate_predictions=True)
    c = pair_reports(a, b)
    assert c["config_diff"] == ["ablate_predictions"] and c["same_commit"]
    assert c["mean_return"]["diff"] == 1.0 and c["mean_return"]["verdict"] == "different"
    with pytest.raises(ValueError, match="逐世界"):
        pair_reports({**a, "worlds": {}}, b)
    pa, pb = tmp_path / "a.json", tmp_path / "b.json"
    pa.write_text(json.dumps(a))
    pb.write_text(json.dumps(b))
    out = tmp_path / "new" / "dir" / "RESULTS.md"                       # 目录不存在也能写
    assert main([str(pa), str(pb), "--pair", str(pa), str(pb), "--out", str(out)]) == 0
    assert "跨运行配对比较" in out.read_text() and "ablate_predictions" in out.read_text()
