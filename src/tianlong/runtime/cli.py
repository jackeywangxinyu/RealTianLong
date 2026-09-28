"""
[INPUT]: 依赖 runtime/session 的 GameSession / TurnReport，scenarios 的 build_warehouse，language/llm 的 llm_from_env，
         language/templates 的 render_fact，core 的 Fact；按需加载 persistence/neo4j_store、learning/predictor、learning/rl/policy
[OUTPUT]: 对外提供 main()（命令行入口 `tianlong` / `python -m tianlong`）
[POS]: runtime 的终端前端；/debug 显示真相与 NPC 理由（开发者视角），/beliefs 显示玩家自己的认知——两者刻意分开；
       --store/--save 选择持久化与存档，--predictor/--policy 让训练好的 GNN 与 RL 策略驱动 NPC
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import replace

from tianlong.core import Fact
from tianlong.language.llm import llm_from_env
from tianlong.language.templates import render_fact
from tianlong.runtime.session import GameSession, TurnReport
from tianlong.scenarios import build_warehouse

_HELP = "指令示例：拿走桌上的钥匙 / 去仓库入口 / 用钥匙打开仓库门 / 问守卫钥匙在哪 / 等待\n" \
        "元指令：/beliefs 查看你的认知  /debug 切换开发者视角  /quit 退出"


def _debug_lines(r: TurnReport) -> list[str]:
    out = ["  ── 真相 ──"]
    for e in r.events:
        if e.op.value != "wait":
            out.append(f"  {e.actor} {e.op.value} {e.intent.target or ''} {e.intent.obj or ''} → {e.outcome.value}"
                       f"{' (' + e.reason + ')' if e.reason else ''}")
    for d in r.deliberations:
        out.append(f"  [{d.agent}] {d.intent.op.value} {d.intent.target or ''} ← {d.rationale}")
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="tianlong", description="图世界文字游戏")
    ap.add_argument("--llm", choices=["auto", "none"], default="auto", help="auto：有 GEMINI_API_KEY 则启用")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--store", choices=["memory", "neo4j"], default="memory", help="neo4j：读取 NEO4J_URI 等环境变量")
    ap.add_argument("--save", default=None, help="存档名（作为 world_id，Neo4j 下可跨进程保留）")
    ap.add_argument("--predictor", choices=["heuristic", "gnn"], default="heuristic",
                    help="gnn：加载 artifacts/dynamics_agent.pt 作为 NPC 的后果预测器")
    ap.add_argument("--policy", choices=["scripted", "learned"], default="scripted",
                    help="learned：加载 artifacts/policy_ppo.pt 作为 NPC 的决策策略")
    ap.add_argument("--artifacts", default="artifacts")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.WARNING)

    llm = llm_from_env() if args.llm == "auto" else None
    scenario = build_warehouse(args.seed)
    if args.save:
        scenario = replace(scenario, world_id=args.save)
    store = None
    if args.store == "neo4j":
        from tianlong.persistence.neo4j_store import Neo4jWorldStore
        store = Neo4jWorldStore.from_env()
    predictor, policies, max_cands = None, None, 64
    if args.predictor == "gnn":
        from tianlong.learning.predictor import GNNPredictor
        predictor = GNNPredictor.load(f"{args.artifacts}/dynamics_agent.pt")
    if args.policy == "learned":
        from tianlong.learning.rl.policy import LearnedPolicy
        learned = LearnedPolicy.load(f"{args.artifacts}/policy_ppo.pt")
        policies = dict.fromkeys(scenario.npcs, learned)
        max_cands = learned.spec.max_cands
    session = GameSession(scenario, store=store, llm=llm, policies=policies, predictor=predictor,
                          max_candidates=max_cands)
    debug = args.debug
    print(f"【{session.clock()}】{'（Gemini 叙述）' if llm else '（模板叙述）'}")
    print(session.intro())
    print(_HELP)
    while True:
        try:
            text = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not text:
            continue
        if text in ("/quit", "/exit"):
            return 0
        if text == "/debug":
            debug = not debug
            print(f"开发者视角：{'开' if debug else '关'}")
            continue
        if text == "/beliefs":
            store = session.beliefs(session.player)
            for b in store.sorted_beliefs():
                if not b.prop.is_attr or b.holds:
                    tag = "传闻" if b.hearsay else "亲见"
                    print(f"  [{tag} {b.confidence:.1f}] {render_fact(Fact(b.prop, b.holds), store.entities, session.player)}")
            continue
        r = session.turn(text)
        print(f"【{r.clock}】{r.narration}")
        if debug and r.advanced:
            print("\n".join(_debug_lines(r)))


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
