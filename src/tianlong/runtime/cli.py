"""
[INPUT]: 依赖 runtime/session 的 GameSession，scenarios 的 build_warehouse，language/llm 的 llm_from_env，core 的 clock_label
[OUTPUT]: 对外提供 main()（命令行入口 `tianlong` / `python -m tianlong`）
[POS]: runtime 的终端前端；/debug 显示真相与 NPC 理由（开发者视角），/beliefs 显示玩家自己的认知——两者刻意分开
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

from __future__ import annotations

import argparse
import logging
import sys

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
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.WARNING)

    llm = llm_from_env() if args.llm == "auto" else None
    session = GameSession(build_warehouse(args.seed), llm=llm)
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
