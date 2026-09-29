"""
[INPUT]: 依赖 tianlong.language.hedge 的 HedgedLLM，tianlong.language.llm 的 ScriptedLLM / LLMUnavailable
[OUTPUT]: 首字对冲验收：主模型准时出字就不请替补；迟到了替补上场，先出字的一路讲完整段、两路文字从不混在一起；
          主模型提前出错不必等满时限；两路都失败才抛 LLMUnavailable，胜者中途断线照样抛（已交付的留给调用方）；
          after<=0 不对冲；generate 只在主模型失败时交给备用；读流线程跨回合复用（长连接才用得上）
[POS]: tests 的延迟长尾治理：证伪“替补一上场两边的字搅在一起”“主模型挂了还要干等”“每回合都新开线程重新握手”
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

from __future__ import annotations

import threading
import time

import pytest

from tianlong.language.hedge import HedgedLLM
from tianlong.language.llm import LLMUnavailable, ScriptedLLM

MAIN = "主模型写的：龚光杰冷笑一声，长剑斜指。"
SPARE = "备用模型写的：钟灵在梁上嗑着瓜子。"


def _llm(text: str, delay: float = 0.0, fail: bool = False) -> ScriptedLLM:
    def respond(prompt, system, schema):
        if fail:
            raise LLMUnavailable("挂了")
        return text
    return ScriptedLLM(respond, model=text[:3], first_delay=delay, chunk=4)


def _run(llm: HedgedLLM) -> tuple[str, float]:
    t0 = time.perf_counter()
    text = "".join(llm.stream("写一段", system="主持"))
    return text, time.perf_counter() - t0


def test_prompt_primary_is_used_alone():
    backup = _llm(SPARE)
    h = HedgedLLM(_llm(MAIN), backup, after=0.5)
    text, _ = _run(h)
    assert text == MAIN and h.hedged == 0 and h.switched == 0 and backup.prompts == []


def test_late_primary_loses_to_the_backup_and_texts_never_mix():
    h = HedgedLLM(_llm(MAIN, delay=1.5), _llm(SPARE), after=0.1)
    text, took = _run(h)
    assert text == SPARE and h.hedged == 1 and h.switched == 1 and took < 1.0


def test_late_primary_still_wins_if_the_backup_is_slower():
    h = HedgedLLM(_llm(MAIN, delay=0.3), _llm(SPARE, delay=2.0), after=0.1)
    text, took = _run(h)
    assert text == MAIN and h.hedged == 1 and h.switched == 0 and took < 1.5


def test_early_failure_switches_without_waiting_out_the_deadline():
    h = HedgedLLM(_llm(MAIN, fail=True), _llm(SPARE), after=5.0)
    text, took = _run(h)
    assert text == SPARE and h.switched == 1 and took < 1.0


def test_both_failing_raises_unavailable():
    h = HedgedLLM(_llm(MAIN, fail=True), _llm(SPARE, fail=True), after=5.0)
    with pytest.raises(LLMUnavailable):
        _run(h)


def test_winner_breaking_mid_stream_raises_after_delivering():
    class Broken(ScriptedLLM):
        def stream(self, prompt, *, system=None, max_tokens=None):
            yield "开头一段"
            raise LLMUnavailable("断线")

    h = HedgedLLM(Broken(lambda *a: ""), _llm(SPARE), after=5.0)
    got: list[str] = []
    with pytest.raises(LLMUnavailable):
        for piece in h.stream("写一段"):
            got.append(piece)
    assert got == ["开头一段"]


def test_no_hedge_when_disabled_and_generate_falls_back_only_on_failure():
    backup = _llm(SPARE)
    h = HedgedLLM(_llm(MAIN, delay=0.3), backup, after=0)
    assert _run(h)[0] == MAIN and h.hedged == 0 and backup.prompts == []
    assert HedgedLLM(_llm(MAIN), _llm(SPARE)).generate("问") == MAIN
    assert HedgedLLM(_llm(MAIN, fail=True), _llm(SPARE)).generate("问") == SPARE


def test_reader_threads_are_reused_across_turns():
    names: set[str] = set()

    def respond(prompt, system, schema):
        names.add(threading.current_thread().name)
        return MAIN

    h = HedgedLLM(ScriptedLLM(respond), _llm(SPARE), after=5.0)
    for _ in range(5):
        assert _run(h)[0] == MAIN
    assert len(names) == 1                     # 同一条常驻线程：它的 HTTPS 长连接跨回合复用
