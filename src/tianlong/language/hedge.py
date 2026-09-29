"""
[INPUT]: 依赖标准库 queue / threading / concurrent.futures，language/llm 的 LLMClient / LLMUnavailable
[OUTPUT]: 对外提供 HedgedLLM（首字对冲：主模型迟迟不出字，就把同一请求并发给更快的备用模型，谁先出字用谁）
[POS]: language 的延迟长尾治理，包在任何 LLMClient 外面、自己也是 LLMClient。主持人之声的首字延迟由最慢的那几回合决定——
       主模型（文笔更好）首字中位数够快，但偶尔要等五六秒；与其全程换成快而粗的模型，不如只在它迟到时请替补：
       after 秒内主模型没吐出第一段字（或出错），就把同一请求发给备用模型，两路赛跑，先出字的一路讲完整段，另一路作废
       （它的读线程在下一段字到达时关闭连接，不再交付任何文字）。主模型在 after 之前出错也立即换备用；两路都失败才抛
       LLMUnavailable，调用方照旧回退模板。读流放在常驻线程池里，每个线程的 HTTPS 长连接跨回合复用。
       generate()（解释器 JSON、场外问答、终章）不对冲，只在主模型失败时交给备用
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

from __future__ import annotations

import queue
import threading
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from tianlong.language.llm import LLMClient, LLMUnavailable

_CHUNK, _END, _ERROR = "chunk", "end", "error"


class _Runner:
    """在常驻线程里读一路流，把每段字、结束或出错按 (谁, 类别, 内容) 放进共享队列；被作废后不再放任何东西。
    线程来自 HedgedLLM 的常驻池：GeminiClient 每个线程一条长连接，线程常驻，连接才复用得上（每回合新开线程就要重新握手）。"""

    def __init__(self, who: str, chunks: Iterator[str], out: queue.Queue, pool: ThreadPoolExecutor) -> None:
        self.who = who
        self.cancelled = threading.Event()
        self._chunks = chunks
        self._out = out
        pool.submit(self._run)

    def _run(self) -> None:
        try:
            for piece in self._chunks:
                if self.cancelled.is_set():
                    break
                self._out.put((self.who, _CHUNK, piece))
            else:
                self._out.put((self.who, _END, None))
        except Exception as e:  # noqa: BLE001 —— 任何失败都交给赛跑的一方判断
            if not self.cancelled.is_set():
                self._out.put((self.who, _ERROR, e))
        finally:
            close = getattr(self._chunks, "close", None)
            if callable(close):
                close()                                  # 作废的一路：关掉半读的连接


class HedgedLLM:
    """primary 首字超过 after 秒（或提前出错）就并发 backup；after <= 0 等于不对冲，只在主模型失败时用备用。
    hedged / switched 记下对冲次数与最终用了备用的次数，评测读它。"""

    def __init__(self, primary: LLMClient, backup: LLMClient, after: float = 2.5) -> None:
        self.primary = primary
        self.backup = backup
        self.after = after
        self.model = primary.model
        self.stats = getattr(primary, "stats", None)
        self.hedged = 0
        self.switched = 0
        # 常驻读流线程：一回合至多两路，作废的一路要等下一段字到达才关得掉，多留两个免得下一回合排队
        self._pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="hedge")

    def generate(self, prompt: str, *, system: str | None = None, schema: dict[str, Any] | None = None,
                 temperature: float | None = None, max_tokens: int | None = None) -> str:
        kw = {"system": system, "schema": schema, "temperature": temperature, "max_tokens": max_tokens}
        try:
            return self.primary.generate(prompt, **kw)
        except LLMUnavailable:
            self.switched += 1
            return self.backup.generate(prompt, **kw)

    def stream(self, prompt: str, *, system: str | None = None, max_tokens: int | None = None) -> Iterator[str]:
        events: queue.Queue = queue.Queue()

        def start(who: str, llm: LLMClient) -> _Runner:
            return _Runner(who, llm.stream(prompt, system=system, max_tokens=max_tokens), events, self._pool)

        runners = {"primary": start("primary", self.primary)}
        failed: dict[str, BaseException] = {}
        winner: str | None = None
        first: str | None = None
        wait: float | None = self.after if self.after > 0 else None
        try:
            while winner is None:
                try:
                    who, kind, value = events.get(timeout=wait)
                except queue.Empty:                      # 主模型迟到：请替补上场，此后两路赛跑
                    self.hedged += 1
                    runners["backup"] = start("backup", self.backup)
                    wait = None
                    continue
                if who in failed:
                    continue
                if kind == _ERROR:
                    failed[who] = value
                    if "backup" not in runners:          # 主模型提前出错：不必等满 after
                        runners["backup"] = start("backup", self.backup)
                        wait = None
                    elif len(failed) == len(runners):
                        raise LLMUnavailable(f"主备模型都失败了：{failed.get('primary')}；{failed.get('backup')}")
                    continue
                winner, first = who, value if kind == _CHUNK else None
            for who, r in runners.items():
                if who != winner:
                    r.cancelled.set()
            if winner == "backup":
                self.switched += 1
            if first is None:                            # 胜者一个字没有就讲完了（空回复）：交给调用方按空处理
                return
            yield first
            while True:
                who, kind, value = events.get()
                if who != winner:
                    continue
                if kind == _CHUNK:
                    yield value
                elif kind == _END:
                    return
                else:
                    raise LLMUnavailable(f"{who} 中途断线：{value}") from value
        finally:
            for r in runners.values():                   # 调用方中途不读了（或出错）：两路都作废
                r.cancelled.set()
