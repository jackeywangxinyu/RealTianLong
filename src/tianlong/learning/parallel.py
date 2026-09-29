"""
[INPUT]: 依赖标准库 os / multiprocessing / concurrent.futures；torch 可选（子进程里限定为单线程）
[OUTPUT]: 对外提供 ordered_map()（按输入顺序返回的进程池映射）、resolve_workers()（0 = 本机全部核）
[POS]: learning 的并行原语。世界生成、内核结算、认知折叠、GNN 预测都在 CPU 上：放大训练的主要手段是把彼此独立的
       世界/回合分给多个进程。结果按输入顺序拼回，且每个单元（一个世界、一局评测、一局示范）只依赖自己的种子，
       所以并行与顺序执行逐项相同——并行度是资源旋钮，不是实验配置（不进 run_id）。
       用 spawn 而不是 fork：子进程不继承父进程的线程池、ray 与 CUDA 状态（fork 之后的 OpenMP/CUDA 会死锁）；
       每个子进程只开一个 torch 线程，多进程之间不互相抢核
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

from __future__ import annotations

import multiprocessing as mp
import os
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import ProcessPoolExecutor
from typing import Any


def resolve_workers(workers: int) -> int:
    """0 = 本机全部核；负数与 1 都是顺序执行。"""
    return (os.cpu_count() or 1) if workers == 0 else max(1, workers)


def _single_thread(initializer: Callable[..., None] | None, initargs: Sequence[Any]) -> None:
    try:
        import torch
        torch.set_num_threads(1)
    except ImportError:          # pragma: no cover  （核心零依赖：没有 torch 也能并行跑纯 Python 的工作）
        pass
    if initializer is not None:
        initializer(*initargs)


def ordered_map(fn: Callable[[Any], Any], items: Iterable[Any], workers: int = 1,
                initializer: Callable[..., None] | None = None, initargs: Sequence[Any] = ()) -> list:
    """fn(item) 的结果按 items 的顺序返回。workers ≤ 1 时就地顺序执行（此时 initializer 也在本进程里跑一次）。
    fn 与 initializer 必须是模块级函数：spawn 子进程按名字导入它们。"""
    items = list(items)
    n = min(resolve_workers(workers), len(items))
    if n <= 1:
        if initializer is not None:
            initializer(*initargs)
        return [fn(x) for x in items]
    with ProcessPoolExecutor(n, mp_context=mp.get_context("spawn"), initializer=_single_thread,
                             initargs=(initializer, tuple(initargs))) as ex:
        return list(ex.map(fn, items))
