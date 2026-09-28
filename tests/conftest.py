"""
[INPUT]: 依赖 tianlong.scenarios / persistence / runtime.authority / kernel
[OUTPUT]: 对外提供 pytest fixtures：warehouse、authority、intent 工厂、fold 辅助
[POS]: tests 的共享夹具；所有测试从同一个标准场景出发，保证用例之间可比较
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

from __future__ import annotations

import itertools

import pytest

from tianlong.core import Fact, Intent, Manner, Op
from tianlong.persistence import InMemoryWorldStore
from tianlong.runtime.authority import WorldAuthority
from tianlong.scenarios import build_warehouse

_counter = itertools.count()


@pytest.fixture
def warehouse():
    return build_warehouse()


@pytest.fixture
def authority(warehouse):
    return WorldAuthority.found(InMemoryWorldStore(), warehouse)


def make_intent(actor: str, op: Op, target: str | None = None, obj: str | None = None, *,
                based_on: int, manner: Manner = Manner.NORMAL, topic: Fact | None = None,
                intent_id: str | None = None) -> Intent:
    return Intent(intent_id or f"it{next(_counter)}", actor, op, target, obj, manner, topic, based_on)


@pytest.fixture
def act(authority):
    """在权威写入器上让若干角色同时行动：act(("player", Op.TAKE, "key"), ...)。"""

    def _act(*specs, **kw):
        v = authority.head().version
        intents = [make_intent(*spec, based_on=v, **kw) for spec in specs]
        return authority.settle(intents)

    return _act
