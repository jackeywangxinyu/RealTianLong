"""
[INPUT]: 依赖 core 的 Outcome / Change / Fact
[OUTPUT]: 对外提供 Resolution（一条规则对一个意图的裁定结果）及 succeed / fail 构造器
[POS]: kernel 的裁定值对象；规则产出它，kernel 据此生成事件，perception 据此生成行动者的自我感知
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

from __future__ import annotations

from dataclasses import dataclass

from tianlong.core import Change, Fact, Outcome


@dataclass(frozen=True, slots=True)
class Resolution:
    outcome: Outcome
    reason: str | None = None
    changes: tuple[Change, ...] = ()
    learned: tuple[Fact, ...] = ()   # 行动者从结果中额外获知的事实（门是锁着的、钥匙不配……）
    scopes: tuple[str, ...] = ()     # 行动者因此完整看清（含藏匿物）其内容的容纳者


def succeed(
    changes: tuple[Change, ...] = (), learned: tuple[Fact, ...] = (), scopes: tuple[str, ...] = ()
) -> Resolution:
    return Resolution(Outcome.SUCCESS, None, changes, learned, scopes)


def fail(reason: str, learned: tuple[Fact, ...] = ()) -> Resolution:
    return Resolution(Outcome.FAILURE, reason, (), learned)
