"""
[INPUT]: 依赖 core 的 Outcome / Change / Fact / RULE_REASONS
[OUTPUT]: 对外提供 Resolution（一条规则对一个意图的裁定结果）及 succeed / fail 构造器
[POS]: kernel 的裁定值对象；规则产出它，kernel 据此生成事件，perception 据此生成行动者的自我感知。
       原因必须来自 core 的封闭词表：新增失败原因忘了登记，构造时就爆炸，而不是在特征编码里悄悄落进“其他”
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

from __future__ import annotations

from dataclasses import dataclass

from tianlong.core import RULE_REASONS, Change, Fact, Outcome


@dataclass(frozen=True, slots=True)
class Resolution:
    outcome: Outcome
    reason: str | None = None
    changes: tuple[Change, ...] = ()
    learned: tuple[Fact, ...] = ()   # 行动者从结果中额外获知的事实（门是锁着的、钥匙不配……）
    scopes: tuple[str, ...] = ()     # 行动者因此完整看清（含藏匿物）其内容的容纳者

    def __post_init__(self) -> None:
        if self.reason is not None and self.reason not in RULE_REASONS:
            raise ValueError(f"未登记的结算原因 {self.reason!r}：先在 core/events.RULE_REASONS 登记")


def succeed(
    changes: tuple[Change, ...] = (), learned: tuple[Fact, ...] = (), scopes: tuple[str, ...] = ()
) -> Resolution:
    return Resolution(Outcome.SUCCESS, None, changes, learned, scopes)


def fail(reason: str, learned: tuple[Fact, ...] = ()) -> Resolution:
    return Resolution(Outcome.FAILURE, reason, (), learned)
