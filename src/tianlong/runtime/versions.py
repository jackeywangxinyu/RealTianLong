"""
[INPUT]: 依赖 kernel 的 KERNEL_VERSION，core/attributes 的 ATTRIBUTES，core/goals 的 GOALS_VERSION，core/ids 的 digest
[OUTPUT]: 对外提供 SAVE_FORMAT、current_versions()（建档时写进存档的版本表）、check_save()、IncompatibleSave
[POS]: runtime 的存档版本闸门。存档里的事件、认知与会话运行态都是按建档时的规则、属性规格与目标语义产生的；
       读档时版本表不一致就明确拒绝，除非调用方显式声明迁移（allow_migration=True）——
       即便迁移，也只是接受“用当前代码接着玩这份旧档”，绝不凭当前真相补写旧档当时并不知道的信息
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

from __future__ import annotations

from collections.abc import Mapping

from tianlong.core.attributes import ATTRIBUTES
from tianlong.core.goals import GOALS_VERSION
from tianlong.core.ids import digest
from tianlong.kernel import KERNEL_VERSION

# 存档格式：v2 起每次提交携带请求进度（TurnEnvelope）与会话运行态（调度标记、已描写实体）
SAVE_FORMAT = "save-v2"


class IncompatibleSave(Exception):
    """存档版本与当前代码不一致：拒绝静默接续。"""


def _attributes_digest() -> str:
    # frozenset 的迭代顺序随进程哈希种子变化：种类集合先排序再摘要，否则同一份规格在两个进程里摘要不同
    rows = tuple((a.key, a.type.value, tuple(sorted(k.value for k in a.kinds)), a.access.value, a.scale,
                  a.categories, a.dynamic) for a in ATTRIBUTES)
    return digest("attributes", rows)


def current_versions() -> dict[str, str]:
    return {"save": SAVE_FORMAT, "kernel": KERNEL_VERSION, "attributes": _attributes_digest(), "goals": GOALS_VERSION}


def check_save(stored: Mapping[str, str], current: Mapping[str, str], allow_migration: bool = False) -> dict[str, tuple]:
    """返回不一致项 {键: (存档里的, 当前的)}；不一致且未声明迁移则抛 IncompatibleSave。缺失的键也算不一致。"""
    diff = {k: (stored.get(k), current.get(k)) for k in sorted(set(stored) | set(current))
            if stored.get(k) != current.get(k)}
    if diff and not allow_migration:
        detail = "；".join(f"{k}: 存档 {a!r} ≠ 当前 {b!r}" for k, (a, b) in diff.items())
        raise IncompatibleSave(f"存档版本不兼容（{detail}）。确认要用当前代码接续这份旧档，请显式传 allow_migration=True")
    return diff
