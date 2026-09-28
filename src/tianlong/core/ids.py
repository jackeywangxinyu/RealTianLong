"""
[INPUT]: 依赖 hashlib 的 blake2b
[OUTPUT]: 对外提供 digest / derive_seed / make_id 三个确定性派生函数
[POS]: core 的确定性基石；事件 ID、观察 ID、随机种子全部由此派生，保证“相同输入 → 相同回放”
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

from __future__ import annotations

import hashlib

# ============================================================
#  为什么不用 hash() / uuid4()
#  - Python 的 str hash 每个进程随机化（PYTHONHASHSEED），无法回放
#  - uuid4 不可复现；回放同一存档必须得到同一批事件 ID
#  所以一切标识与种子都从“语义输入”经 blake2b 派生
# ============================================================

_SEP = "\x1f"


def digest(*parts: object) -> str:
    """把任意可 repr 的片段稳定地摘要为十六进制串。"""
    payload = _SEP.join(repr(p) for p in parts).encode("utf-8")
    return hashlib.blake2b(payload, digest_size=16).hexdigest()


def derive_seed(*parts: object) -> int:
    """从语义片段派生 64 位随机种子（供 random.Random 使用）。"""
    return int(digest(*parts)[:16], 16)


def make_id(prefix: str, *parts: object) -> str:
    """生成形如 evt_3f2a... 的确定性 ID。"""
    return f"{prefix}_{digest(*parts)[:12]}"
