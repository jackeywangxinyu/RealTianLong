"""
[INPUT]: 依赖 hashlib、math
[OUTPUT]: 对外提供 Embedder 协议、HashingEmbedder（字符 n-gram 特征哈希）
[POS]: memory 的文本嵌入；默认实现确定性、零依赖、离线可用、对中文友好。换成语义模型只需实现同一协议，索引与回忆代码不动
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

from __future__ import annotations

import hashlib
import math
import unicodedata
from collections.abc import Sequence
from typing import Protocol


class Embedder(Protocol):
    dim: int

    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


class HashingEmbedder:
    """按字切分的 1~3 gram，哈希到固定维度并带符号（抵消碰撞偏置），最后 L2 归一化。

    它捕捉的是字面重叠（“仓库 响动” 能召回 “仓库那边传来一阵响动”），
    不是深层语义；这是阶段 A 的有意取舍：检索质量可替换，检索边界不可妥协。
    """

    def __init__(self, dim: int = 256, ngrams: tuple[int, ...] = (1, 2, 3)) -> None:
        self.dim = dim
        self.ngrams = ngrams

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._one(t) for t in texts]

    def _one(self, text: str) -> list[float]:
        chars = [c for c in unicodedata.normalize("NFKC", text.lower()) if c.isalnum()]
        vec = [0.0] * self.dim
        for n in self.ngrams:
            for i in range(len(chars) - n + 1):
                h = hashlib.blake2b("".join(chars[i:i + n]).encode(), digest_size=8).digest()
                idx = int.from_bytes(h[:4], "little") % self.dim
                vec[idx] += 1.0 if h[4] & 1 else -1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]
