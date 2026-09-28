"""
[INPUT]: 依赖 qdrant_client 的 QdrantClient / models，memory/embedder 的 Embedder，core/memories 的 MemoryRecord
[OUTPUT]: 对外提供 MemoryScope（服务端强制检索边界）、Recollection、MemoryIndex 协议、QdrantMemoryIndex
[POS]: memory 的向量索引；检索边界（世界/分支/主人/获知时间/类别）由运行时构造并强制写入过滤条件，不交给角色或 LLM 自觉遵守
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

from __future__ import annotations

import uuid
import warnings
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Protocol

from qdrant_client import QdrantClient, models

from tianlong.core.memories import MemoryRecord
from tianlong.memory.embedder import Embedder, HashingEmbedder

_NAMESPACE = uuid.UUID("5b1f6a0e-7c1d-4b5e-9a57-7469616e6c6f")


@dataclass(frozen=True, slots=True)
class MemoryScope:
    world_id: str
    branch_id: str
    owner: str
    now: int                                # 只能回忆 known_at <= now 的经历
    kinds: tuple[str, ...] | None = None


@dataclass(frozen=True, slots=True)
class Recollection:
    record: MemoryRecord
    score: float


class MemoryIndex(Protocol):
    def upsert(self, records: Sequence[MemoryRecord]) -> None: ...
    def search(self, scope: MemoryScope, query: str, limit: int = 5) -> list[Recollection]: ...


def point_id(record_id: str) -> str:
    """记录 ID → 稳定 UUID：重复索引同一记录是覆盖而非追加，outbox 重放天然幂等。"""
    return str(uuid.uuid5(_NAMESPACE, record_id))


class QdrantMemoryIndex:
    def __init__(
        self, client: QdrantClient | None = None, embedder: Embedder | None = None, collection: str = "memories"
    ) -> None:
        self.client = client or QdrantClient(location=":memory:")
        self.embedder = embedder or HashingEmbedder()
        self.collection = collection
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        if self.client.collection_exists(self.collection):
            return
        self.client.create_collection(
            self.collection,
            vectors_config=models.VectorParams(size=self.embedder.dim, distance=models.Distance.COSINE),
        )
        # 服务端需要载荷索引支撑过滤；本地模式会忽略并告警，这里静默之
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            for key in ("world_id", "branch_id", "owner", "kind"):
                self.client.create_payload_index(self.collection, key, models.PayloadSchemaType.KEYWORD)
            self.client.create_payload_index(self.collection, "known_at", models.PayloadSchemaType.INTEGER)

    def upsert(self, records: Sequence[MemoryRecord]) -> None:
        if not records:
            return
        vectors = self.embedder.embed([r.text for r in records])
        points = [
            models.PointStruct(id=point_id(r.id), vector=v, payload={**asdict(r), "subjects": list(r.subjects)})
            for r, v in zip(records, vectors, strict=True)
        ]
        self.client.upsert(self.collection, points=points)

    def search(self, scope: MemoryScope, query: str, limit: int = 5) -> list[Recollection]:
        must: list[models.Condition] = [
            models.FieldCondition(key="world_id", match=models.MatchValue(value=scope.world_id)),
            models.FieldCondition(key="branch_id", match=models.MatchValue(value=scope.branch_id)),
            models.FieldCondition(key="owner", match=models.MatchValue(value=scope.owner)),
            models.FieldCondition(key="known_at", range=models.Range(lte=scope.now)),
        ]
        if scope.kinds:
            must.append(models.FieldCondition(key="kind", match=models.MatchAny(any=list(scope.kinds))))
        vector = self.embedder.embed([query])[0]
        hits = self.client.query_points(
            self.collection, query=vector, query_filter=models.Filter(must=must), limit=limit
        ).points
        return [Recollection(_record(h.payload or {}), float(h.score)) for h in hits]


def _record(payload: dict) -> MemoryRecord:
    return MemoryRecord(**{**payload, "subjects": tuple(payload.get("subjects", ()))})
