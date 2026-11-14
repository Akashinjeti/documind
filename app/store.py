from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from .chunking import Chunk
from .embeddings import Embedder


@dataclass(frozen=True)
class Hit:
    id: str
    text: str
    source: str
    chunk_index: int
    score: float

    def as_dict(self) -> dict:
        return {
            "id": self.id,
            "text": self.text,
            "source": self.source,
            "chunk_index": self.chunk_index,
            "score": round(self.score, 4),
        }


class VectorStore:
    """Thin Qdrant wrapper. Remote (url), on-disk (path) or in-memory (default)."""

    def __init__(
        self,
        embedder: Embedder,
        collection: str = "documind",
        url: str | None = None,
        path: str | None = None,
    ) -> None:
        self.embedder = embedder
        self.collection = collection
        if url:
            self.client = QdrantClient(url=url)
        elif path:
            self.client = QdrantClient(path=path)
        else:
            self.client = QdrantClient(":memory:")
        if not self.client.collection_exists(collection):
            self.client.create_collection(
                collection_name=collection,
                vectors_config=VectorParams(size=embedder.dim, distance=Distance.COSINE),
            )

    @staticmethod
    def _point_id(chunk: Chunk) -> str:
        # Deterministic id => re-ingesting the same document is idempotent.
        digest = hashlib.sha1(f"{chunk.source}|{chunk.index}|{chunk.text}".encode()).hexdigest()
        return str(uuid.uuid5(uuid.NAMESPACE_URL, digest))

    def add(self, chunks: list[Chunk]) -> int:
        if not chunks:
            return 0
        vectors = self.embedder.embed([c.text for c in chunks])
        points = [
            PointStruct(
                id=self._point_id(c),
                vector=v,
                payload={"text": c.text, "source": c.source, "chunk_index": c.index},
            )
            for c, v in zip(chunks, vectors)
        ]
        self.client.upsert(collection_name=self.collection, points=points)
        return len(points)

    def search(self, query: str, k: int = 4) -> list[Hit]:
        vec = self.embedder.embed([query])[0]
        res = self.client.query_points(
            collection_name=self.collection, query=vec, limit=k, with_payload=True
        )
        return [
            Hit(
                id=str(p.id),
                text=p.payload["text"],
                source=p.payload["source"],
                chunk_index=p.payload["chunk_index"],
                score=float(p.score),
            )
            for p in res.points
        ]

    def count(self) -> int:
        return self.client.count(collection_name=self.collection, exact=True).count
