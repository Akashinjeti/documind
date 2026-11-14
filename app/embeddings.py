from __future__ import annotations

import hashlib
import math
import re
from typing import Protocol

DIM = 384
_TOKEN = re.compile(r"[a-z0-9]+")
_STOP = frozenset(
    "a an and are as at be by for from has have i in is it of on or that the this to was were will with what when where which who how do does can".split()
)


class Embedder(Protocol):
    dim: int

    def embed(self, texts: list[str]) -> list[list[float]]: ...


class HashEmbedder:
    """Dependency-free feature-hashing embedder (unigrams + bigrams).

    Good enough for local dev, tests and demos with no model download.
    Use FastEmbedder for real semantic quality.
    """

    dim = DIM

    def _vec(self, text: str) -> list[float]:
        toks = [t for t in _TOKEN.findall(text.lower()) if t not in _STOP]
        feats = toks + [f"{a}_{b}" for a, b in zip(toks, toks[1:])]
        v = [0.0] * self.dim
        for f in feats:
            h = int.from_bytes(hashlib.blake2b(f.encode(), digest_size=8).digest(), "big")
            v[h % self.dim] += 1.0 if (h >> 63) & 1 else -1.0
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / norm for x in v]

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]


class FastEmbedder:
    """ONNX sentence embeddings via fastembed (BAAI/bge-small-en-v1.5, 384-dim)."""

    dim = DIM

    def __init__(self, model: str = "BAAI/bge-small-en-v1.5") -> None:
        from fastembed import TextEmbedding  # lazy: optional dependency

        self._model = TextEmbedding(model_name=model)

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [list(map(float, v)) for v in self._model.embed(texts)]


def get_embedder(kind: str) -> Embedder:
    if kind == "fastembed":
        return FastEmbedder()
    if kind == "hash":
        return HashEmbedder()
    raise ValueError(f"unknown embeddings backend: {kind!r} (use 'hash' or 'fastembed')")
