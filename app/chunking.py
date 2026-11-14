from __future__ import annotations

import re
from dataclasses import dataclass

_SENTENCE = re.compile(r"(?<=[.!?])\s+|\n{2,}")


@dataclass(frozen=True)
class Chunk:
    text: str
    source: str
    index: int


def _split_long(sentence: str, max_chars: int) -> list[str]:
    """Hard-split a single oversized sentence on word boundaries."""
    parts: list[str] = []
    cur = ""
    for w in sentence.split():
        if cur and len(cur) + 1 + len(w) > max_chars:
            parts.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    if cur:
        parts.append(cur)
    return parts


def chunk_text(text: str, source: str, max_chars: int = 800, overlap: int = 120) -> list[Chunk]:
    """Pack sentences into chunks of <= max_chars, carrying `overlap` chars of context forward."""
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")
    sentences: list[str] = []
    for s in _SENTENCE.split(text.strip()):
        s = " ".join(s.split())
        if not s:
            continue
        sentences.extend(_split_long(s, max_chars) if len(s) > max_chars else [s])

    chunks: list[str] = []
    cur: list[str] = []
    cur_len = 0
    for s in sentences:
        if cur and cur_len + 1 + len(s) > max_chars:
            chunks.append(" ".join(cur))
            carry: list[str] = []
            size = 0
            for prev in reversed(cur):
                if size + len(prev) > overlap:
                    break
                carry.insert(0, prev)
                size += len(prev) + 1
            cur, cur_len = carry, sum(len(c) + 1 for c in carry)
            if cur_len + 1 + len(s) > max_chars:  # carry would overflow: drop it
                cur, cur_len = [], 0
        cur.append(s)
        cur_len += len(s) + 1
    if cur:
        chunks.append(" ".join(cur))
    return [Chunk(text=c, source=source, index=i) for i, c in enumerate(chunks)]
