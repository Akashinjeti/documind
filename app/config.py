from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Settings:
    qdrant_url: str | None = field(default_factory=lambda: os.getenv("QDRANT_URL") or None)
    qdrant_path: str | None = field(default_factory=lambda: os.getenv("QDRANT_PATH") or None)
    collection: str = field(default_factory=lambda: os.getenv("QDRANT_COLLECTION", "documind"))
    embeddings: str = field(default_factory=lambda: os.getenv("EMBEDDINGS", "hash"))  # hash | fastembed
    groq_api_key: str | None = field(default_factory=lambda: os.getenv("GROQ_API_KEY") or None)
    groq_model: str = field(default_factory=lambda: os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"))
    groq_base_url: str = field(
        default_factory=lambda: os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
    )
    top_k: int = field(default_factory=lambda: int(os.getenv("TOP_K", "4")))
    max_retries: int = field(default_factory=lambda: int(os.getenv("MAX_RETRIES", "1")))
    cors_origins: tuple[str, ...] = field(
        default_factory=lambda: tuple(
            o.strip() for o in os.getenv("CORS_ORIGINS", "http://localhost:5173").split(",") if o.strip()
        )
    )
