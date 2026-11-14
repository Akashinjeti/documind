from __future__ import annotations

import httpx
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .chunking import chunk_text
from .config import Settings
from .embeddings import get_embedder
from .graph import DocuMindAgent
from .llm import LLM, DemoLLM, GroqLLM
from .store import VectorStore

MAX_UPLOAD_BYTES = 1_000_000
ALLOWED_SUFFIXES = (".txt", ".md")


class IngestRequest(BaseModel):
    source: str = Field(min_length=1, max_length=200)
    text: str = Field(min_length=1)


class IngestResponse(BaseModel):
    source: str
    chunks_added: int
    total_chunks: int


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)


class AskResponse(BaseModel):
    answer: str
    grounded: bool
    attempts: int
    sources: list[dict]
    trace: list[str]


def create_app(
    settings: Settings | None = None,
    store: VectorStore | None = None,
    llm: LLM | None = None,
) -> FastAPI:
    settings = settings or Settings()
    store = store or VectorStore(
        get_embedder(settings.embeddings),
        collection=settings.collection,
        url=settings.qdrant_url,
        path=settings.qdrant_path,
    )
    if llm is None:
        llm = (
            GroqLLM(settings.groq_api_key, settings.groq_model, settings.groq_base_url)
            if settings.groq_api_key
            else DemoLLM()
        )
    mode = "demo" if isinstance(llm, DemoLLM) else "llm"
    agent = DocuMindAgent(store, llm, k=settings.top_k, max_retries=settings.max_retries)

    app = FastAPI(title="DocuMind", version="1.0.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.cors_origins),
        allow_methods=["*"],
        allow_headers=["*"],
    )

    def _ingest(source: str, text: str) -> IngestResponse:
        added = store.add(chunk_text(text, source))
        if added == 0:
            raise HTTPException(status_code=422, detail="document has no usable text")
        return IngestResponse(source=source, chunks_added=added, total_chunks=store.count())

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "mode": mode, "chunks": store.count()}

    @app.post("/ingest", response_model=IngestResponse)
    def ingest(req: IngestRequest) -> IngestResponse:
        return _ingest(req.source, req.text)

    @app.post("/ingest/file", response_model=IngestResponse)
    async def ingest_file(file: UploadFile = File(...)) -> IngestResponse:
        name = file.filename or "upload"
        if not name.lower().endswith(ALLOWED_SUFFIXES):
            raise HTTPException(status_code=415, detail="only .txt and .md files are supported")
        data = await file.read(MAX_UPLOAD_BYTES + 1)
        if len(data) > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail="file larger than 1 MB")
        return _ingest(name, data.decode("utf-8", errors="replace"))

    @app.post("/ask", response_model=AskResponse)
    def ask(req: AskRequest) -> AskResponse:
        try:
            return AskResponse(**agent.ask(req.question))
        except httpx.HTTPError as exc:
            raise HTTPException(status_code=502, detail=f"LLM provider error: {exc}") from exc

    return app
