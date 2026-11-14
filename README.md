# DocuMind

Multi-agent RAG over your own documents. Upload `.txt`/`.md` files, ask questions, get **cited answers** that a critic agent has checked against the retrieved context.

**Stack:** LangGraph · Qdrant · FastAPI · Groq (LLaMA 3.3 70B) · React + TypeScript (Vite) · Docker

## How it works

```
START -> planner -> retriever -> answerer -> critic --grounded--> END
                        ^                       |
                        +------ rewriter <------+   (not grounded, retries left)
```

| Agent | Job |
|---|---|
| Planner | Splits the question into 1-3 focused search queries |
| Retriever | Searches Qdrant for each query, de-duplicates chunks, keeps the best |
| Answerer | Writes an answer from the numbered passages only, with `[n]` citations |
| Critic | Checks every claim is supported; if not, proposes a better search query |
| Rewriter | Feeds the critic's query back into retrieval (bounded by `MAX_RETRIES`) |

Design choices: deterministic point IDs make re-ingesting a file idempotent; a broken critic reply fails open instead of looping; if nothing is retrieved the answerer and critic are skipped and the API says so instead of guessing; every response includes the agent trace.

## Run it

**Docker (full stack, real embeddings):**
```bash
cp .env.example .env      # add GROQ_API_KEY
docker compose up --build
# UI  http://localhost:8080   API docs  http://localhost:8000/docs
```

**Local (no Docker, no downloads):**
```bash
pip install -r requirements-dev.txt
uvicorn app.main:create_app --factory --reload      # in-memory Qdrant, demo mode without a key
cd frontend && npm install && npm run dev            # http://localhost:5173
```

Try the sample data:
```bash
for f in data/sample_docs/*.md; do curl -s -F "file=@$f" localhost:8000/ingest/file; echo; done
curl -s localhost:8000/ask -H 'content-type: application/json' \
  -d '{"question":"How long do refunds take to process?"}'
```

## Modes

- **Demo mode** (no `GROQ_API_KEY`): an offline extractive stand-in replaces the LLM so the whole pipeline runs end to end. It proves the wiring, not answer quality.
- **LLM mode** (`GROQ_API_KEY` set): planner, answerer and critic are real LLaMA 3.3 70B calls.
- `EMBEDDINGS=hash` is a dependency-free hashing embedder for dev/tests; use `EMBEDDINGS=fastembed` (default in Docker) for real semantic search.

## API

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | status, mode, chunk count |
| POST | `/ingest` | `{source, text}` |
| POST | `/ingest/file` | multipart `.txt`/`.md`, max 1 MB |
| POST | `/ask` | `{question}` returns `{answer, grounded, attempts, sources, trace}` |

## Tests

```bash
pytest -q     # 20 tests: chunking, Qdrant store, graph routing/retry caps, API validation
```

## Limitations / next steps

- Plain-text and Markdown only (no PDF yet).
- Critic quality depends on the model; no automated answer-quality eval yet (a small labelled Q&A set with a pass-rate metric is the next step).
- No auth or per-user collections.
