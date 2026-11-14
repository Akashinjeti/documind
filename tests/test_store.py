from pathlib import Path

from app.chunking import chunk_text
from app.embeddings import HashEmbedder, get_embedder
from app.store import VectorStore

DOCS = Path(__file__).resolve().parent.parent / "data" / "sample_docs"


def make_store() -> VectorStore:
    store = VectorStore(HashEmbedder(), collection="t")
    for p in sorted(DOCS.glob("*.md")):
        store.add(chunk_text(p.read_text(), p.name, max_chars=300, overlap=50))
    return store


def test_search_returns_the_relevant_document():
    store = make_store()
    top = store.search("How long do refunds take to process?", k=2)[0]
    assert top.source == "refund-policy.md"
    top = store.search("When does the on-call rotation start for new engineers?", k=2)[0]
    assert top.source == "onboarding-guide.md"


def test_ingest_is_idempotent():
    store = make_store()
    before = store.count()
    p = DOCS / "refund-policy.md"
    store.add(chunk_text(p.read_text(), p.name, max_chars=300, overlap=50))
    assert store.count() == before


def test_scores_are_sorted_desc():
    hits = make_store().search("refund credit gift card", k=4)
    scores = [h.score for h in hits]
    assert scores == sorted(scores, reverse=True)


def test_hash_embedder_is_normalised_and_deterministic():
    e = HashEmbedder()
    a, b = e.embed(["refund policy"])[0], e.embed(["refund policy"])[0]
    assert a == b
    assert abs(sum(x * x for x in a) - 1.0) < 1e-6


def test_unknown_embedder_rejected():
    import pytest

    with pytest.raises(ValueError):
        get_embedder("nope")
