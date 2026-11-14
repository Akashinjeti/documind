import pytest

from app.chunking import chunk_text


def test_chunks_respect_max_chars_and_indexes():
    text = " ".join(f"Sentence number {i} talks about topic {i}." for i in range(60))
    chunks = chunk_text(text, "doc.md", max_chars=200, overlap=50)
    assert len(chunks) > 3
    assert all(len(c.text) <= 200 for c in chunks)
    assert [c.index for c in chunks] == list(range(len(chunks)))
    assert {c.source for c in chunks} == {"doc.md"}


def test_overlap_carries_context_forward():
    text = "Alpha one. Bravo two. Charlie three. Delta four. Echo five. Foxtrot six."
    chunks = chunk_text(text, "d", max_chars=40, overlap=25)
    assert len(chunks) >= 2
    first_last = chunks[0].text.split(". ")[-1].rstrip(".")
    assert first_last.split()[0] in chunks[1].text


def test_oversized_sentence_is_hard_split():
    chunks = chunk_text("word " * 500, "d", max_chars=100, overlap=0)
    assert len(chunks) >= 20
    assert all(len(c.text) <= 100 for c in chunks)


def test_empty_text_gives_no_chunks():
    assert chunk_text("   \n\n  ", "d") == []


def test_invalid_max_chars():
    with pytest.raises(ValueError):
        chunk_text("hello", "d", max_chars=0)
