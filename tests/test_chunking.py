"""Chunks must tile the document and keep absolute offsets."""

import pytest

from api.chunking import (
    DEFAULT_MAX_CHARS,
    chunk_document,
    estimate_tokens,
    section_boundaries,
)
from api.ingest import find_span, ingest_text


def test_short_document_is_one_chunk(nda):
    chunks = chunk_document(nda, max_chars=100_000)
    assert len(chunks) == 1
    assert chunks[0].text == nda.text
    assert (chunks[0].start, chunks[0].end) == (0, len(nda.text))


def test_long_document_is_split(northwind):
    chunks = chunk_document(northwind)
    assert len(chunks) > 1
    assert all(c.total == len(chunks) for c in chunks)


def test_chunk_text_matches_its_absolute_offsets(northwind):
    """The invariant that keeps Spans valid after chunking."""
    for chunk in chunk_document(northwind):
        assert northwind.text[chunk.start:chunk.end] == chunk.text


def test_chunks_cover_the_whole_document(northwind):
    chunks = chunk_document(northwind)
    assert chunks[0].start == 0
    assert chunks[-1].end == len(northwind.text)
    for a, b in zip(chunks, chunks[1:]):
        assert b.start <= a.end, "gap between chunks would lose clauses"


def test_chunks_respect_the_size_budget(northwind, acme):
    for doc in (northwind, acme):
        for chunk in chunk_document(doc):
            assert len(chunk.text) <= DEFAULT_MAX_CHARS * 1.8


def test_headings_are_detected(northwind):
    bounds = section_boundaries(northwind.text)
    assert len(bounds) > 15
    starts = [northwind.text[b:b + 6] for b in bounds[1:-1]]
    assert any(s.startswith("11.") for s in starts)


def test_key_clauses_survive_chunking_whole(northwind):
    """A clause split across a boundary would be lost from both chunks."""
    critical = [
        "SHALL NOT EXCEED FIFTY THOUSAND DOLLARS ($50,000)",
        "sixty (60) days prior to the end of the then-current Term",
        "Provider may modify the terms of this Agreement at any time",
    ]
    chunks = chunk_document(northwind)
    for phrase in critical:
        assert any(phrase in c.text for c in chunks), phrase


def test_quotes_from_any_chunk_ground_against_the_full_document(northwind):
    for chunk in chunk_document(northwind):
        for line in chunk.text.split("\n"):
            line = line.strip()
            if len(line) < 40:
                continue
            span = find_span(northwind, line, hint=chunk.start)
            assert span is not None
            assert span.is_grounded_in(northwind.text)


def test_token_estimate_is_conservative(northwind):
    assert estimate_tokens(northwind.text) > len(northwind.text) / 4


# -- truncation recovery ---------------------------------------------------

def test_split_chunk_halves_at_a_structural_boundary(northwind):
    from api.chunking import split_chunk

    chunk = chunk_document(northwind)[0]
    halves = split_chunk(chunk)
    assert len(halves) == 2
    assert halves[0].text + halves[1].text == chunk.text
    assert halves[0].start == chunk.start
    assert halves[1].end == chunk.end


def test_split_chunk_preserves_absolute_offsets(northwind):
    """Spans found in a half must still resolve against the full document."""
    from api.chunking import split_chunk

    for chunk in chunk_document(northwind):
        for half in split_chunk(chunk):
            assert northwind.text[half.start:half.end] == half.text


def test_tiny_chunks_are_not_split():
    from api.chunking import Chunk, split_chunk

    tiny = Chunk(text="short clause", start=0, end=12, index=0, total=1)
    assert split_chunk(tiny) == [tiny]
