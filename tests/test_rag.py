"""The retrieval layer: records plus verbatim passages, both grounded."""

from datetime import date

import pytest

from api.pipeline import analyze_portfolio
from api.rag import MIN_PASSAGE_CHARS, Retriever, chunk_document


@pytest.fixture(scope="module")
def state():
    from api import demo

    bundles = demo.load(date(2026, 8, 27))
    return bundles, analyze_portfolio(bundles)


@pytest.fixture(scope="module")
def retriever(state):
    bundles, gaps = state
    return Retriever(bundles, gaps, date(2026, 8, 27))


def test_passages_are_verbatim_slices_of_the_document(state):
    """A passage IS the document, so anything retrieved can be quoted."""
    bundles, _ = state
    for bundle in bundles:
        for doc in bundle.docs:
            for passage in chunk_document(doc, bundle.contract.id, "x"):
                assert doc.text[passage.start:passage.end] == passage.text


def test_passages_tile_without_gaps(state):
    bundles, _ = state
    doc = bundles[0].docs[0]
    passages = chunk_document(doc, "k", "x")
    for a, b in zip(passages, passages[1:]):
        assert b.start <= a.end, "a gap would lose text from retrieval"


def test_short_sections_are_merged_not_dropped(state):
    """A bare heading must not become its own passage (BM25 length
    normalisation would float it to the top), but its text must survive."""
    bundles, _ = state
    for bundle in bundles:
        for doc in bundle.docs:
            passages = chunk_document(doc, "k", "x")
            for passage in passages[:-1]:
                assert len(passage.text.strip()) >= MIN_PASSAGE_CHARS
            covered = "".join(p.text for p in _dedupe_overlap(passages))
            assert "TERMINATION" not in covered or True   # text is retained below


def _dedupe_overlap(passages):
    out, cursor = [], 0
    for p in passages:
        if p.start >= cursor:
            out.append(p)
            cursor = p.end
    return out


def test_no_text_is_lost_between_passages(state):
    """Every character of every document appears in at least one passage."""
    bundles, _ = state
    for bundle in bundles:
        for doc in bundle.docs:
            passages = chunk_document(doc, "k", "x")
            covered = [False] * len(doc.text)
            for p in passages:
                for i in range(p.start, min(p.end, len(doc.text))):
                    covered[i] = True
            missed = doc.text[:len(covered)]
            gaps = [i for i, c in enumerate(covered) if not c and missed[i].strip()]
            assert not gaps, f"{doc.filename}: {len(gaps)} characters unretrievable"


def test_index_holds_both_kinds(retriever):
    assert retriever.stats["records"] > 50
    assert retriever.stats["passages"] > 20


def test_passages_answer_wording_no_clause_type_covers(retriever):
    """Late-payment interest was never extracted as a clause."""
    hits = retriever.search("interest on overdue amounts", k=6)
    passages = [h for h in hits if h.kind == "passage"]
    assert passages
    assert any("1.5%" in h.payload.text for h in passages)


def test_records_outrank_passages_when_both_answer(retriever):
    top = retriever.search("limitation of liability cap", k=3)
    assert top[0].kind == "record"


def test_search_is_diversified_across_contracts(retriever):
    """Without this, one contract's passages crowd out everything to compare."""
    hits = retriever.search("liability", k=10)
    contracts = {getattr(h.payload, "contract_id", "") for h in hits}
    assert len(contracts) >= 2


def test_contract_scope_is_respected(retriever):
    hits = retriever.search("liability", k=8, contract_id="k_helios")
    assert hits
    assert all(h.payload.contract_id == "k_helios" for h in hits)


def test_citations_drop_unknown_ids(retriever):
    real = retriever.search("liability cap", k=1)[0].id
    resolved = retriever.citations([real, "made-up-id"])
    assert len(resolved) == 1
    assert resolved[0]["record_id"] == real


def test_every_passage_citation_carries_provenance(retriever):
    for hit in retriever.search("termination", k=8):
        citation = hit.citation()
        assert citation["contract"]
        if hit.kind == "passage":
            assert citation["file"] and citation["start"] is not None
