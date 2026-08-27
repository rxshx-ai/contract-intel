"""Invariant 1: ungrounded output must be unable to reach a user."""

import pytest

from api.schemas import ClauseClaim, ClauseType, Finding, Span
from api.verify import verify_claims, verify_findings


def _claim(doc, start, end, quote=None):
    return ClauseClaim(
        id="c1", contract_id="k1", clause_type=ClauseType.LIABILITY_CAP,
        span=Span(doc_id=doc.id, char_start=start, char_end=end,
                  quote=quote if quote is not None else doc.text[start:end]),
    )


def test_grounded_claim_survives(northwind):
    docs = {northwind.id: northwind}
    kept, rep = verify_claims([_claim(northwind, 0, 25)], docs)
    assert len(kept) == 1
    assert rep.grounding_rate == 1.0


def test_fabricated_quote_is_dropped(northwind):
    """The core defence: a plausible invented cap never reaches the user."""
    docs = {northwind.id: northwind}
    fake = _claim(northwind, 0, 25, quote="liability shall not exceed $9,000,000")
    kept, rep = verify_claims([fake], docs)
    assert kept == []
    assert rep.dropped == 1
    assert rep.grounding_rate == 0.0
    assert "not the document text" in rep.reasons[0]


def test_offset_drift_is_caught(northwind):
    """A quote that is real but points at the wrong place is still dropped."""
    docs = {northwind.id: northwind}
    real_quote = northwind.text[400:460]
    drifted = _claim(northwind, 0, 60, quote=real_quote)
    kept, _ = verify_claims([drifted], docs)
    assert kept == []


def test_out_of_bounds_and_inverted_spans_are_dropped(northwind):
    docs = {northwind.id: northwind}
    huge = ClauseClaim(
        id="c", contract_id="k", clause_type=ClauseType.TERM,
        span=Span(doc_id=northwind.id, char_start=0,
                  char_end=len(northwind.text) + 500, quote="x"),
    )
    inverted = ClauseClaim(
        id="c", contract_id="k", clause_type=ClauseType.TERM,
        span=Span(doc_id=northwind.id, char_start=50, char_end=10, quote=""),
    )
    kept, rep = verify_claims([huge, inverted], docs)
    assert kept == []
    assert rep.dropped == 2


def test_unknown_document_is_dropped(northwind):
    kept, rep = verify_claims([_claim(northwind, 0, 25)], {})
    assert kept == []
    assert "unknown document" in rep.reasons[0]


def test_missing_clause_finding_is_exempt(northwind):
    """Invariant 5: absence has nothing to quote."""
    finding = Finding(id="f", kind="missing_clause", severity="high",
                      title="No liability cap", explanation="...")
    kept, _ = verify_findings([finding], {northwind.id: northwind})
    assert len(kept) == 1


def test_evidenced_finding_with_bad_span_is_dropped(northwind):
    finding = Finding(
        id="f", kind="risky_clause", severity="high", title="t", explanation="e",
        evidence=[Span(doc_id=northwind.id, char_start=0, char_end=10, quote="WRONG TEXT")],
    )
    kept, rep = verify_findings([finding], {northwind.id: northwind})
    assert kept == []
    assert rep.dropped == 1


def test_empty_input_reports_full_grounding():
    kept, rep = verify_claims([], {})
    assert kept == [] and rep.grounding_rate == 1.0
