"""Fuzzy recovery must raise recall without ever surfacing model-written text."""

import pytest

from api.ingest import find_span_fuzzy, locate


def test_exact_quote_reports_exact(northwind):
    quote = "Fees are non-refundable."
    span, how = locate(northwind, quote)
    assert how == "exact" and span.is_grounded_in(northwind.text)


def test_reflowed_quote_reports_whitespace(northwind):
    span, how = locate(northwind, "Fees are\n   non-refundable.")
    assert how == "whitespace" and span.is_grounded_in(northwind.text)


def test_dropped_word_is_recovered(northwind):
    """Model omits 'commercially' from inside the quote."""
    span, how = locate(
        northwind,
        "Provider will use reasonable efforts to make the Services available 99.9% "
        "of the time in each calendar month, excluding scheduled maintenance.",
    )
    assert how == "fuzzy"
    assert span.is_grounded_in(northwind.text)
    assert "commercially reasonable efforts" in span.quote   # the REAL text


def test_normalized_number_is_recovered(northwind):
    """Model writes '45 days' where the contract says 'forty-five (45) days'."""
    span, how = locate(
        northwind,
        "Customer shall pay all undisputed invoices within 45 days of the invoice date.",
    )
    assert how == "fuzzy"
    assert "forty-five (45) days" in span.quote


def test_recovered_span_always_quotes_the_document(all_docs):
    """Invariant 1 holds by construction for every fuzzy hit."""
    for doc in all_docs:
        for line in doc.text.split("\n"):
            line = line.strip()
            if len(line) < 60:
                continue
            mangled = line.replace(" the ", " ").replace(" of ", " ")
            span = find_span_fuzzy(doc, mangled)
            if span is not None:
                assert span.is_grounded_in(doc.text)
                assert span.quote == doc.text[span.char_start:span.char_end]


def test_invented_text_is_still_rejected(northwind):
    """Fuzzy must not become a licence to hallucinate."""
    span, how = locate(
        northwind,
        "Provider shall pay Customer nine million dollars upon any service outage.",
    )
    assert span is None and how == "dropped"


def test_unrelated_boilerplate_is_rejected(northwind):
    span, _ = locate(northwind, "The parties agree to arbitrate in Singapore.")
    assert span is None


def test_threshold_is_enforced(northwind):
    quote = "Provider will use commercially reasonable efforts to make the Services"
    assert find_span_fuzzy(northwind, quote, threshold=0.99) is not None
    heavily_mangled = "Provider might try hard sometimes to maybe make the Services"
    assert find_span_fuzzy(northwind, heavily_mangled, threshold=0.95) is None
