"""Invariant 1, mechanized. If these fail, every quote in the system is suspect."""

import pytest

from api.ingest import PAGE_SEP, _assemble, find_span, ingest_text, normalize
from api.schemas import ContractType


# -- the property test -----------------------------------------------------

@pytest.mark.parametrize("start", [0, 137, 900])
def test_every_offset_slice_round_trips(northwind, start):
    """text[span.start:span.end] == span.quote, for arbitrary windows."""
    text = northwind.text
    for length in (1, 40, 300):
        end = min(start + length, len(text))
        quote = text[start:end]
        span = find_span(northwind, quote)
        assert span is not None
        assert span.is_grounded_in(text)


def test_find_span_recovers_offsets_for_every_line(all_docs):
    """Every non-trivial line of every fixture must be locatable and grounded."""
    checked = 0
    for doc in all_docs:
        for line in doc.text.split("\n"):
            line = line.strip()
            if len(line) < 25:
                continue
            span = find_span(doc, line)
            assert span is not None, f"could not locate: {line[:60]!r}"
            assert span.is_grounded_in(doc.text)
            checked += 1
    assert checked > 80


def test_page_marks_tile_the_document_exactly():
    doc = ingest_text("page one text here")
    assert doc.pages[0].char_start == 0
    assert doc.pages[0].char_end == len(doc.text)
    assert doc.text[doc.pages[0].char_start : doc.pages[0].char_end] == doc.text


def test_page_marks_account_for_separator():
    text, marks = _assemble(["alpha", "beta", "gamma"], [None, None, None])
    assert text == f"alpha{PAGE_SEP}beta{PAGE_SEP}gamma"
    assert [text[m.char_start : m.char_end] for m in marks] == ["alpha", "beta", "gamma"]


def test_page_lookup_maps_offset_back_to_page():
    text, marks = _assemble(["alpha", "beta"], [None, None])
    doc = ingest_text("x")
    doc.pages = marks
    doc.text = text
    assert doc.page_for(0) == 1
    assert doc.page_for(text.index("beta")) == 2


# -- normalization happens before offsets exist ----------------------------

def test_ligatures_are_folded():
    assert normalize("the oﬃce ﬁle") == "the office file"


def test_hyphen_line_breaks_are_joined():
    assert normalize("termi-\nnation clause") == "termination clause"


def test_smart_quotes_are_folded():
    assert normalize("the “Agreement” and ‘Term’") == "the \"Agreement\" and 'Term'"


def test_normalize_is_idempotent(all_docs):
    """Downstream must never re-normalize -- but if it did, offsets survive."""
    for doc in all_docs:
        assert normalize(doc.text) == doc.text


# -- the fallback path -----------------------------------------------------

def test_find_span_tolerates_reflowed_whitespace(northwind):
    """Models reflow line breaks inside quotes. We must still ground them."""
    original = "automatically renew for successive twelve (12) month periods"
    reflowed = "automatically renew for successive\n   twelve (12)   month periods"
    span = find_span(northwind, reflowed)
    assert span is not None
    assert span.is_grounded_in(northwind.text)
    assert span.quote == original  # the REAL text, not what the model sent


def test_find_span_rejects_invented_text(northwind):
    assert find_span(northwind, "liability shall not exceed nine million dollars") is None
    assert find_span(northwind, "") is None
    assert find_span(northwind, "   ") is None


def test_contract_type_is_guessed(northwind, nda, amendment2, order_form):
    assert northwind.contract_type == ContractType.MSA
    assert nda.contract_type == ContractType.NDA
    assert amendment2.contract_type == ContractType.AMENDMENT
    assert order_form.contract_type == ContractType.ORDER_FORM
