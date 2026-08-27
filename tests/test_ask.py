"""Ask answers from the extracted layer only. No network in any test here."""

from datetime import date

import pytest

from api import ask as ask_mod
from api.ask import Index, Record, build_records, tokenize
from api.pipeline import analyze_portfolio


@pytest.fixture(scope="module")
def portfolio():
    from api import demo

    bundles = demo.load(date(2026, 8, 27))
    gaps = analyze_portfolio(bundles)
    return bundles, gaps


@pytest.fixture(scope="module")
def records(portfolio):
    bundles, gaps = portfolio
    return build_records(bundles, gaps, date(2026, 8, 27))


@pytest.fixture(scope="module")
def index(records):
    return Index(records)


# ── the index ────────────────────────────────────────────────────────────

def test_every_record_type_is_indexed(records):
    kinds = {r.kind for r in records}
    assert kinds == {"clause", "obligation", "absence", "finding", "gap", "fact"}


def test_records_carry_their_verified_provenance(records):
    quoted = [r for r in records if r.quote]
    assert quoted
    for record in quoted:
        if record.kind == "gap":
            continue          # gap quotes span two contracts
        assert record.src_file
        assert record.src_start is not None and record.src_end is not None


def test_absence_records_have_no_quote(records):
    """Invariant 5 again: you cannot quote a clause that isn't there."""
    absences = [r for r in records if r.kind == "absence"]
    assert absences
    assert all(r.quote is None for r in absences)


def test_tokenizer_drops_stopwords():
    assert "the" not in tokenize("the liability cap")
    assert "liability" in tokenize("the liability cap")


# ── retrieval ────────────────────────────────────────────────────────────

def test_renewal_question_ranks_the_renewal_deadline_first(index):
    """Not a quarterly report that merely shares the word 'deadline'."""
    top = index.rank("when does northwind renew?", top_k=1)[0][0]
    assert top.kind == "obligation"
    assert top.meta["anchor"] == "term_end"
    assert "Northwind" in top.contract


def test_exposure_question_surfaces_cross_contract_gaps(index):
    kinds = [r.kind for r, _ in index.rank("where are we exposed across contracts?",
                                           top_k=4)]
    assert kinds.count("gap") >= 3


def test_absence_question_surfaces_absences(index):
    top = index.rank("what is missing from the helios nda?", top_k=2)
    assert all(r.kind == "absence" for r, _ in top)


def test_contract_scope_filters_results(index):
    hits = index.rank("liability cap", contract_id="k_helios")
    assert hits
    assert all(r.contract_id == "k_helios" for r, _ in hits)


def test_unrelated_question_returns_weak_or_no_hits(index):
    hits = index.rank("zqxjkv unrelated nonsense token", top_k=5)
    assert hits == []


# ── answering ────────────────────────────────────────────────────────────

def _fake(monkeypatch, **fields):
    def fake_complete_json(system, user, model, **kw):
        return ask_mod.RawAnswer(**fields)

    monkeypatch.setattr(ask_mod, "complete_json", fake_complete_json)


def test_answer_resolves_citations_to_real_records(index, monkeypatch):
    real_id = index.rank("when does northwind renew?", top_k=1)[0][0].id
    _fake(monkeypatch, answer="It renews in March.",
          cited_record_ids=[real_id], sufficient=True)
    result = ask_mod.ask("when does northwind renew?", index)
    assert result.sufficient
    assert len(result.citations) == 1
    assert result.citations[0]["record_id"] == real_id


def test_fabricated_citation_ids_are_dropped(index, monkeypatch):
    """The model emits ids, not quotes — an invented id cites nothing."""
    _fake(monkeypatch, answer="Something.",
          cited_record_ids=["totally-made-up-id", "another-fake"],
          sufficient=True)
    result = ask_mod.ask("liability cap", index)
    assert result.citations == []


def test_insufficient_records_are_reported_not_papered_over(index, monkeypatch):
    """Records were retrieved, but none answer the question. Say so."""
    _fake(monkeypatch, answer=None, cited_record_ids=[], sufficient=False,
          missing="the CEO's name")
    result = ask_mod.ask("who is the CEO of northwind?", index)
    assert result.considered > 0          # the model WAS consulted
    assert result.sufficient is False
    assert "cannot answer" in result.answer or "not something" in result.answer
    assert result.missing == "the CEO's name"


def test_no_matching_records_short_circuits_without_calling_the_model(index,
                                                                     monkeypatch):
    def explode(*a, **k):
        raise AssertionError("the model must not be called with zero records")

    monkeypatch.setattr(ask_mod, "complete_json", explode)
    result = ask_mod.ask("zqxjkv unrelated nonsense token", index)
    assert result.sufficient is False
    assert result.citations == []
    assert result.considered == 0


def test_citations_carry_the_quote_and_offsets(index, monkeypatch):
    hit = next(r for r, _ in index.rank("liability cap", top_k=8)
               if r.kind == "clause")
    _fake(monkeypatch, answer="x", cited_record_ids=[hit.id], sufficient=True)
    citation = ask_mod.ask("liability cap", index).citations[0]
    assert citation["quote"] == hit.quote
    assert citation["start"] == hit.src_start
    assert citation["file"] == hit.src_file


def test_records_sent_to_the_model_never_include_raw_document_text(index):
    """Ask reads the extracted layer, not the contract."""
    for record, _ in index.rank("liability cap", top_k=6):
        payload = record.for_model()
        assert set(payload) <= {"id", "contract", "kind", "title", "detail",
                                "verbatim_quote"}
        if "verbatim_quote" in payload:
            assert len(payload["verbatim_quote"]) <= ask_mod.MAX_QUOTE_CHARS
