"""A contract is a stack of documents. The effective value is what matters."""

from datetime import date

import pytest

from api.extract import RawExtraction, ground_clauses
from api.family import (
    amendment_rank,
    build_contract,
    effective,
    lineage_text,
    order_documents,
    parse_date,
    resolve_supersession,
)
from api.schemas import ClauseType, ContractType, OurRole
from tests.conftest import FIXTURES


@pytest.fixture(scope="session")
def family(northwind, amendment2, order_form):
    docs = [northwind, amendment2, order_form]
    claims_by_doc = {}
    for doc, name in ((northwind, "msa_northwind.json"),
                      (amendment2, "amendment_2_northwind.json"),
                      (order_form, "order_form_northwind.json")):
        raw = RawExtraction.model_validate_json((FIXTURES / name).read_text())
        claims, dropped = ground_clauses(raw, doc, "k_nw")
        assert dropped == 0
        claims_by_doc[doc.id] = claims
    claims, lineage = resolve_supersession(claims_by_doc, docs)
    return docs, claims, lineage


def test_dates_parse_in_several_formats():
    assert parse_date("as of 1 June 2026 by and between") == date(2026, 6, 1)
    assert parse_date("dated March 15, 2026") == date(2026, 3, 15)
    assert parse_date("on 2026-03-01 the parties") == date(2026, 3, 1)
    assert parse_date("no date here") is None


def test_amendments_sort_after_base_documents(northwind, amendment2, order_form):
    ordered = order_documents([amendment2, northwind, order_form])
    assert ordered[-1].contract_type == ContractType.AMENDMENT


def test_amendment_number_drives_ordering(amendment2):
    assert amendment_rank(amendment2)[0] == 2


def test_amended_liability_cap_supersedes_the_original(family):
    """The MSA says $50k. Amendment No. 2 says $250k. $250k is the truth."""
    _, claims, _ = family
    caps = [c for c in claims if c.clause_type == ClauseType.LIABILITY_CAP]
    assert len(caps) == 2
    live = [c for c in caps if c.effective]
    assert len(live) == 1
    assert live[0].fields["amount"] == 250000.0
    dead = [c for c in caps if not c.effective]
    assert dead[0].fields["amount"] == 50000.0
    assert dead[0].superseded_by == live[0].id


def test_amended_payment_terms_supersede(family):
    _, claims, _ = family
    live = [c for c in effective(claims) if c.clause_type == ClauseType.PAYMENT_TERMS]
    assert len(live) == 1
    assert live[0].fields["days"] == 30       # was 45 in the MSA


def test_clauses_the_amendment_is_silent_on_remain_in_force(family):
    """An amendment that says nothing about SLAs does not delete the SLA."""
    _, claims, _ = family
    slas = [c for c in effective(claims) if c.clause_type == ClauseType.SLA]
    assert len(slas) == 1
    assert slas[0].fields["uptime_percent"] == 99.9


def test_lineage_is_human_readable(family):
    docs, claims, _ = family
    by_id = {d.id: d for d in docs}
    live_cap = next(c for c in effective(claims)
                    if c.clause_type == ClauseType.LIABILITY_CAP)
    text = lineage_text(live_cap, claims, by_id)
    assert "set by amendment_2_northwind.txt" in text
    assert "supersedes msa_northwind.txt" in text


def test_effective_date_comes_from_the_order_form(family):
    """The MSA defers to the Order Form -- the single most common reason
    single-document analysis produces the wrong renewal date."""
    docs, claims, _ = family
    contract = build_contract("k_nw", "Northwind MSA", docs, claims,
                              "Northwind Observability, Inc.", OurRole.BUYER)
    assert contract.effective_date == date(2026, 3, 1)
    assert contract.annual_value == 84000.0
    assert contract.contract_type == ContractType.MSA
    assert len(contract.doc_ids) == 3


def test_risk_uses_the_amended_cap_not_the_original(family):
    """Proof the whole chain honours supersession."""
    from api.risk import score

    docs, claims, _ = family
    contract = build_contract("k_nw", "Northwind MSA", docs, claims,
                              "Northwind Observability, Inc.", OurRole.BUYER)
    liability = next(a for a in score(claims, contract).axes if a.axis == "liability")
    # $250k cap against $84k annual value is no longer a cap finding at all.
    assert not any("annual contract value" in c.reason for c in liability.contributions)
