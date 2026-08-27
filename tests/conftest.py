import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pytest

from api.ingest import ingest_text

CONTRACTS = ROOT / "contracts"


def _load(name: str):
    return ingest_text((CONTRACTS / name).read_text(), name)


@pytest.fixture(scope="session")
def northwind():
    return _load("msa_northwind.txt")


@pytest.fixture(scope="session")
def amendment2():
    return _load("amendment_2_northwind.txt")


@pytest.fixture(scope="session")
def order_form():
    return _load("order_form_northwind.txt")


@pytest.fixture(scope="session")
def acme():
    return _load("customer_msa_acme.txt")


@pytest.fixture(scope="session")
def nda():
    return _load("nda_helios.txt")


@pytest.fixture(scope="session")
def poisoned():
    return _load("poisoned_msa_vertex.txt")


@pytest.fixture(scope="session")
def all_docs(northwind, amendment2, order_form, acme, nda, poisoned):
    return [northwind, amendment2, order_form, acme, nda, poisoned]


# -- grounded extraction fixtures ------------------------------------------

from datetime import date

from api.extract import RawExtraction, ground_clauses, ground_rules
from api.schemas import Contract, OurRole

FIXTURES = ROOT / "contracts" / "fixtures"


def _raw(name: str) -> RawExtraction:
    return RawExtraction.model_validate_json((FIXTURES / name).read_text())


@pytest.fixture(scope="session")
def northwind_raw():
    return _raw("msa_northwind.json")


@pytest.fixture(scope="session")
def northwind_claims(northwind_raw, northwind):
    claims, stats = ground_clauses(northwind_raw, northwind, "k_nw")
    assert stats.dropped == 0, "fixture must ground cleanly"
    return claims


@pytest.fixture(scope="session")
def northwind_rules(northwind_raw, northwind):
    rules, stats = ground_rules(northwind_raw, northwind, "k_nw")
    assert stats.dropped == 0
    return rules


@pytest.fixture(scope="session")
def northwind_contract():
    return Contract(id="k_nw", title="Northwind Observability MSA",
                    counterparty="Northwind Observability, Inc.",
                    our_role=OurRole.BUYER, effective_date=date(2026, 3, 1),
                    annual_value=84000.0)


@pytest.fixture(scope="session")
def acme_claims(acme):
    claims, stats = ground_clauses(_raw("customer_msa_acme.json"), acme, "k_acme")
    assert stats.dropped == 0
    return claims


@pytest.fixture(scope="session")
def acme_contract():
    return Contract(id="k_acme", title="Acme Master Subscription Agreement",
                    counterparty="Acme Financial Group plc",
                    our_role=OurRole.SELLER, effective_date=date(2026, 5, 1),
                    annual_value=640000.0)


@pytest.fixture(scope="session")
def nda_claims(nda):
    claims, stats = ground_clauses(_raw("nda_helios.json"), nda, "k_nda")
    assert stats.dropped == 0
    return claims


@pytest.fixture(scope="session")
def poisoned_claims(poisoned):
    claims, stats = ground_clauses(_raw("poisoned_msa_vertex.json"), poisoned, "k_px")
    assert stats.dropped == 0
    return claims


@pytest.fixture(scope="session")
def poisoned_contract():
    return Contract(id="k_px", title="Vertex Cloud MSA", counterparty="Vertex Cloud Systems Inc.",
                    our_role=OurRole.BUYER, effective_date=date(2026, 7, 10),
                    annual_value=420000.0)


TODAY = date(2026, 8, 27)
