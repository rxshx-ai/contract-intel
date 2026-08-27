"""End-to-end through the HTTP surface."""

from datetime import date

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client():
    from api import main

    main.load_demo(date(2026, 8, 27))
    with TestClient(main.app) as c:
        yield c


def test_portfolio_stats_report_grounding(client):
    stats = client.get("/portfolio/stats").json()
    assert stats["contracts"] == 4
    assert stats["documents"] == 6
    assert stats["grounding_rate"] == 1.0
    assert stats["hallucination_rate"] == 0.0
    assert stats["injections_detected"] >= 5


def test_contract_list_is_risk_scored(client):
    rows = client.get("/contracts").json()
    assert {r["id"] for r in rows} == {"k_northwind", "k_acme", "k_vertex", "k_helios"}
    vertex = next(r for r in rows if r["id"] == "k_vertex")
    assert vertex["band"] == "critical"


def test_every_returned_clause_quote_is_grounded(client):
    """The HTTP layer cannot leak an ungrounded quote."""
    detail = client.get("/contracts/k_northwind").json()
    docs = {d["id"]: d for d in detail["documents"]}
    clauses = client.get("/contracts/k_northwind/clauses?effective_only=false").json()
    assert clauses
    for clause in clauses:
        assert clause["span"]["doc_id"] in docs
        assert clause["span"]["quote"]


def test_notice_deadline_is_derived_with_its_chain(client):
    body = client.get("/contracts/k_northwind/obligations").json()
    notice = next(o for o in body["obligations"] if o["kind"] == "notice")
    assert notice["due_date"] == "2026-12-31"
    assert notice["days_remaining"] == 126
    assert any("Effective Date = 2026-03-01" in s for s in notice["derivation"])


def test_risk_endpoint_publishes_its_rubric(client):
    body = client.get("/contracts/k_northwind/risk").json()
    assert body["rubric"]["unilateral_amendment"] == 30
    assert body["overall"] == max(a["score"] for a in body["profile"]["axes"])


def test_findings_include_absence_and_cross_contract_gaps(client):
    findings = client.get("/contracts/k_acme/findings").json()
    kinds = {f["kind"] for f in findings}
    assert "missing_clause" in kinds
    assert "backtoback_gap" in kinds
    for f in findings:
        if f["kind"] != "missing_clause":
            assert f["evidence"], f"{f['id']} unevidenced"


def test_portfolio_gaps_span_two_contracts(client):
    body = client.get("/portfolio/gaps").json()
    assert body["summary"]["gap_count"] >= 4
    for gap in body["gaps"]:
        assert len(gap["contract_ids"]) == 2
        assert len({e["doc_id"] for e in gap["evidence"]}) == 2


def test_deadlines_are_sorted_by_urgency(client):
    rows = client.get("/portfolio/deadlines?within_days=400").json()
    assert rows == sorted(rows, key=lambda r: r["days_remaining"])


def test_termination_cost_is_itemized(client):
    cost = client.post("/contracts/k_northwind/termination-cost",
                       data={"exit_date": "2026-10-01"}).json()
    assert cost["total"] == 52500.0
    assert len(cost["line_items"]) == 2


def test_bad_input_is_rejected(client):
    assert client.post("/contracts/k_northwind/termination-cost",
                       data={"exit_date": "not-a-date"}).status_code == 400
    assert client.get("/contracts/k_missing").status_code == 404


def test_ics_feed_is_valid_and_has_alarms(client):
    body = client.get("/portfolio/deadlines.ics").text
    assert body.startswith("BEGIN:VCALENDAR")
    assert body.rstrip().endswith("END:VCALENDAR")
    assert body.count("BEGIN:VEVENT") == body.count("END:VEVENT") > 0
    assert "TRIGGER:-P14D" in body


def test_audit_log_records_views(client):
    client.get("/contracts/k_helios")
    entries = client.get("/audit?limit=50").json()
    assert any(e["action"] == "view" and e["subject_id"] == "k_helios" for e in entries)


def test_upload_reports_firewall_verdict(client):
    poisoned = open("contracts/poisoned_msa_vertex.txt", "rb").read()
    response = client.post("/documents",
                           files={"file": ("vendor_msa.txt", poisoned, "text/plain")})
    body = response.json()
    assert body["firewall"]["quarantined"] is True
    assert body["contract_type"] == "msa"

    clean = open("contracts/nda_helios.txt", "rb").read()
    body = client.post("/documents",
                       files={"file": ("nda.txt", clean, "text/plain")}).json()
    assert body["firewall"]["quarantined"] is False


def test_uncached_upload_fails_loudly_rather_than_guessing(client):
    """No API key and no cache must produce an error, never a fabricated result."""
    response = client.post(
        "/contracts",
        files={"files": ("novel.txt", b"MASTER SERVICES AGREEMENT\n\n1. Fees are due.",
                         "text/plain")},
        data={"title": "Novel", "counterparty": "X", "our_role": "buyer"},
    )
    assert response.status_code == 503
    assert "Extraction unavailable" in response.json()["detail"]
