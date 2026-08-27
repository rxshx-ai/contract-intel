"""The differentiating analyses. All pure functions over the extracted layer."""

from datetime import date

import pytest

from api.findings import (
    detect_adversarial,
    detect_silence,
    find_gaps,
    measure_asymmetry,
    termination_cost,
)
from api.findings.backtoback import exposure_summary
from api.findings.silence import coverage
from api.schemas import ContractType, OurRole


# ===== silence detection ==================================================

def test_missing_liability_cap_is_found_in_the_nda(nda_claims, northwind_contract):
    """The headline: you cannot quote a clause that does not exist."""
    contract = northwind_contract.model_copy(
        update={"id": "k_nda", "our_role": OurRole.MUTUAL})
    findings = detect_silence(nda_claims, contract, ContractType.NDA)
    titles = [f.title for f in findings]
    assert any("limitation of liability" in t for t in titles)
    cap = next(f for f in findings if "limitation of liability" in f.title)
    assert cap.kind == "missing_clause"
    assert cap.evidence == []        # invariant 5
    assert cap.severity == "high"


def test_silence_findings_carry_no_evidence_by_construction(nda_claims,
                                                            northwind_contract):
    findings = detect_silence(nda_claims, northwind_contract, ContractType.NDA)
    assert findings
    assert all(f.evidence == [] and f.kind == "missing_clause" for f in findings)


def test_a_thorough_msa_produces_few_silences(northwind_claims, northwind_contract):
    """No false alarms on a contract that actually covers its bases."""
    findings = detect_silence(northwind_claims, northwind_contract, ContractType.MSA)
    critical = [f for f in findings if f.severity == "critical"]
    assert critical == []
    assert coverage(northwind_claims, ContractType.MSA) > 0.85


def test_findings_are_severity_ranked(nda_claims, northwind_contract):
    findings = detect_silence(nda_claims, northwind_contract, ContractType.NDA)
    order = {"critical": 0, "high": 1, "medium": 2, "low": 3, "info": 4}
    assert [order[f.severity] for f in findings] == \
           sorted(order[f.severity] for f in findings)


def test_seller_is_not_warned_about_clauses_it_owes_outward(acme_claims,
                                                            acme_contract):
    findings = detect_silence(acme_claims, acme_contract, ContractType.MSA)
    assert not any("service level" in f.title for f in findings)


# ===== power asymmetry ====================================================

def test_asymmetry_counts_one_sided_rights(northwind_claims, northwind_contract):
    report, findings = measure_asymmetry(northwind_claims, northwind_contract)
    assert len(report.their_rights) > len(report.our_rights)
    assert report.index > 0.7
    assert findings and findings[0].kind == "asymmetry"


def test_asymmetry_finding_names_the_asks(northwind_claims, northwind_contract):
    _, findings = measure_asymmetry(northwind_claims, northwind_contract)
    asks = findings[0].metadata["asks"]
    assert "amend the agreement unilaterally" in asks
    assert findings[0].evidence  # evidenced, unlike a silence finding


def test_balanced_agreement_raises_no_asymmetry_finding(nda_claims,
                                                        northwind_contract):
    report, findings = measure_asymmetry(nda_claims, northwind_contract)
    assert findings == []
    assert report.index <= 0.7


# ===== adversarial patterns ===============================================

def test_unilateral_amendment_is_flagged_as_a_blank_cheque(northwind_claims,
                                                           northwind_contract):
    findings = detect_adversarial(northwind_claims, northwind_contract)
    blank = [f for f in findings if "blank_cheque" in f.id]
    assert blank and blank[0].severity == "critical"


def test_poisoned_contract_trips_multiple_dark_patterns(poisoned_claims,
                                                        poisoned_contract):
    findings = detect_adversarial(poisoned_claims, poisoned_contract)
    ids = " ".join(f.id for f in findings)
    assert "long_notice" in ids        # 180-day window
    assert "uncapped_pricing" in ids   # sole discretion pricing
    assert "one_way_exit" in ids
    assert len(findings) >= 4


def test_every_adversarial_finding_is_evidenced(northwind_claims, poisoned_claims,
                                                northwind_contract, poisoned_contract):
    for claims, contract in ((northwind_claims, northwind_contract),
                             (poisoned_claims, poisoned_contract)):
        for finding in detect_adversarial(claims, contract):
            assert finding.evidence, f"{finding.id} has no evidence"


def test_clean_nda_trips_no_dark_patterns(nda_claims, northwind_contract):
    assert detect_adversarial(nda_claims, northwind_contract) == []


# ===== back-to-back gaps ==================================================

def test_uptime_flow_down_gap_is_detected(northwind_claims, northwind_contract,
                                          acme_claims, acme_contract):
    """We promise Acme 99.99%; Northwind gives us 99.9%. We own the difference."""
    gaps = find_gaps([(acme_contract, acme_claims),
                      (northwind_contract, northwind_claims)])
    uptime = [g for g in gaps if g.metadata["dimension"] == "uptime"]
    assert len(uptime) == 1
    assert uptime[0].metadata["outbound_value"] == 99.99
    assert uptime[0].metadata["inbound_value"] == 99.9
    assert uptime[0].severity == "critical"
    assert len(uptime[0].evidence) == 2          # one span from EACH contract
    assert len(set(s.doc_id for s in uptime[0].evidence)) == 2


def test_breach_notification_gap_is_detected(northwind_claims, northwind_contract,
                                             acme_claims, acme_contract):
    """We owe Acme 24h; Northwind owes us 72h. Impossible to satisfy."""
    gaps = find_gaps([(acme_contract, acme_claims),
                      (northwind_contract, northwind_claims)])
    breach = [g for g in gaps if g.metadata["dimension"] == "breach_notice"]
    assert breach
    assert "GDPR" in breach[0].explanation


def test_liability_gap_quantifies_uninsured_exposure(northwind_claims,
                                                     northwind_contract,
                                                     acme_claims, acme_contract):
    gaps = find_gaps([(acme_contract, acme_claims),
                      (northwind_contract, northwind_claims)])
    liability = [g for g in gaps if g.metadata["dimension"] == "liability"][0]
    assert liability.metadata["outbound_value"] == 5_000_000
    assert liability.metadata["inbound_value"] == 50_000


def test_data_deletion_gap_is_detected(northwind_claims, northwind_contract,
                                       acme_claims, acme_contract):
    gaps = find_gaps([(acme_contract, acme_claims),
                      (northwind_contract, northwind_claims)])
    assert any(g.metadata["dimension"] == "deletion" for g in gaps)


def test_no_gaps_when_only_one_side_of_the_chain_exists(northwind_claims,
                                                        northwind_contract):
    assert find_gaps([(northwind_contract, northwind_claims)]) == []


def test_exposure_summary_aggregates_the_portfolio(northwind_claims,
                                                   northwind_contract,
                                                   acme_claims, acme_contract):
    gaps = find_gaps([(acme_contract, acme_claims),
                      (northwind_contract, northwind_claims)])
    summary = exposure_summary(gaps)
    assert summary["gap_count"] >= 4
    assert summary["exposed_revenue"] == 640000.0
    assert "uptime" in summary["dimensions"]


# ===== termination cost ===================================================

def _obligations(northwind_rules, contract, today):
    from api.temporal import materialize
    obs, _ = materialize(northwind_rules, contract, today, renewal_months=12)
    return obs


def test_exit_cost_is_itemized_and_totalled(northwind_claims, northwind_rules,
                                            northwind_contract):
    today = date(2026, 8, 27)
    obs = _obligations(northwind_rules, northwind_contract, today)
    cost = termination_cost(northwind_contract, northwind_claims, obs,
                            exit_date=date(2026, 10, 1), today=today)
    labels = [i["label"] for i in cost.line_items]
    assert any("Committed fees" in l for l in labels)
    assert any("Early termination fee" in l for l in labels)
    assert cost.total == round(sum(i["amount"] for i in cost.line_items), 2)
    assert cost.total > 0


def test_open_notice_window_is_reported_as_the_cheaper_route(northwind_claims,
                                                             northwind_rules,
                                                             northwind_contract):
    today = date(2026, 8, 27)
    obs = _obligations(northwind_rules, northwind_contract, today)
    cost = termination_cost(northwind_contract, northwind_claims, obs,
                            exit_date=date(2026, 10, 1), today=today)
    assert any("still open" in n and "reduces this cost to zero" in n
               for n in cost.notes)


def test_missed_notice_window_is_stated_plainly(northwind_claims, northwind_rules,
                                                northwind_contract):
    """After the window closes, the renewal is locked in. Say it in days."""
    today = date(2027, 1, 15)   # past the 2026-12-31 notice deadline
    obs = _obligations(northwind_rules, northwind_contract, today)
    cost = termination_cost(northwind_contract, northwind_claims, obs,
                            exit_date=date(2027, 4, 1), today=today)
    assert any("window has closed" in n for n in cost.notes)


def test_surviving_obligations_are_listed(northwind_claims, northwind_rules,
                                          northwind_contract):
    today = date(2026, 8, 27)
    obs = _obligations(northwind_rules, northwind_contract, today)
    cost = termination_cost(northwind_contract, northwind_claims, obs,
                            exit_date=date(2026, 10, 1), today=today)
    notes = " ".join(cost.notes)
    assert "Data return/deletion obligation: 90 days" in notes
    assert "survives termination" in notes


def test_no_effective_date_means_no_invented_number(northwind_claims,
                                                    northwind_contract):
    contract = northwind_contract.model_copy(update={"effective_date": None})
    cost = termination_cost(contract, northwind_claims, [],
                            exit_date=date(2026, 10, 1), today=date(2026, 8, 27))
    assert cost.total == 0.0
    assert cost.line_items == []
    assert "cannot be computed" in cost.notes[0]
