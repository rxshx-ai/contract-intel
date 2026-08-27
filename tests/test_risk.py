"""Invariant 2: scores are computed from a published rubric, never generated."""

from api.risk import RUBRIC, band, score


def _axis(profile, name):
    return next(a for a in profile.axes if a.axis == name)


def test_every_point_traces_to_a_reason(northwind_claims, northwind_contract):
    profile = score(northwind_claims, northwind_contract)
    for axis in profile.axes:
        assert axis.score == min(100, sum(c.points for c in axis.contributions))
        for contribution in axis.contributions:
            assert contribution.reason.strip()
            assert contribution.points > 0


def test_scoring_is_deterministic(northwind_claims, northwind_contract):
    a = score(northwind_claims, northwind_contract)
    b = score(northwind_claims, northwind_contract)
    assert [x.score for x in a.axes] == [x.score for x in b.axes]


def test_low_cap_relative_to_contract_value_scores_liability(
    northwind_claims, northwind_contract
):
    """$50k cap against $84k annual spend: below annual value, above half of it."""
    liability = _axis(score(northwind_claims, northwind_contract), "liability")
    assert liability.score >= RUBRIC["cap_below_annual_value"]
    assert any("is below the annual contract value" in c.reason
               for c in liability.contributions)
    assert not any("under half" in c.reason for c in liability.contributions)


def test_cap_is_judged_against_contract_value_not_an_absolute(northwind_claims,
                                                              northwind_contract):
    """The same $50k cap is fine on a $20k contract and alarming on a $2M one."""
    small = northwind_contract.model_copy(update={"annual_value": 40000.0})
    large = northwind_contract.model_copy(update={"annual_value": 2000000.0})
    assert _axis(score(northwind_claims, small), "liability").score < \
           _axis(score(northwind_claims, large), "liability").score


def test_unilateral_amendment_is_scored(northwind_claims, northwind_contract):
    operational = _axis(score(northwind_claims, northwind_contract), "operational")
    assert any("change the terms unilaterally" in c.reason
               for c in operational.contributions)


def test_lockin_captures_one_sided_termination(northwind_claims, northwind_contract):
    lockin = _axis(score(northwind_claims, northwind_contract), "lockin")
    reasons = " ".join(c.reason for c in lockin.contributions)
    assert "terminate for convenience and we may not" in reasons
    assert "no right to terminate for convenience" in reasons


def test_overall_is_worst_axis_not_mean(northwind_claims, northwind_contract):
    """A single catastrophic axis must not be averaged into comfort."""
    profile = score(northwind_claims, northwind_contract)
    assert profile.overall == max(a.score for a in profile.axes)


def test_role_changes_the_verdict(acme_claims, acme_contract):
    """Same rubric, other side of the paper: we are the seller here."""
    profile = score(acme_claims, acme_contract)
    assert profile.our_role.value == "seller"
    # Our $5M cap against $640k revenue is not a cap risk for us.
    assert not any("under half the annual contract value" in c.reason
                   for c in _axis(profile, "liability").contributions)


def test_poisoned_contract_scores_high_despite_its_instructions(
    poisoned_claims, poisoned_contract
):
    """The document tells the reader to report risk 0. The rubric does not care."""
    profile = score(poisoned_claims, poisoned_contract)
    assert profile.overall >= 70
    assert band(profile.overall) == "critical"


def test_clean_contract_does_not_manufacture_risk(nda_claims, northwind_contract):
    """An unremarkable mutual NDA should not light up the board."""
    contract = northwind_contract.model_copy(
        update={"id": "k_nda", "annual_value": None})
    profile = score(nda_claims, contract)
    assert profile.overall <= 45


def test_superseded_clauses_are_excluded(northwind_claims, northwind_contract):
    before = _axis(score(northwind_claims, northwind_contract), "liability").score
    patched = [
        c.model_copy(update={"superseded_by": "amend2"})
        if c.clause_type.value == "limitation_of_liability" else c
        for c in northwind_claims
    ]
    after = _axis(score(patched, northwind_contract), "liability").score
    assert after < before


def test_band_thresholds():
    assert band(80) == "critical"
    assert band(50) == "high"
    assert band(25) == "medium"
    assert band(5) == "low"
