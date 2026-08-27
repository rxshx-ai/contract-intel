"""Comparison is computed, party-aware, and quotes every figure."""

from datetime import date

import pytest

from api.compare import DIMENSIONS, compare, resolve_dimension


@pytest.fixture(scope="module")
def bundles():
    from api import demo

    return demo.load(date(2026, 8, 27))


def test_dimension_aliases_resolve():
    assert resolve_dimension("liability").key == "liability_cap"
    assert resolve_dimension("SLA").key == "uptime"
    assert resolve_dimension("renewal notice").key == "notice_days"
    assert resolve_dimension("nonsense") is None


def test_unknown_dimension_reports_what_is_available(bundles):
    result = compare(bundles, "colour")
    assert result["ok"] is False
    assert set(result["available"]) == set(DIMENSIONS)


def test_every_figure_carries_its_wording(bundles):
    result = compare(bundles, "liability_cap")
    assert result["rows"]
    for row in result["rows"]:
        assert row["quote"] and row["record_id"]
        assert row["start"] is not None and row["end"] is not None


def test_direction_flips_with_which_side_we_are_on(bundles):
    """A 99.99% uptime commitment is a protection from a supplier and an
    exposure to a customer. One league table across both is meaningless."""
    result = compare(bundles, "uptime")
    buyer = [r for r in result["rows"] if r["side"] == "buyer"][0]
    seller = [r for r in result["rows"] if r["side"] == "seller"][0]
    assert buyer["higher_is_better"] is True
    assert seller["higher_is_better"] is False


def test_verdict_ranks_within_a_side_not_across(bundles):
    verdict = compare(bundles, "liability_cap")["verdict"]
    assert "Supplier side" in verdict and "Customer side" in verdict


def test_silence_is_reported_not_treated_as_a_good_value(bundles):
    result = compare(bundles, "liability_cap")
    assert "Vertex Cloud Systems Inc." in result["not_stated"]
    assert "Silence is not a good value" in result["not_stated_note"]


def test_notice_days_comes_from_the_verified_rule(bundles):
    """The clause label was wrong on live output; the temporal rule was not."""
    rows = {r["contract"]: r["value"] for r in compare(bundles, "notice_days")["rows"]}
    assert rows["Northwind Observability, Inc."] == 60
    assert rows["Vertex Cloud Systems Inc."] == 180


def test_scope_to_named_contracts(bundles):
    result = compare(bundles, "uptime", contract_ids=["k_acme"])
    assert [r["contract_id"] for r in result["rows"]] == ["k_acme"]
