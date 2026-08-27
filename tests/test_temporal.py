"""Invariant 2: dates are computed, never generated. No network, no model."""

from datetime import date

import pytest

from api.schemas import Contract, OurRole, Span, TemporalRule
from api.temporal import (
    add_months,
    materialize,
    next_deadline,
    parse_recurrence_months,
    resolve_term_end,
)

SPAN = Span(doc_id="d1", char_start=0, char_end=4, quote="MAST")


def _contract(effective=date(2026, 3, 1)):
    return Contract(id="k1", title="Northwind MSA", counterparty="Northwind",
                    our_role=OurRole.BUYER, effective_date=effective,
                    annual_value=84000.0)


def _rule(**kw):
    base = dict(id="tr1", contract_id="k1", kind="notice", anchor="term_end",
                offset_days=-60, span=SPAN, owed_by="us")
    base.update(kw)
    return TemporalRule(**base)


# -- arithmetic ------------------------------------------------------------

def test_add_months_clamps_short_months():
    assert add_months(date(2026, 1, 31), 1) == date(2026, 2, 28)
    assert add_months(date(2024, 1, 31), 1) == date(2024, 2, 29)  # leap year


def test_add_months_crosses_years():
    assert add_months(date(2026, 11, 15), 14) == date(2028, 1, 15)


def test_parse_recurrence():
    assert parse_recurrence_months("P12M") == 12
    assert parse_recurrence_months("P1Y") == 12
    assert parse_recurrence_months("P3M") == 3
    assert parse_recurrence_months(None) is None
    assert parse_recurrence_months("banana") is None


# -- term rolling ----------------------------------------------------------

def test_term_end_within_initial_term():
    end, renewals, steps = resolve_term_end(
        date(2026, 3, 1), 12, 12, today=date(2026, 8, 27))
    assert end == date(2027, 3, 1)
    assert renewals == 0
    assert any("initial term" in s.lower() for s in steps)


def test_term_rolls_through_elapsed_renewals():
    """The failure mode this whole module exists to prevent."""
    end, renewals, steps = resolve_term_end(
        date(2020, 3, 1), 12, 12, today=date(2026, 8, 27))
    assert end == date(2027, 3, 1)
    assert renewals == 6
    assert "Auto-renewed 6x" in " ".join(steps)


def test_no_renewal_clause_means_term_simply_ends():
    end, renewals, _ = resolve_term_end(
        date(2020, 3, 1), 12, None, today=date(2026, 8, 27))
    assert end == date(2021, 3, 1)
    assert renewals == 0


# -- the headline case -----------------------------------------------------

def test_notice_deadline_derived_from_relative_language():
    """'60 days prior to the end of the then-current Term' -> a real date."""
    today = date(2026, 8, 27)
    obs, unresolved = materialize(
        [_rule()], _contract(), today, initial_term_months=12, renewal_months=12)
    assert unresolved == []
    assert len(obs) == 1
    assert obs[0].due_date == date(2026, 12, 31)   # 2027-03-01 minus 60 days
    assert obs[0].days_remaining(today) == 126


def test_derivation_chain_is_auditable():
    obs, _ = materialize([_rule()], _contract(), date(2026, 8, 27),
                         renewal_months=12)
    chain = " | ".join(obs[0].derivation)
    assert "Effective Date = 2026-03-01" in chain
    assert "60 days before" in chain
    assert "2026-12-31" in chain


def test_renewal_months_inferred_from_the_renewal_rule():
    """Caller does not have to supply the renewal period; it is extracted."""
    rules = [
        _rule(id="r_renew", kind="renewal", offset_days=0, recurrence="P12M"),
        _rule(id="r_notice"),
    ]
    obs, _ = materialize(rules, _contract(date(2020, 3, 1)), date(2026, 8, 27))
    notice = [o for o in obs if o.rule_id == "r_notice"][0]
    assert notice.due_date == date(2026, 12, 31)  # proves the term rolled forward


# -- honest failure --------------------------------------------------------

def test_event_anchored_rules_are_reported_not_invented():
    """A cure period has no date until a breach happens. Say so."""
    obs, unresolved = materialize(
        [_rule(kind="cure", anchor="breach_date", offset_days=30)],
        _contract(), date(2026, 8, 27), renewal_months=12)
    assert obs == []
    assert len(unresolved) == 1
    assert "event-driven" in unresolved[0]


def test_missing_effective_date_yields_no_dates():
    obs, unresolved = materialize([_rule()], _contract(effective=None),
                                  date(2026, 8, 27))
    assert obs == []
    assert "No Effective Date" in unresolved[0]


# -- recurrence ------------------------------------------------------------

def test_quarterly_report_obligation_recurs():
    rules = [_rule(kind="report", anchor="effective_date", offset_days=15,
                   recurrence="P3M", owed_by="us")]
    obs, _ = materialize(rules, _contract(), date(2026, 8, 27),
                         renewal_months=12, horizon_days=365)
    assert len(obs) >= 4
    assert all(o.kind == "report" for o in obs)
    gaps = [(b.due_date - a.due_date).days for a, b in zip(obs, obs[1:])]
    assert all(85 <= g <= 95 for g in gaps)


def test_obligations_are_sorted_and_next_deadline_picks_the_soonest():
    today = date(2026, 8, 27)
    rules = [
        _rule(id="r1", kind="notice", offset_days=-60),
        _rule(id="r2", kind="report", anchor="effective_date", offset_days=15),
    ]
    obs, _ = materialize(rules, _contract(), today, renewal_months=12)
    assert obs == sorted(obs, key=lambda o: o.due_date)
    nxt = next_deadline(obs, today)
    assert nxt is not None and nxt.due_date >= today


# -- implied recurrence is deliberately narrow ----------------------------

def test_recurring_report_anchored_to_quarter_end_recurs():
    obs, _ = materialize(
        [_rule(kind="report", anchor="quarter_end", offset_days=15, recurrence=None)],
        _contract(), date(2026, 8, 27), renewal_months=12, horizon_days=365)
    assert len(obs) >= 3
    assert all(o.kind == "report" for o in obs)


def test_one_off_notice_is_not_multiplied_into_phantom_deadlines():
    """A single mis-modelled notice must not become a calendar of fake rows."""
    obs, _ = materialize(
        [_rule(kind="notice", anchor="month_end", offset_days=-30, recurrence=None)],
        _contract(), date(2026, 8, 27), renewal_months=12, horizon_days=730)
    assert len(obs) == 1


def test_event_anchored_rule_is_surfaced_as_conditional_not_dropped():
    obs, unresolved = materialize(
        [_rule(kind="notice", anchor="event", offset_days=-30)],
        _contract(), date(2026, 8, 27), renewal_months=12)
    assert obs == []
    assert len(unresolved) == 1
    assert "event-driven" in unresolved[0]
    assert "MAST" in unresolved[0]      # the quote travels with the reason


def test_quarter_and_month_ends_are_calendar_correct():
    from api.temporal import next_month_end, next_quarter_end

    assert next_month_end(date(2026, 8, 27)) == date(2026, 8, 31)
    assert next_month_end(date(2026, 12, 5)) == date(2026, 12, 31)
    assert next_quarter_end(date(2026, 8, 27)) == date(2026, 9, 30)
    assert next_quarter_end(date(2026, 10, 1)) == date(2026, 12, 31)
