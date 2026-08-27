"""The temporal obligation compiler. Invariant 2.

Contracts almost never contain deadlines. They contain RULES:

    "...unless either party provides written notice of non-renewal no less
     than sixty (60) days prior to the end of the then-current Term."

There is no date in that sentence. This module turns the rule into a real
calendar date by pure arithmetic, and records every step of the derivation so
the number can be audited rather than trusted. No model is involved.
"""

from __future__ import annotations

import calendar
import re
from datetime import date, timedelta

from api.schemas import Contract, Obligation, TemporalRule

DEFAULT_HORIZON_DAYS = 730
MAX_OCCURRENCES_PER_RULE = 4


def add_months(start: date, months: int) -> date:
    """Calendar-correct month arithmetic, clamping to the end of short months."""
    total = start.month - 1 + months
    year = start.year + total // 12
    month = total % 12 + 1
    day = min(start.day, calendar.monthrange(year, month)[1])
    return date(year, month, day)


def parse_recurrence_months(recurrence: str | None) -> int | None:
    """'P12M' -> 12. Years are folded into months."""
    if not recurrence:
        return None
    match = re.fullmatch(r"P(?:(\d+)Y)?(?:(\d+)M)?", recurrence.strip().upper())
    if not match or not any(match.groups()):
        return None
    years = int(match.group(1) or 0)
    months = int(match.group(2) or 0)
    return years * 12 + months or None


# --------------------------------------------------------------------------
# term rolling
# --------------------------------------------------------------------------

def resolve_term_end(
    effective: date,
    initial_term_months: int,
    renewal_months: int | None,
    today: date,
) -> tuple[date, int, list[str]]:
    """Roll the term forward through elapsed auto-renewals.

    Returns (end of the CURRENT term, renewals elapsed, derivation steps).
    This is where 'we thought we were still in the initial term' goes wrong in
    real life, so the count is reported rather than assumed.
    """
    steps = [
        f"Effective Date = {effective.isoformat()} (from Order Form)",
        f"Initial Term = {initial_term_months} months "
        f"-> initial term ends {add_months(effective, initial_term_months).isoformat()}",
    ]
    term_end = add_months(effective, initial_term_months)
    renewals = 0
    if renewal_months:
        while term_end <= today:
            term_end = add_months(term_end, renewal_months)
            renewals += 1
        if renewals:
            steps.append(
                f"Auto-renewed {renewals}x for {renewal_months} months "
                f"-> current term ends {term_end.isoformat()}"
            )
        else:
            steps.append(f"Still in initial term; current term ends {term_end.isoformat()}")
    return term_end, renewals, steps


def next_month_end(today: date) -> date:
    year, month = (today.year + 1, 1) if today.month == 12 else (today.year, today.month + 1)
    return date(year, month, 1) - timedelta(days=1)


def next_quarter_end(today: date) -> date:
    month = ((today.month - 1) // 3 + 1) * 3
    year = today.year
    end = date(year, month, calendar.monthrange(year, month)[1])
    if end < today:
        month += 3
        if month > 12:
            month, year = month - 12, year + 1
        end = date(year, month, calendar.monthrange(year, month)[1])
    return end


# Anchors that resolve to a calendar date, and their natural recurrence.
_IMPLIED_RECURRENCE = {"anniversary": 12, "quarter_end": 3, "month_end": 1}

EVENT_ANCHORS = {"invoice_date", "breach_date", "event"}


def _anchor_date(
    anchor: str, contract: Contract, term_end: date | None, today: date
) -> tuple[date | None, str]:
    if anchor == "effective_date":
        return contract.effective_date, "Effective Date"
    if anchor == "signature_date":
        return contract.effective_date, "Signature Date (using Effective Date)"
    if anchor == "anniversary":
        return contract.effective_date, "anniversary of the Effective Date"
    if anchor in ("term_end", "expiry"):
        return term_end, "end of the then-current Term"
    if anchor == "month_end":
        return next_month_end(today), "end of the calendar month"
    if anchor == "quarter_end":
        return next_quarter_end(today), "end of the calendar quarter"
    return None, anchor  # event-driven


# --------------------------------------------------------------------------
# materialization
# --------------------------------------------------------------------------

def materialize(
    rules: list[TemporalRule],
    contract: Contract,
    today: date,
    initial_term_months: int = 12,
    renewal_months: int | None = None,
    horizon_days: int = DEFAULT_HORIZON_DAYS,
    max_occurrences: int = MAX_OCCURRENCES_PER_RULE,
) -> tuple[list[Obligation], list[str]]:
    """Turn temporal rules into dated obligations.

    Returns (obligations, unresolved-reasons). A rule anchored to an event we
    have not observed (an invoice, a breach) is reported as unresolved rather
    than given a made-up date.
    """
    unresolved: list[str] = []
    obligations: list[Obligation] = []

    if contract.effective_date is None:
        return [], ["No Effective Date found; no deadline can be derived."]

    if renewal_months is None:
        for rule in rules:
            if rule.kind == "renewal":
                renewal_months = parse_recurrence_months(rule.recurrence)
                if renewal_months:
                    break

    term_end, renewals, term_steps = resolve_term_end(
        contract.effective_date, initial_term_months, renewal_months, today
    )
    horizon = today + timedelta(days=horizon_days)

    for rule in rules:
        anchor_date, anchor_label = _anchor_date(rule.anchor, contract, term_end, today)
        if anchor_date is None:
            unresolved.append(
                f"{rule.kind}: anchored to '{rule.anchor}', which is event-driven "
                f"and has not occurred. Deadline is conditional, not calendar-based. "
                f"\"{rule.span.quote[:110]}\""
            )
            continue

        recurrence = parse_recurrence_months(rule.recurrence)
        # A quarterly report anchored to quarter_end recurs quarterly even when
        # the model did not say so. Only RECURRING DELIVERABLES get this: a
        # one-off notice must never be multiplied into a calendar full of
        # phantom deadlines just because it references a month end.
        if recurrence is None and rule.kind == "report":
            recurrence = _IMPLIED_RECURRENCE.get(rule.anchor)
        occurrences = _occurrences(
            anchor_date, rule.offset_days, recurrence, today, horizon)[:max_occurrences]

        for due in occurrences:
            steps = list(term_steps)
            sign = "before" if rule.offset_days < 0 else "after"
            if rule.offset_days:
                steps.append(
                    f"Rule: {abs(rule.offset_days)} days {sign} the {anchor_label} "
                    f"({anchor_date.isoformat()})"
                )
            steps.append(f"=> deadline {due.isoformat()} ({(due - today).days} days from {today.isoformat()})")
            if rule.condition:
                steps.append(f"Condition: {rule.condition}")

            obligations.append(
                Obligation(
                    rule_id=rule.id,
                    contract_id=contract.id,
                    kind=rule.kind,
                    anchor=rule.anchor,
                    due_date=due,
                    owed_by=rule.owed_by,
                    description=rule.consequence or f"{rule.kind} deadline",
                    derivation=steps,
                    consequence_if_missed=rule.consequence,
                )
            )

    obligations.sort(key=lambda o: o.due_date)
    return obligations, unresolved


def _occurrences(
    anchor: date, offset_days: int, recurrence: int | None, today: date, horizon: date
) -> list[date]:
    """Every due date inside the horizon. Past deadlines are kept when they
    are the most recent one -- a missed notice window is the finding."""
    first = anchor + timedelta(days=offset_days)
    if recurrence is None:
        return [first] if first <= horizon else []

    dates: list[date] = []
    cursor = first
    guard = 0
    while cursor <= horizon and guard < 200:
        if cursor >= today - timedelta(days=90):
            dates.append(cursor)
        cursor = add_months(cursor, recurrence)
        guard += 1
    return dates


def next_deadline(obligations: list[Obligation], today: date) -> Obligation | None:
    future = [o for o in obligations if o.due_date >= today]
    return min(future, key=lambda o: o.due_date) if future else None
