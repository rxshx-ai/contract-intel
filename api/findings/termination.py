"""Exit cost calculator: 'we want out on this date' -> an itemized bill.

Every input is already extracted. This converts the structured layer into a
decision a human is actively trying to make, which is the difference between a
report and a product.
"""

from __future__ import annotations

from datetime import date

from api.schemas import (
    ClauseClaim,
    ClauseType,
    Contract,
    Obligation,
    TerminationCost,
)
from api.temporal import add_months, resolve_term_end

CT = ClauseType


def termination_cost(
    contract: Contract,
    claims: list[ClauseClaim],
    obligations: list[Obligation],
    exit_date: date,
    today: date,
    initial_term_months: int = 12,
    renewal_months: int | None = 12,
) -> TerminationCost:
    cost = TerminationCost(
        contract_id=contract.id, exit_date=exit_date, currency=contract.currency
    )
    if contract.effective_date is None:
        cost.notes.append("No Effective Date; exit cost cannot be computed.")
        return cost

    term_end, renewals, steps = resolve_term_end(
        contract.effective_date, initial_term_months, renewal_months, today
    )
    cost.notes.extend(steps)

    # Did the notice window for the current term already close?
    notice = next((o for o in obligations if o.kind == "notice"), None)
    if notice and notice.due_date < today:
        missed_by = (today - notice.due_date).days
        cost.notes.append(
            f"Non-renewal notice was due {notice.due_date.isoformat()} "
            f"({missed_by} days ago). That window has closed, so the term now runs to "
            f"{term_end.isoformat()} regardless of when notice is given."
        )
    elif notice:
        cost.notes.append(
            f"Notice window is still open until {notice.due_date.isoformat()} "
            f"({(notice.due_date - today).days} days). Giving notice by then avoids "
            f"the renewal entirely and reduces this cost to zero."
        )

    # Committed fees from the exit date to the end of the term we are locked into.
    monthly = (contract.annual_value or 0.0) / 12.0
    months_remaining = max(0, _months_between(exit_date, term_end))
    remaining_fees = monthly * months_remaining
    if remaining_fees > 0:
        cost.line_items.append({
            "label": f"Committed fees to end of term ({term_end.isoformat()})",
            "detail": f"{months_remaining} months x {monthly:,.0f}/month",
            "amount": round(remaining_fees, 2),
        })

    # Early termination fee, applied to the remaining fees.
    for c in claims:
        if not c.effective or c.clause_type != CT.EARLY_TERMINATION_FEE:
            continue
        pct = c.fields.get("percent")
        if pct:
            fee = remaining_fees * (pct / 100.0)
            cost.line_items.append({
                "label": f"Early termination fee ({pct:g}% of remaining fees)",
                "detail": c.span.quote[:160],
                "amount": round(fee, 2),
                "clause_span": c.span.model_dump(),
            })
        elif c.fields.get("amount"):
            cost.line_items.append({
                "label": "Early termination fee",
                "detail": c.span.quote[:160],
                "amount": float(c.fields["amount"]),
                "clause_span": c.span.model_dump(),
            })
        else:
            cost.notes.append(
                "An early termination fee applies but is not stated as a computable "
                f"amount: \"{c.span.quote[:140]}\""
            )

    # Non-financial obligations that survive the exit.
    for c in claims:
        if not c.effective:
            continue
        if c.clause_type == CT.DATA_RETENTION_DELETION:
            days = c.fields.get("days")
            cost.notes.append(
                f"Data return/deletion obligation: {days} days after termination."
                if days else "A data return/deletion obligation applies on termination."
            )
        elif c.clause_type in (CT.INDEMNIFICATION, CT.CONFIDENTIALITY) and \
                c.fields.get("survives_termination"):
            cost.notes.append(
                f"{c.clause_type.value.replace('_', ' ').title()} survives termination "
                f"and remains live after exit."
            )

    cost.total = round(sum(item["amount"] for item in cost.line_items), 2)
    return cost


def _months_between(start: date, end: date) -> int:
    if end <= start:
        return 0
    months = (end.year - start.year) * 12 + (end.month - start.month)
    if add_months(start, months) < end:
        months += 1
    return months
