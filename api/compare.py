"""Structured comparison across contracts. Deterministic, quoted, no model.

The agent decides WHICH dimension to compare and across which contracts. The
comparison itself is arithmetic over verified clause fields, so "which contract
has the weakest liability cap" is computed, not generated -- and every cell
carries the wording it came from.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from api.schemas import ClauseType, OurRole

CT = ClauseType


@dataclass(frozen=True)
class Dimension:
    key: str
    label: str
    clause_types: tuple[ClauseType, ...]
    field: str
    unit: str
    # True when a HIGHER number is better for us AS THE BUYER. Every dimension
    # here flips when we are the seller: a 99.99% uptime commitment is a
    # protection when a supplier gives it to us and an exposure when we give it
    # to a customer. Comparing the two on one axis without the flip is how a
    # tool confidently tells you your worst contract is your best.
    higher_is_better_for_buyer: bool
    note: str = ""
    # Some dimensions are better read from the computed temporal rules than
    # from a clause label. Notice periods especially: the rule carries the
    # offset the deadline was actually derived from.
    rule_kind: str | None = None
    rule_anchor: str | None = None

    def higher_is_better(self, role: OurRole) -> bool:
        if role == OurRole.SELLER:
            return not self.higher_is_better_for_buyer
        return self.higher_is_better_for_buyer


DIMENSIONS: dict[str, Dimension] = {
    d.key: d for d in [
        Dimension("liability_cap", "Limit on liability",
                  (CT.LIABILITY_CAP,), "amount", "currency", True,
                  "As buyer, a higher cap means more can be recovered from them. "
                  "As seller, the same number is our own exposure."),
        Dimension("notice_days", "Notice needed to stop a renewal",
                  (CT.NOTICE_PERIOD, CT.AUTO_RENEWAL), "days", "days", False,
                  "Read from the rule the renewal deadline was computed from, "
                  "not from a clause label.",
                  rule_kind="notice", rule_anchor="term_end"),
        Dimension("uptime", "Uptime commitment", (CT.SLA,), "uptime_percent",
                  "percent", True, ""),
        Dimension("payment_days", "Days to pay an invoice",
                  (CT.PAYMENT_TERMS,), "days", "days", True,
                  "More days is better for our cash when we are paying."),
        Dimension("breach_notice", "Breach notification window",
                  (CT.BREACH_NOTIFICATION,), "days", "days", False,
                  "Shorter is better when they owe us the notice."),
        Dimension("deletion_days", "Data deletion after termination",
                  (CT.DATA_RETENTION_DELETION,), "days", "days", False, ""),
        Dimension("price_increase", "Cap on price rises",
                  (CT.PRICE_INCREASE,), "percent", "percent", False, ""),
        Dimension("term_months", "Length of the term", (CT.TERM,), "months",
                  "months", False, ""),
        Dimension("early_exit_fee", "Charge for leaving early",
                  (CT.EARLY_TERMINATION_FEE,), "percent", "percent", False, ""),
        Dimension("support_response", "Support response time",
                  (CT.SUPPORT_RESPONSE,), "days", "days", False, ""),
    ]
}

ALIASES = {
    "liability": "liability_cap", "cap": "liability_cap",
    "notice": "notice_days", "renewal": "notice_days",
    "sla": "uptime", "availability": "uptime",
    "payment": "payment_days", "terms": "payment_days",
    "breach": "breach_notice", "incident": "breach_notice",
    "deletion": "deletion_days", "retention": "deletion_days",
    "price": "price_increase", "uplift": "price_increase",
    "term": "term_months", "duration": "term_months",
    "exit": "early_exit_fee", "termination_fee": "early_exit_fee",
    "support": "support_response",
}


def resolve_dimension(name: str) -> Dimension | None:
    key = (name or "").strip().lower().replace(" ", "_").replace("-", "_")
    if key in DIMENSIONS:
        return DIMENSIONS[key]
    if key in ALIASES:
        return DIMENSIONS[ALIASES[key]]
    for alias, target in ALIASES.items():
        if alias in key:
            return DIMENSIONS[target]
    return None


def _format(value: float, unit: str, currency: str = "USD") -> str:
    if unit == "currency":
        return f"{value:,.0f} {currency}"
    if unit == "percent":
        return f"{value:g}%"
    if unit == "days":
        return f"{value:g} day" + ("" if value == 1 else "s")
    if unit == "months":
        return f"{value:g} month" + ("" if value == 1 else "s")
    return f"{value:g}"


def compare(
    bundles,
    dimension: str,
    contract_ids: list[str] | None = None,
) -> dict[str, Any]:
    """One dimension across contracts, with the wording behind every figure."""
    dim = resolve_dimension(dimension)
    if dim is None:
        return {
            "ok": False,
            "error": f"Unknown dimension {dimension!r}.",
            "available": sorted(DIMENSIONS),
        }

    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for bundle in bundles:
        contract = bundle.contract
        if contract_ids and contract.id not in contract_ids:
            continue
        name = contract.counterparty or contract.title

        if dim.rule_kind:
            rule = next((r for r in bundle.rules
                         if r.kind == dim.rule_kind and r.anchor == dim.rule_anchor
                         and r.offset_days), None)
            if rule is not None:
                value = float(abs(rule.offset_days))
                rows.append({
                    "contract_id": contract.id, "contract": name,
                    "side": contract.our_role.value,
                    "we_are": {"buyer": "buying", "seller": "selling",
                               "mutual": "mutual"}.get(contract.our_role.value, ""),
                    "higher_is_better": dim.higher_is_better(contract.our_role),
                    "value": value,
                    "display": _format(value, dim.unit, contract.currency),
                    "record_id": rule.id, "quote": rule.span.quote,
                    "file": next((d.filename for d in bundle.docs
                                  if d.id == rule.span.doc_id), None),
                    "start": rule.span.char_start, "end": rule.span.char_end,
                    "annual_value": contract.annual_value,
                })
                continue

        claim = next(
            (c for c in bundle.claims
             if c.effective and c.clause_type in dim.clause_types
             and c.fields.get(dim.field) is not None),
            None,
        )
        if claim is None:
            missing.append(name)
            continue
        value = float(claim.fields[dim.field])
        rows.append({
            "contract_id": contract.id,
            "contract": name,
            "side": contract.our_role.value,
            "we_are": {"buyer": "buying", "seller": "selling",
                       "mutual": "mutual"}.get(contract.our_role.value, ""),
            "higher_is_better": dim.higher_is_better(contract.our_role),
            "value": value,
            "display": _format(value, dim.unit, contract.currency),
            "record_id": claim.id,
            "quote": claim.span.quote,
            "file": next((d.filename for d in bundle.docs
                          if d.id == claim.span.doc_id), None),
            "start": claim.span.char_start,
            "end": claim.span.char_end,
            "annual_value": contract.annual_value,
        })

    rows.sort(key=lambda r: r["value"], reverse=True)

    # Rank WITHIN a side. Ranking a supplier contract against a customer
    # contract on one axis is the flow-down comparison, not a league table, and
    # a single "best" across both is meaningless.
    verdict_parts: list[str] = []
    for side, label in (("buyer", "Supplier side"), ("seller", "Customer side"),
                        ("mutual", "Mutual")):
        side_rows = [r for r in rows if r["side"] == side]
        if len(side_rows) < 1:
            continue
        better_high = side_rows[0]["higher_is_better"]
        ranked = sorted(side_rows, key=lambda r: r["value"], reverse=better_high)
        if len(ranked) == 1:
            verdict_parts.append(
                f"{label}: only {ranked[0]['contract']} states this "
                f"({ranked[0]['display']}).")
        else:
            verdict_parts.append(
                f"{label}: best is {ranked[0]['contract']} at "
                f"{ranked[0]['display']}, worst is {ranked[-1]['contract']} at "
                f"{ranked[-1]['display']}.")
    verdict = " ".join(verdict_parts)

    return {
        "ok": True,
        "dimension": dim.key,
        "label": dim.label,
        "unit": dim.unit,
        "note": dim.note,
        "rows": rows,
        "verdict": verdict,
        "not_stated": missing,
        "not_stated_note": (
            f"{len(missing)} contract(s) say nothing on this: "
            f"{', '.join(missing)}. Silence is not a good value -- it means the "
            f"protection is absent." if missing else ""
        ),
    }
