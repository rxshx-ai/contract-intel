"""Party-aware, dimension-decomposed risk scoring. Invariant 2.

The published rubric below IS the algorithm. No model is asked for a number.
Every point on every axis names the clause that produced it, so a score can be
argued with rather than merely believed.

Risk is signed by WHO WE ARE. An uncapped indemnity is catastrophic for the
indemnitor and excellent for the indemnitee; a single scalar "risk = 7.3"
hides that and is why most contract scoring is unfalsifiable.
"""

from __future__ import annotations

from typing import Callable, Iterator

from api.schemas import (
    ClauseClaim,
    ClauseType,
    Contract,
    OurRole,
    RiskAxis,
    RiskContribution,
    RiskProfile,
)

Axis = str
Emit = tuple[Axis, int, str, str | None]  # axis, points, reason, clause_id

AXES: tuple[Axis, ...] = ("financial", "lockin", "liability", "compliance", "operational")

# ---- the published rubric -------------------------------------------------
# Point values live here, in one table, so the scoring can be reviewed without
# reading the code.
RUBRIC = {
    "cap_below_half_annual_value": 40,
    "cap_below_annual_value": 25,
    "uncapped_exposure_on_us": 35,
    "indemnity_against_us": 20,
    "indemnity_survives_forever": 10,
    "notice_window_over_90_days": 25,
    "notice_window_over_60_days": 15,
    "auto_renewal_present": 10,
    "no_termination_for_convenience": 20,
    "unilateral_termination_by_them": 20,
    "early_termination_fee": 15,
    "uncapped_price_increase": 25,
    "price_increase_over_5pct": 12,
    "payment_terms_under_30_days": 8,
    "unilateral_amendment": 30,
    "unilateral_audit": 10,
    "assignment_blocked_on_change_of_control": 15,
    "breach_notification_over_72h": 15,
    "no_subprocessor_consent": 12,
    "data_deletion_over_60_days": 10,
    "sla_below_99_9": 15,
    "sla_credit_requires_claim": 8,
    "non_compete_present": 15,
    "feedback_licence_perpetual": 5,
}

_RULES: list[Callable[[list[ClauseClaim], Contract], Iterator[Emit]]] = []


def rule(fn):
    _RULES.append(fn)
    return fn


def _of(claims: list[ClauseClaim], *types: ClauseType) -> list[ClauseClaim]:
    return [c for c in claims if c.clause_type in types and c.effective]


def _against_us(claim: ClauseClaim) -> bool:
    return claim.party_favored == "counterparty"


# ---- liability ------------------------------------------------------------

@rule
def _liability_cap(claims, contract):
    for c in _of(claims, ClauseType.LIABILITY_CAP):
        cap = c.fields.get("amount")
        value = contract.annual_value
        if cap is None or not value:
            continue
        if cap < value * 0.5:
            yield ("liability", RUBRIC["cap_below_half_annual_value"],
                   f"Liability cap of {cap:,.0f} is under half the annual contract "
                   f"value of {value:,.0f} — a total service failure recovers less "
                   f"than six months of spend.", c.id)
        elif cap < value:
            yield ("liability", RUBRIC["cap_below_annual_value"],
                   f"Liability cap of {cap:,.0f} is below the annual contract value "
                   f"of {value:,.0f}.", c.id)


@rule
def _uncapped_carveouts(claims, contract):
    for c in _of(claims, ClauseType.UNCAPPED_CARVEOUT):
        if _against_us(c):
            yield ("liability", RUBRIC["uncapped_exposure_on_us"],
                   "Liability is uncapped for us under this carve-out while the "
                   "counterparty remains capped — the limitation is one-directional.",
                   c.id)


@rule
def _indemnity(claims, contract):
    for c in _of(claims, ClauseType.INDEMNIFICATION):
        if not _against_us(c):
            continue
        yield ("liability", RUBRIC["indemnity_against_us"],
               "We indemnify the counterparty; the obligation is not reciprocal.", c.id)
        if c.fields.get("survives_termination"):
            yield ("liability", RUBRIC["indemnity_survives_forever"],
                   "This indemnity survives termination without a time limit — "
                   "exposure continues after the relationship ends.", c.id)


# ---- lock-in --------------------------------------------------------------

@rule
def _renewal_and_notice(claims, contract):
    for c in _of(claims, ClauseType.AUTO_RENEWAL):
        yield ("lockin", RUBRIC["auto_renewal_present"],
               "Agreement renews automatically; exit requires affirmative action "
               "inside a notice window.", c.id)
    for c in _of(claims, ClauseType.NOTICE_PERIOD, ClauseType.AUTO_RENEWAL):
        days = c.fields.get("days")
        if not days:
            continue
        if days > 90:
            yield ("lockin", RUBRIC["notice_window_over_90_days"],
                   f"Non-renewal notice must be given {days} days ahead — the "
                   f"decision must be made more than a quarter before renewal.", c.id)
        elif days > 60:
            yield ("lockin", RUBRIC["notice_window_over_60_days"],
                   f"Non-renewal notice must be given {days} days ahead.", c.id)


@rule
def _termination(claims, contract):
    convenience = _of(claims, ClauseType.TERMINATION_CONVENIENCE)
    for c in convenience:
        if c.fields.get("unilateral") and _against_us(c):
            yield ("lockin", RUBRIC["unilateral_termination_by_them"],
                   "The counterparty may terminate for convenience and we may not — "
                   "they can exit at will while we are committed.", c.id)
    if contract.our_role == OurRole.BUYER and not any(
        c.party_favored in ("us", "mutual") for c in convenience
    ):
        yield ("lockin", RUBRIC["no_termination_for_convenience"],
               "We have no right to terminate for convenience; we are committed for "
               "the full term regardless of whether the service is still needed.",
               convenience[0].id if convenience else None)
    for c in _of(claims, ClauseType.EARLY_TERMINATION_FEE):
        yield ("lockin", RUBRIC["early_termination_fee"],
               "Early exit carries a fee, raising the cost of leaving.", c.id)


# ---- financial ------------------------------------------------------------

@rule
def _price(claims, contract):
    for c in _of(claims, ClauseType.PRICE_INCREASE):
        pct = c.fields.get("percent")
        if pct is None or c.fields.get("unilateral"):
            yield ("financial", RUBRIC["uncapped_price_increase"],
                   "Price increases are not capped at a stated percentage — future "
                   "cost is set by the counterparty.", c.id)
        elif pct > 5:
            yield ("financial", RUBRIC["price_increase_over_5pct"],
                   f"Fees may rise {pct:g}% per renewal, above typical 3-5% ceilings.",
                   c.id)


@rule
def _payment(claims, contract):
    for c in _of(claims, ClauseType.PAYMENT_TERMS):
        days = c.fields.get("days")
        if days and days < 30 and contract.our_role == OurRole.BUYER:
            yield ("financial", RUBRIC["payment_terms_under_30_days"],
                   f"Payment due in {days} days, tightening working capital.", c.id)


# ---- compliance -----------------------------------------------------------

@rule
def _data(claims, contract):
    for c in _of(claims, ClauseType.BREACH_NOTIFICATION):
        days = c.fields.get("days")
        hours = days * 24 if days else None
        if hours and hours > 72:
            yield ("compliance", RUBRIC["breach_notification_over_72h"],
                   f"Breach notification within {hours}h exceeds the 72h regulatory "
                   f"reporting window we ourselves must meet.", c.id)
    for c in _of(claims, ClauseType.SUBPROCESSORS):
        if c.fields.get("unilateral") or _against_us(c):
            yield ("compliance", RUBRIC["no_subprocessor_consent"],
                   "Subprocessors may be added without our consent or prior notice — "
                   "our data can reach parties we never approved.", c.id)
    for c in _of(claims, ClauseType.DATA_RETENTION_DELETION):
        days = c.fields.get("days")
        if days and days > 60:
            yield ("compliance", RUBRIC["data_deletion_over_60_days"],
                   f"Our data is retained for up to {days} days after termination.", c.id)


# ---- operational ----------------------------------------------------------

@rule
def _service(claims, contract):
    for c in _of(claims, ClauseType.SLA):
        uptime = c.fields.get("uptime_percent")
        if uptime is not None and uptime < 99.9:
            yield ("operational", RUBRIC["sla_below_99_9"],
                   f"Availability commitment of {uptime}% is below the 99.9% norm.", c.id)
    for c in _of(claims, ClauseType.SLA_CREDIT):
        if c.fields.get("days"):
            yield ("operational", RUBRIC["sla_credit_requires_claim"],
                   f"Service credits must be claimed within {c.fields['days']} days or "
                   f"they are waived — money is forfeited by inaction.", c.id)


@rule
def _control(claims, contract):
    for c in _of(claims, ClauseType.UNILATERAL_AMENDMENT):
        yield ("operational", RUBRIC["unilateral_amendment"],
               "The counterparty may change the terms unilaterally — the agreement "
               "we reviewed is not the agreement we are bound by tomorrow.", c.id)
    for c in _of(claims, ClauseType.AUDIT_RIGHTS):
        if c.fields.get("unilateral") or _against_us(c):
            yield ("operational", RUBRIC["unilateral_audit"],
                   "Audit rights run one way, at our cost.", c.id)
    for c in _of(claims, ClauseType.ASSIGNMENT, ClauseType.CHANGE_OF_CONTROL):
        if _against_us(c):
            yield ("operational", RUBRIC["assignment_blocked_on_change_of_control"],
                   "We cannot assign on a change of control without consent, while "
                   "the counterparty may assign freely — this can block an "
                   "acquisition or become leverage during one.", c.id)
    for c in _of(claims, ClauseType.NON_COMPETE, ClauseType.EXCLUSIVITY):
        if _against_us(c):
            yield ("operational", RUBRIC["non_compete_present"],
                   "A restraint on our own commercial activity.", c.id)
    for c in _of(claims, ClauseType.LICENSE_GRANT):
        if _against_us(c) and "perpetual" in c.span.quote.lower():
            yield ("operational", RUBRIC["feedback_licence_perpetual"],
                   "A perpetual, irrevocable licence is granted to the counterparty.",
                   c.id)


# ---- scoring --------------------------------------------------------------

def score(claims: list[ClauseClaim], contract: Contract) -> RiskProfile:
    buckets: dict[Axis, list[RiskContribution]] = {a: [] for a in AXES}
    for fn in _RULES:
        for axis, points, reason, clause_id in fn(claims, contract):
            buckets[axis].append(
                RiskContribution(clause_id=clause_id, points=points, reason=reason)
            )
    axes = [
        RiskAxis(
            axis=axis,
            score=min(100, sum(c.points for c in contributions)),
            contributions=sorted(contributions, key=lambda c: -c.points),
        )
        for axis, contributions in buckets.items()
    ]
    return RiskProfile(contract_id=contract.id, our_role=contract.our_role, axes=axes)


def band(value: int) -> str:
    if value >= 70:
        return "critical"
    if value >= 45:
        return "high"
    if value >= 20:
        return "medium"
    return "low"
