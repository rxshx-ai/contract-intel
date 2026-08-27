"""Back-to-back gap analysis: obligations we owe outward vs. what we secured upstream.

Companies sit in the middle of a contract chain. Obligations flow IN from
suppliers and OUT to customers. When what we promised downstream exceeds what
we secured upstream, WE absorb the difference out of our own balance sheet --
and nobody notices, because the two contracts live in different folders, were
signed eighteen months apart, and were reviewed by different people.

This analysis is structurally impossible for a chatbot or a single-document
analyzer. It requires normalized, comparable, structured fields across a
portfolio -- which is precisely what the extraction layer produces, and
precisely why the architecture is shaped the way it is.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from api.schemas import (
    ClauseClaim,
    ClauseType,
    Contract,
    Finding,
    OurRole,
    Severity,
)

CT = ClauseType


@dataclass(frozen=True)
class Dimension:
    """One comparable promise. `worse` decides which side is more onerous."""

    key: str
    clause_type: ClauseType
    field: str
    label: str
    # True when the OUTBOUND promise is stronger than the INBOUND one.
    exceeds: Callable[[float, float], bool]
    unit: str
    severity: Severity
    consequence: str


def _fmt(value: float, unit: str) -> str:
    if unit == "%":
        return f"{value:g}%"
    if unit == "days":
        return f"{value:g} days"
    if unit == "hours":
        return f"{value * 24:g} hours"
    if unit == "currency":
        return f"{value:,.0f}"
    return f"{value:g}"


DIMENSIONS: list[Dimension] = [
    Dimension(
        key="uptime", clause_type=CT.SLA, field="uptime_percent",
        label="availability commitment",
        exceeds=lambda out, inn: out > inn, unit="%", severity="critical",
        consequence=(
            "Every minute of downtime between the two figures is a breach of our "
            "customer contract that our supplier contract does not cover. We pay the "
            "service credits; we recover nothing."
        ),
    ),
    Dimension(
        key="breach_notice", clause_type=CT.BREACH_NOTIFICATION, field="days",
        label="breach notification window",
        exceeds=lambda out, inn: out < inn, unit="hours", severity="critical",
        consequence=(
            "We are contractually required to notify our customer before our own "
            "supplier is required to notify us. The obligation is impossible to meet "
            "by any means other than luck, and under GDPR Article 33 the regulatory "
            "clock runs in parallel."
        ),
    ),
    Dimension(
        key="deletion", clause_type=CT.DATA_RETENTION_DELETION, field="days",
        label="data deletion deadline",
        exceeds=lambda out, inn: out < inn, unit="days", severity="high",
        consequence=(
            "We promise deletion sooner than our supplier promises it to us. For the "
            "gap period the data still exists somewhere in the chain while we have "
            "certified to our customer that it does not."
        ),
    ),
    Dimension(
        key="liability", clause_type=CT.LIABILITY_CAP, field="amount",
        label="liability cap",
        exceeds=lambda out, inn: out > inn, unit="currency", severity="critical",
        consequence=(
            "The difference between the two caps is uninsured, unrecoverable exposure "
            "carried on our own balance sheet. A single upstream failure that harms "
            "our customer is our loss, not our supplier's."
        ),
    ),
    Dimension(
        key="subprocessor_notice", clause_type=CT.SUBPROCESSORS, field="days",
        label="subprocessor change notice",
        exceeds=lambda out, inn: out > inn, unit="days", severity="medium",
        consequence=(
            "We owe our customer more notice of a subprocessor change than our own "
            "supplier owes us, so we cannot pass the notice through in time."
        ),
    ),
]


def _pick(claims: list[ClauseClaim], ctype: ClauseType, field: str) -> ClauseClaim | None:
    """The most onerous stated value of its kind, so gaps are not hidden by an
    adjacent softer clause."""
    candidates = [
        c for c in claims
        if c.effective and c.clause_type == ctype and c.fields.get(field) is not None
    ]
    return candidates[0] if candidates else None


def find_gaps(
    portfolio: list[tuple[Contract, list[ClauseClaim]]],
) -> list[Finding]:
    """Compare every outbound (we are seller) promise against every inbound one."""
    outbound = [(k, c) for k, c in portfolio if k.our_role == OurRole.SELLER]
    inbound = [(k, c) for k, c in portfolio if k.our_role == OurRole.BUYER]
    findings: list[Finding] = []

    for dim in DIMENSIONS:
        for out_contract, out_claims in outbound:
            out_claim = _pick(out_claims, dim.clause_type, dim.field)
            if out_claim is None:
                continue
            out_value = float(out_claim.fields[dim.field])

            for in_contract, in_claims in inbound:
                in_claim = _pick(in_claims, dim.clause_type, dim.field)
                if in_claim is None:
                    continue
                in_value = float(in_claim.fields[dim.field])
                if not dim.exceeds(out_value, in_value):
                    continue

                findings.append(
                    Finding(
                        id=f"gap_{dim.key}_{out_contract.id}_{in_contract.id}",
                        kind="backtoback_gap",
                        severity=dim.severity,
                        title=(
                            f"Flow-down gap in {dim.label}: we promise "
                            f"{_fmt(out_value, dim.unit)} to {out_contract.counterparty}, "
                            f"our supplier gives us {_fmt(in_value, dim.unit)}"
                        ),
                        explanation=(
                            f"Outbound: {out_contract.title} commits us to "
                            f"{_fmt(out_value, dim.unit)}.\n"
                            f"Inbound: {in_contract.title} secures only "
                            f"{_fmt(in_value, dim.unit)} from "
                            f"{in_contract.counterparty}.\n\n"
                            f"{dim.consequence}\n\n"
                            f"Fix: raise the upstream commitment to match, or lower the "
                            f"downstream promise. Until one of those happens we are "
                            f"underwriting the difference ourselves."
                        ),
                        evidence=[out_claim.span, in_claim.span],
                        contract_ids=[out_contract.id, in_contract.id],
                        metadata={
                            "dimension": dim.key,
                            "outbound_value": out_value,
                            "inbound_value": in_value,
                            "outbound_contract": out_contract.title,
                            "inbound_contract": in_contract.title,
                            "exposed_revenue": out_contract.annual_value,
                        },
                    )
                )

    order = {"critical": 0, "high": 1, "medium": 2, "low": 3, "info": 4}
    findings.sort(key=lambda f: order[f.severity])
    return findings


def exposure_summary(findings: list[Finding]) -> dict:
    """Total customer revenue sitting behind at least one flow-down gap."""
    gaps = [f for f in findings if f.kind == "backtoback_gap"]
    revenue = {
        f.metadata.get("outbound_contract"): f.metadata.get("exposed_revenue") or 0.0
        for f in gaps
    }
    return {
        "gap_count": len(gaps),
        "contracts_affected": len(revenue),
        "exposed_revenue": sum(revenue.values()),
        "dimensions": sorted({f.metadata.get("dimension") for f in gaps}),
    }
