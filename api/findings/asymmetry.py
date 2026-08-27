"""Power asymmetry index.

Counts rights that run one way. Instantly legible to a non-lawyer, quantifies
something everyone feels but struggles to articulate, and -- unlike an absolute
"risk = 7.3" -- it is a COMPARATIVE metric, so it is honest.

It also produces a negotiation artifact for free: the list of rights to ask for.
"""

from __future__ import annotations

from api.schemas import (
    AsymmetryReport,
    ClauseClaim,
    ClauseType,
    Contract,
    Finding,
    Span,
)

# Clause types that confer a RIGHT on one party rather than an obligation.
RIGHT_TYPES = {
    ClauseType.TERMINATION_CONVENIENCE: "terminate for convenience",
    ClauseType.UNILATERAL_AMENDMENT: "amend the agreement unilaterally",
    ClauseType.AUDIT_RIGHTS: "audit the other party",
    ClauseType.ASSIGNMENT: "assign the agreement",
    ClauseType.CHANGE_OF_CONTROL: "act on a change of control",
    ClauseType.PRICE_INCREASE: "raise prices",
    ClauseType.SUBPROCESSORS: "appoint subprocessors",
    ClauseType.EARLY_TERMINATION_FEE: "charge an early termination fee",
    ClauseType.EXCLUSIVITY: "demand exclusivity",
    ClauseType.NON_COMPETE: "restrain the other party's activity",
}


def measure_asymmetry(
    claims: list[ClauseClaim], contract: Contract
) -> tuple[AsymmetryReport, list[Finding]]:
    ours: list[Span] = []
    theirs: list[Span] = []
    labels_theirs: list[str] = []
    labels_ours: list[str] = []

    for claim in claims:
        if not claim.effective or claim.clause_type not in RIGHT_TYPES:
            continue
        label = RIGHT_TYPES[claim.clause_type]
        # A mutual right counts for both sides; a one-sided right counts once.
        if claim.party_favored == "counterparty":
            theirs.append(claim.span)
            labels_theirs.append(label)
        elif claim.party_favored == "us":
            ours.append(claim.span)
            labels_ours.append(label)
        elif claim.party_favored == "mutual" and not claim.fields.get("unilateral"):
            ours.append(claim.span)
            theirs.append(claim.span)
            labels_ours.append(label)
            labels_theirs.append(label)

    report = AsymmetryReport(contract_id=contract.id, our_rights=ours, their_rights=theirs)

    findings: list[Finding] = []
    unmatched = [l for l in labels_theirs if l not in labels_ours]
    if report.index >= 0.7 and theirs:
        severity = "high" if report.index >= 0.85 else "medium"
        asks = "; ".join(sorted(set(unmatched))[:5]) or "reciprocity on the above rights"
        findings.append(
            Finding(
                id=f"asym_{contract.id}",
                kind="asymmetry",
                severity=severity,
                title=(
                    f"{contract.counterparty or 'The counterparty'} holds "
                    f"{len(theirs)} unilateral rights; we hold {len(ours)}"
                ),
                explanation=(
                    f"Power asymmetry index {report.index:.0%} (0% is balanced, 100% is "
                    f"entirely one-sided). Rights granted to them but not to us: {asks}. "
                    f"Each is a specific, quotable thing to ask for in negotiation."
                ),
                evidence=theirs[:8],
                contract_ids=[contract.id],
                metadata={
                    "index": round(report.index, 3),
                    "our_rights": len(ours),
                    "their_rights": len(theirs),
                    "asks": sorted(set(unmatched)),
                },
            )
        )
    return report, findings
