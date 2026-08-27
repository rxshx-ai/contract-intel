"""Contract dark patterns: language that reads as boilerplate and isn't.

Pairs with the prompt-injection firewall under one story -- we detect
adversarial behaviour in the document whether it is aimed at your lawyer or at
our model.
"""

from __future__ import annotations

import re

from api.schemas import ClauseClaim, ClauseType, Contract, Finding

CT = ClauseType

_SOLE_DISCRETION = re.compile(r"\bsole discretion\b", re.IGNORECASE)


def _by_type(claims: list[ClauseClaim]) -> dict[ClauseType, list[ClauseClaim]]:
    out: dict[ClauseType, list[ClauseClaim]] = {}
    for c in claims:
        if c.effective:
            out.setdefault(c.clause_type, []).append(c)
    return out


def detect_adversarial(claims: list[ClauseClaim], contract: Contract) -> list[Finding]:
    by = _by_type(claims)
    findings: list[Finding] = []

    def add(key, severity, title, explanation, evidence, **meta):
        findings.append(
            Finding(
                id=f"adv_{contract.id}_{key}", kind="adversarial_pattern",
                severity=severity, title=title, explanation=explanation,
                evidence=evidence, contract_ids=[contract.id], metadata=meta,
            )
        )

    # 1. The blank cheque
    for c in by.get(CT.UNILATERAL_AMENDMENT, []):
        add("blank_cheque", "critical",
            "Unilateral amendment right: the terms can change without us",
            "The counterparty may change the agreement at any time, typically by "
            "posting a new version. Every other protection in this contract is "
            "provisional, because the clause granting them can itself be rewritten. "
            "In practice this means the agreement reviewed today is not the "
            "agreement binding tomorrow. Standard ask: changes take effect only at "
            "renewal, with notice and a right to reject.",
            [c.span])

    # 2. Evergreen with a punitive notice window
    for c in by.get(CT.AUTO_RENEWAL, []) + by.get(CT.NOTICE_PERIOD, []):
        days = c.fields.get("days")
        months = c.fields.get("months")
        if days and days >= 90:
            add(f"long_notice_{days}", "high",
                f"Auto-renewal with a {days}-day notice window",
                f"Exit requires a decision {days} days before renewal -- "
                f"{days // 30} months of foresight about a service you may not have "
                f"finished evaluating. Windows this long exist to be missed; the "
                f"renewal is the default outcome and inaction is the vendor's ally."
                + (f" Each missed window commits a further {months} months."
                   if months and months >= 24 else ""),
                [c.span], notice_days=days)

    # 3. Immortal indemnity
    for c in by.get(CT.INDEMNIFICATION, []) + by.get(CT.UNCAPPED_CARVEOUT, []):
        if c.fields.get("survives_termination") and c.party_favored == "counterparty":
            add("immortal_indemnity", "high",
                "Indemnity survives termination with no time limit",
                "This obligation outlives the agreement indefinitely. Years after the "
                "relationship ends, and after the commercial benefit has stopped, the "
                "exposure remains live. Standard ask: survival capped at 2-3 years, "
                "or tied to the applicable statute of limitations.",
                [c.span])

    # 4. Discretion creep
    discretion = [c for c in claims if c.effective and _SOLE_DISCRETION.search(c.span.quote)]
    if len(discretion) >= 2:
        add("discretion_creep", "medium",
            f"'Sole discretion' appears in {len(discretion)} operative clauses",
            "Individually unremarkable, collectively decisive: each instance converts "
            "a negotiated term into a one-party decision. Read together, the "
            "commercial substance of the agreement is set by the counterparty after "
            "signature, not by the document.",
            [c.span for c in discretion[:5]], count=len(discretion))

    # 5. Vendor-controlled pricing
    for c in by.get(CT.PRICE_INCREASE, []):
        if c.fields.get("unilateral") or c.fields.get("percent") is None:
            add("uncapped_pricing", "high",
                "Price increases are not capped at a stated percentage",
                "Future cost is set by the counterparty rather than by the agreement. "
                "The contract fixes what we must pay for but not what we must pay. "
                "Standard ask: a stated ceiling (typically 3-5%) or an increase tied "
                "to a published index neither party controls.",
                [c.span])

    # 6. Acceleration on exit
    for c in by.get(CT.EARLY_TERMINATION_FEE, []):
        pct = c.fields.get("percent")
        quote = c.span.quote.lower()
        if "immediately due" in quote or (pct and pct >= 50):
            add("acceleration", "high",
                "Termination triggers acceleration of remaining fees",
                "Leaving early does not reduce the bill; it brings it forward. The "
                "termination right is nominal, because exercising it costs the same "
                "as staying. Combined with a long term, this converts a service "
                "agreement into a financing arrangement.",
                [c.span])

    # 7. A cap smaller than the cost of enforcing it
    caps = by.get(CT.LIABILITY_CAP, [])
    venues = by.get(CT.VENUE, [])
    for cap in caps:
        amount = cap.fields.get("amount")
        if amount is not None and amount <= 50000 and venues:
            add("uneconomic_remedy", "medium",
                "Liability cap is below the practical cost of enforcing it",
                f"With a cap of {amount:,.0f} and an exclusive foreign venue, pursuing "
                f"a claim would likely cost more than the maximum recovery. The remedy "
                f"exists on paper and is uneconomic in practice -- which is the point.",
                [cap.span, venues[0].span], cap=amount)

    # 8. One-way exit
    for c in by.get(CT.TERMINATION_CONVENIENCE, []):
        if c.fields.get("unilateral") and c.party_favored == "counterparty":
            add("one_way_exit", "high",
                "Only the counterparty may terminate for convenience",
                "They may walk away at will; we are committed for the full term. All "
                "of the flexibility and none of the commitment sits on their side, "
                "while we must plan around a service that can be withdrawn on notice.",
                [c.span])

    return findings
