"""Silence detection: the risk that ISN'T there. Invariant 5.

Every other tool in this space finds risky clauses. None of them find missing
ones, because you cannot quote a clause that does not exist -- keyword search
cannot see it, and an extractor happily reports the eighteen clauses present
and never mentions the one that is absent.

A missing liability cap is usually worse than a bad one.

This is a set difference between a playbook of expected clauses and what
extraction actually found. It is the one finding kind permitted to carry no
evidence span, because it asserts absence.
"""

from __future__ import annotations

from api.schemas import (
    ClauseClaim,
    ClauseType,
    Contract,
    ContractType,
    Finding,
    OurRole,
    Severity,
)

CT = ClauseType

# expected clause -> (severity, why its absence matters)
Expectation = dict[ClauseType, tuple[Severity, str]]

_COMMERCIAL: Expectation = {
    CT.LIABILITY_CAP: (
        "critical",
        "There is no limitation of liability clause at all. Liability is therefore "
        "unlimited by default for both parties. A single incident can exceed the "
        "entire value of the agreement, and no insurance policy is sized for it.",
    ),
    CT.TERM: (
        "high",
        "No term is stated, so it is unclear when this agreement ends or whether it "
        "is perpetual. Renewal and exit rights cannot be computed without it.",
    ),
    CT.TERMINATION_CAUSE: (
        "high",
        "There is no termination-for-cause right. Even a serious, uncured breach by "
        "the counterparty may leave no contractual route out.",
    ),
    CT.GOVERNING_LAW: (
        "medium",
        "No governing law is specified. Which country's law applies would itself "
        "become the first dispute, before the actual dispute is reached.",
    ),
    CT.INDEMNIFICATION: (
        "medium",
        "No indemnity is present. A third-party IP or data claim arising from the "
        "counterparty's service would be defended at our own cost.",
    ),
    CT.CONFIDENTIALITY: (
        "high",
        "No confidentiality obligation. Information shared under this agreement is "
        "not contractually protected.",
    ),
}

_SERVICE: Expectation = {
    CT.SLA: (
        "high",
        "No availability commitment. The service may be down indefinitely without "
        "breaching the agreement.",
    ),
    CT.SLA_CREDIT: (
        "medium",
        "No service credits. Downtime carries no financial consequence for the "
        "provider, which removes the main incentive to restore service quickly.",
    ),
    CT.SUPPORT_RESPONSE: (
        "low",
        "No support response commitment. 'We will look at it' is the whole promise.",
    ),
}

_DATA: Expectation = {
    CT.DATA_PROTECTION: (
        "high",
        "No data protection or security obligations, despite the service handling "
        "our data. There is no contractual security standard to hold them to.",
    ),
    CT.BREACH_NOTIFICATION: (
        "critical",
        "No breach notification obligation. The counterparty could suffer a breach "
        "of our data and never be required to tell us -- while our own regulatory "
        "clock (72 hours under GDPR) starts the moment we become aware.",
    ),
    CT.DATA_RETENTION_DELETION: (
        "high",
        "No data deletion obligation on termination. Our data may be retained "
        "indefinitely after the relationship ends.",
    ),
    CT.SUBPROCESSORS: (
        "medium",
        "No subprocessor controls. Our data may be passed to parties we have never "
        "assessed and cannot enumerate for our own compliance obligations.",
    ),
}

_NDA: Expectation = {
    CT.CONFIDENTIALITY: (
        "critical",
        "A non-disclosure agreement with no confidentiality obligation.",
    ),
    CT.TERM: (
        "high",
        "No duration for the confidentiality obligation.",
    ),
    CT.DATA_RETENTION_DELETION: (
        "medium",
        "No return-or-destroy obligation once the evaluation ends.",
    ),
    CT.GOVERNING_LAW: (
        "medium",
        "No governing law specified.",
    ),
    CT.LIABILITY_CAP: (
        "high",
        "No limitation of liability. Exposure for an inadvertent disclosure is "
        "unlimited, which is a common and expensive oversight in mutual NDAs.",
    ),
}

PLAYBOOKS: dict[ContractType, Expectation] = {
    ContractType.MSA: {**_COMMERCIAL, **_SERVICE, **_DATA},
    ContractType.SOW: {**_COMMERCIAL, **_SERVICE},
    ContractType.DPA: {**_DATA, CT.AUDIT_RIGHTS: (
        "medium", "No audit right over a processor handling our personal data.")},
    ContractType.ORDER_FORM: {CT.TERM: _COMMERCIAL[CT.TERM]},
    ContractType.NDA: _NDA,
    ContractType.AMENDMENT: {},
    ContractType.UNKNOWN: _COMMERCIAL,
}

# Clauses only a buyer of services needs; skip them when we are the seller.
_BUYER_ONLY = {CT.SLA, CT.SLA_CREDIT, CT.SUPPORT_RESPONSE, CT.BREACH_NOTIFICATION,
               CT.DATA_RETENTION_DELETION, CT.SUBPROCESSORS, CT.DATA_PROTECTION}


def detect_silence(
    claims: list[ClauseClaim],
    contract: Contract,
    contract_type: ContractType | None = None,
) -> list[Finding]:
    """Diff the playbook against what was actually extracted."""
    ctype = contract_type or contract.contract_type
    playbook = PLAYBOOKS.get(ctype, PLAYBOOKS[ContractType.UNKNOWN])

    present = {c.clause_type for c in claims if c.effective}
    findings: list[Finding] = []

    for expected, (severity, why) in playbook.items():
        if expected in present:
            continue
        if contract.our_role == OurRole.SELLER and expected in _BUYER_ONLY:
            continue  # we owe these outward; their absence is not our exposure
        findings.append(
            Finding(
                id=f"missing_{contract.id}_{expected.value}",
                kind="missing_clause",
                severity=severity,
                title=f"No {expected.value.replace('_', ' ')} clause in this {ctype.value.upper()}",
                explanation=why,
                evidence=[],  # invariant 5: absence has nothing to quote
                contract_ids=[contract.id],
                metadata={"clause_type": expected.value, "playbook": ctype.value},
            )
        )

    order = {"critical": 0, "high": 1, "medium": 2, "low": 3, "info": 4}
    findings.sort(key=lambda f: order[f.severity])
    return findings


def coverage(claims: list[ClauseClaim], contract_type: ContractType) -> float:
    """Share of the playbook this contract actually addresses."""
    playbook = PLAYBOOKS.get(contract_type, PLAYBOOKS[ContractType.UNKNOWN])
    if not playbook:
        return 1.0
    present = {c.clause_type for c in claims if c.effective}
    return len(present & set(playbook)) / len(playbook)
