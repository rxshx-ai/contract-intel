"""Contract family graph and clause supersession.

A contract is not a PDF. It is a STACK: MSA -> Order Form -> amendments ->
side letters -> DPA. Amendment No. 2 may have raised a liability cap that
Amendment No. 1 lowered.

Analyze the MSA alone and you will confidently report a cap that has not been
true for two years. That is worse than no tool at all, because it manufactures
false confidence. Every clause here resolves to an EFFECTIVE value with visible
lineage.
"""

from __future__ import annotations

import re
from datetime import date

from api.schemas import (
    ClauseClaim,
    ClauseType,
    Contract,
    ContractType,
    Document,
)

# Clause types an amendment replaces wholesale when it restates them.
# Types that legitimately appear many times are excluded from supersession.
_MULTI_INSTANCE = {
    ClauseType.UNCAPPED_CARVEOUT,
    ClauseType.INDEMNIFICATION,
    ClauseType.CONFIDENTIALITY,
    ClauseType.LICENSE_GRANT,
}

_DATE_PATTERNS = [
    (re.compile(r"\b(\d{1,2})\s+(January|February|March|April|May|June|July|August|"
                r"September|October|November|December)\s+(\d{4})\b", re.I), "dmy"),
    (re.compile(r"\b(January|February|March|April|May|June|July|August|September|"
                r"October|November|December)\s+(\d{1,2}),?\s+(\d{4})\b", re.I), "mdy"),
    (re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b"), "iso"),
]

_MONTHS = {m.lower(): i for i, m in enumerate(
    ["January", "February", "March", "April", "May", "June", "July", "August",
     "September", "October", "November", "December"], start=1)}

_AMENDMENT_NO = re.compile(r"amendment\s+(?:no\.?|number)\s*(\d+)", re.IGNORECASE)


def parse_date(text: str) -> date | None:
    """First plausible date in a string. Used for amendment ordering."""
    for pattern, kind in _DATE_PATTERNS:
        match = pattern.search(text)
        if not match:
            continue
        try:
            if kind == "dmy":
                return date(int(match.group(3)), _MONTHS[match.group(2).lower()],
                            int(match.group(1)))
            if kind == "mdy":
                return date(int(match.group(3)), _MONTHS[match.group(1).lower()],
                            int(match.group(2)))
            return date(int(match.group(1)), int(match.group(2)), int(match.group(3)))
        except (ValueError, KeyError):
            continue
    return None


def amendment_rank(doc: Document) -> tuple[int, str]:
    """Order amendments by their stated number, falling back to date then name."""
    match = _AMENDMENT_NO.search(doc.text[:400]) or _AMENDMENT_NO.search(doc.filename)
    if match:
        return int(match.group(1)), doc.filename
    signed = parse_date(doc.text[:600])
    return (signed.toordinal() if signed else 0), doc.filename


# --------------------------------------------------------------------------

def order_documents(docs: list[Document]) -> list[Document]:
    """Base documents first, then amendments in the order they were made."""
    base = [d for d in docs if d.contract_type != ContractType.AMENDMENT]
    amendments = sorted(
        (d for d in docs if d.contract_type == ContractType.AMENDMENT),
        key=amendment_rank,
    )
    return base + amendments


def resolve_supersession(
    claims_by_doc: dict[str, list[ClauseClaim]],
    docs: list[Document],
) -> tuple[list[ClauseClaim], dict[str, str]]:
    """Mark superseded clauses and return the lineage map.

    Later documents in the family win. A clause is superseded only by a clause
    of the same type in a LATER document, so an amendment that is silent on a
    topic leaves the original in force -- which is how amendments actually work.
    """
    ordered = order_documents(docs)
    by_type: dict[ClauseType, ClauseClaim] = {}
    lineage: dict[str, str] = {}
    all_claims: list[ClauseClaim] = []

    for doc in ordered:
        for claim in claims_by_doc.get(doc.id, []):
            all_claims.append(claim)
            if claim.clause_type in _MULTI_INSTANCE:
                continue
            previous = by_type.get(claim.clause_type)
            if previous is not None and previous.span.doc_id != claim.span.doc_id:
                previous.superseded_by = claim.id
                claim.supersedes = previous.id
                lineage[claim.id] = previous.id
            by_type[claim.clause_type] = claim

    return all_claims, lineage


def lineage_text(
    claim: ClauseClaim, claims: list[ClauseClaim], docs: dict[str, Document]
) -> str:
    """Human-readable provenance: what set this value, and what it replaced."""
    index = {c.id: c for c in claims}
    doc = docs.get(claim.span.doc_id)
    parts = [f"set by {doc.filename if doc else claim.span.doc_id}"]
    cursor = claim
    seen = {claim.id}
    while cursor.supersedes and cursor.supersedes in index:
        cursor = index[cursor.supersedes]
        if cursor.id in seen:
            break
        seen.add(cursor.id)
        prior_doc = docs.get(cursor.span.doc_id)
        parts.append(f"supersedes {prior_doc.filename if prior_doc else cursor.span.doc_id}")
    return "; ".join(parts)


def effective(claims: list[ClauseClaim]) -> list[ClauseClaim]:
    return [c for c in claims if c.effective]


def build_contract(
    contract_id: str,
    title: str,
    docs: list[Document],
    claims: list[ClauseClaim],
    counterparty: str,
    our_role,
    annual_value: float | None = None,
    currency: str = "USD",
) -> Contract:
    """Assemble the contract-level facts from whichever document carries them.

    The Effective Date is typically on the Order Form, not the MSA -- which is
    exactly why single-document analysis gets renewal dates wrong.
    """
    effective_date = None
    for doc in order_documents(docs):
        if doc.contract_type in (ContractType.ORDER_FORM, ContractType.MSA,
                                 ContractType.NDA, ContractType.SOW):
            found = parse_date(doc.text[:1200])
            if found:
                effective_date = found
                break
    if effective_date is None:
        for doc in docs:
            found = parse_date(doc.text[:1200])
            if found:
                effective_date = found
                break

    if annual_value is None:
        for claim in effective(claims):
            if claim.clause_type == ClauseType.MINIMUM_COMMITMENT:
                annual_value = claim.fields.get("amount")
                break

    base_type = next((d.contract_type for d in docs
                      if d.contract_type not in (ContractType.AMENDMENT,
                                                 ContractType.ORDER_FORM,
                                                 ContractType.UNKNOWN)),
                     ContractType.UNKNOWN)

    return Contract(
        id=contract_id, title=title, counterparty=counterparty, our_role=our_role,
        contract_type=base_type, doc_ids=[d.id for d in docs],
        effective_date=effective_date, annual_value=annual_value, currency=currency,
    )
