"""Adapter: analysis output -> the exact shapes the designed front end renders.

The design ships its own presentation logic (labels, colours, ordering). This
module feeds it real data in the shapes it already expects, so the design is
never edited to fit the backend, and the mapping lives in Python where it can
be tested.

One presentation decision worth stating: a contract is a FAMILY of documents,
so the source view concatenates them into one readable text. Offsets shown to
the user stay the ORIGINAL per-document offsets -- the combined text is a
display artefact and never becomes the thing we claim to have verified.
"""

from __future__ import annotations

import json
import pathlib
from datetime import date
from typing import Any

from api.findings.silence import PLAYBOOKS
from api.risk import band
from api.schemas import (
    ClauseType,
    ContractType,
    Finding,
    OurRole,
    Span,
)
from api.temporal import resolve_term_end

ROOT = pathlib.Path(__file__).resolve().parents[1]
DOC_SEPARATOR = "\n\n\n"

CT = ClauseType

CLAUSE_LABELS: dict[ClauseType, str] = {
    CT.EFFECTIVE_DATE: "Start date",
    CT.TERM: "How long it runs",
    CT.AUTO_RENEWAL: "Automatic renewal",
    CT.NOTICE_PERIOD: "Notice you must give",
    CT.TERMINATION_CONVENIENCE: "Leaving for convenience",
    CT.TERMINATION_CAUSE: "Leaving for cause",
    CT.EARLY_TERMINATION_FEE: "Charge for leaving early",
    CT.CURE_PERIOD: "Time to fix a breach",
    CT.PAYMENT_TERMS: "When you pay",
    CT.PRICE_INCREASE: "How prices rise",
    CT.MINIMUM_COMMITMENT: "What you committed to",
    CT.MOST_FAVORED_NATION: "Best-price promise",
    CT.LIABILITY_CAP: "Limit on liability",
    CT.UNCAPPED_CARVEOUT: "Where the limit stops applying",
    CT.INDEMNIFICATION: "Who covers whose claims",
    CT.INSURANCE: "Insurance you must hold",
    CT.WARRANTY: "What is promised about the service",
    CT.SLA: "Uptime promise",
    CT.SLA_CREDIT: "What you get back for downtime",
    CT.SUPPORT_RESPONSE: "How fast they answer",
    CT.IP_ASSIGNMENT: "Who owns what is made",
    CT.LICENSE_GRANT: "Rights you hand over",
    CT.CONFIDENTIALITY: "Keeping things confidential",
    CT.DATA_PROTECTION: "How your data is looked after",
    CT.DATA_RETENTION_DELETION: "Data deletion",
    CT.BREACH_NOTIFICATION: "Telling you about an incident",
    CT.SUBPROCESSORS: "Who else touches your data",
    CT.AUDIT_RIGHTS: "Audit rights",
    CT.GOVERNING_LAW: "Which law applies",
    CT.VENUE: "Where disputes are heard",
    CT.ASSIGNMENT: "Handing the contract on",
    CT.CHANGE_OF_CONTROL: "If either side is acquired",
    CT.UNILATERAL_AMENDMENT: "Changing the terms later",
    CT.NON_COMPETE: "Restrictions on you",
    CT.EXCLUSIVITY: "Exclusivity",
    CT.FORCE_MAJEURE: "Events outside anyone's control",
    CT.SURVIVAL: "What outlives the contract",
}

CONTRACT_KIND: dict[ContractType, str] = {
    ContractType.MSA: "Master services agreement",
    ContractType.NDA: "Mutual non-disclosure agreement",
    ContractType.SOW: "Statement of work",
    ContractType.DPA: "Data processing agreement",
    ContractType.ORDER_FORM: "Order form",
    ContractType.AMENDMENT: "Amendment",
    ContractType.UNKNOWN: "Agreement",
}

SIDE = {
    OurRole.BUYER: "They supply you",
    OurRole.SELLER: "You supply them",
    OurRole.MUTUAL: "Mutual",
}

FINDING_KIND_LABEL = {
    "risky_clause": "Wording worth reading",
    "missing_clause": "Wording that isn't there",
    "adversarial_pattern": "A term that works against you",
    "backtoback_gap": "Cross-contract gap",
    "injection": "Hidden instruction",
    "asymmetry": "One-sided rights",
}

# Which team a finding lands on. Teams, not invented people.
OWNER_BY_KIND = {
    "injection": "Security",
    "backtoback_gap": "Legal",
    "missing_clause": "Legal",
    "adversarial_pattern": "Legal",
    "asymmetry": "Legal",
    "risky_clause": "Legal",
}

OBLIGATION_KIND_LABEL = {
    "notice": "Renewal notice",
    "renewal": "Renewal",
    "expiry": "Expiry",
    "payment": "Payment",
    "report": "Report due",
    "cure": "Time to fix a breach",
}

COVERAGE_COLUMNS: list[tuple[ClauseType, str]] = [
    (CT.LIABILITY_CAP, "Limit on liability"),
    (CT.NOTICE_PERIOD, "Notice deadline found"),
    (CT.BREACH_NOTIFICATION, "Incident notice"),
    (CT.DATA_RETENTION_DELETION, "Data deletion"),
    (CT.CONFIDENTIALITY, "Confidentiality"),
    (CT.SLA, "Uptime promise"),
    (CT.INDEMNIFICATION, "Indemnity"),
]


def _money(value: float | None) -> str:
    if not value:
        return "—"
    return f"${value:,.0f}"


def _plural(n: int, one: str, many: str) -> str:
    return f"{n} {one if n == 1 else many}"


# --------------------------------------------------------------------------
# span registry -- one stable id per distinct piece of quoted text
# --------------------------------------------------------------------------

class QuoteRegistry:
    """Assigns stable ids to spans and maps them into the combined document."""

    def __init__(self, docs, base: dict[str, int], doc_len: int = 0):
        self.docs = {d.id: d for d in docs}
        self.base = base
        self.by_key: dict[tuple[str, int, int], str] = {}
        self.entries: list[dict[str, Any]] = []
        self.extra: list[str] = []      # appended cross-contract passages
        self.tail = doc_len

    def attach_external(self, span: Span, doc, source_title: str) -> None:
        """Append a passage quoted from ANOTHER contract to this source view.

        A cross-contract gap cites one passage from each side. Dropping the
        foreign one left the chain beat showing "one quoted passage" for a
        finding whose entire point is that there are two. The appended block is
        clearly labelled, and the offsets we report stay the original
        document's -- the combined text is only ever a reading surface.
        """
        key = (span.doc_id, span.char_start, span.char_end)
        if key in self.by_key:
            return
        header = f"\n\n\n[Related passage — {source_title}, {doc.filename}]\n\n"
        self.extra.append(header + span.quote)
        start = self.tail + len(header)
        qid = f"q{len(self.entries) + 1}"
        self.by_key[key] = qid
        self.entries.append({
            "id": qid,
            "type": f"From {source_title}",
            "favors": "na",
            "text": span.quote,
            "start": start,
            "end": start + len(span.quote),
            "srcStart": span.char_start,
            "srcEnd": span.char_end,
            "srcFile": doc.filename,
            "page": doc.page_for(span.char_start),
            "external": True,
        })
        self.tail = start + len(span.quote)

    def add(self, span: Span, label: str, favors: str = "na") -> str | None:
        doc = self.docs.get(span.doc_id)
        if doc is None or span.doc_id not in self.base:
            return None
        key = (span.doc_id, span.char_start, span.char_end)
        if key in self.by_key:
            return self.by_key[key]
        qid = f"q{len(self.entries) + 1}"
        self.by_key[key] = qid
        offset = self.base[span.doc_id]
        self.entries.append({
            "id": qid,
            "type": label,
            "favors": favors,
            "text": span.quote,
            # offsets into the combined source view (for highlighting)
            "start": offset + span.char_start,
            "end": offset + span.char_end,
            # the offsets we actually verified, shown to the user
            "srcStart": span.char_start,
            "srcEnd": span.char_end,
            "srcFile": doc.filename,
            "page": doc.page_for(span.char_start),
        })
        return qid


# --------------------------------------------------------------------------

def _combined_document(docs) -> tuple[str, dict[str, int]]:
    """Concatenate a contract family into one readable source view."""
    parts: list[str] = []
    base: dict[str, int] = {}
    cursor = 0
    for i, doc in enumerate(docs):
        if i:
            parts.append(DOC_SEPARATOR)
            cursor += len(DOC_SEPARATOR)
        base[doc.id] = cursor
        parts.append(doc.text)
        cursor += len(doc.text)
    return "".join(parts), base


def _obligation_sentence(ob, contract, term_end: date | None, today: date) -> str:
    days = ob.days_remaining(today)
    value = _money(contract.annual_value)
    if ob.kind == "notice" and ob.anchor == "term_end":
        if days < 0:
            return (
                f"The window to stop this renewing closed on {ob.due_date:%-d %B}. "
                f"A new term has already begun"
                + (f" at {value} a year." if contract.annual_value else ".")
            )
        return (
            f"Write to {contract.counterparty} by {ob.due_date:%-d %B} or this "
            f"renews automatically"
            + (f" for another year at {value}." if contract.annual_value else ".")
        )
    if ob.consequence_if_missed and len(ob.consequence_if_missed) > 25:
        return ob.consequence_if_missed
    party = contract.counterparty or "the counterparty"
    if ob.kind == "report":
        return (f"You owe {party} a written report on this date. "
                f"It recurs, so missing one is a pattern rather than an accident.")
    if ob.kind == "payment":
        return f"Payment to {party} falls due on this date."
    if ob.kind == "expiry":
        return f"Obligations under this agreement with {party} lapse on this date."
    if ob.kind == "cure":
        return "A breach must be put right by this date or the agreement can be ended."
    if ob.kind == "notice":
        return (f"Written notice must reach {party} by this date to change what "
                f"happens next.")
    return ob.description or f"{OBLIGATION_KIND_LABEL.get(ob.kind, ob.kind)} due."


def _derivation(ob, rules_by_id, registry: QuoteRegistry) -> list[list[Any]]:
    """Turn the computed chain into [tag, text, quoteId] triples.

    Steps that restate contract wording are tagged `quoted` and carry the span
    they came from; the rest are arithmetic and say so.
    """
    rule = rules_by_id.get(ob.rule_id)
    qid = None
    if rule is not None:
        qid = registry.add(rule.span, "The wording this comes from")

    rows: list[list[Any]] = []
    if rule is not None and qid:
        rows.append(["quoted", "The contract says this about the deadline.", qid])
    for step in ob.derivation:
        is_quoted = step.startswith(("Rule:", "Condition:", "Effective Date ="))
        rows.append(["quoted" if is_quoted else "computed", step,
                     qid if is_quoted else None])
    return rows


def _coverage_cell(claims, findings, ctype: ClauseType) -> str:
    present = any(c.effective and c.clause_type == ctype for c in claims)
    if not present:
        return "missing"
    ids = {c.id for c in claims if c.clause_type == ctype}
    risky = any(
        f.kind in ("adversarial_pattern", "risky_clause", "backtoback_gap")
        and any(
            e.char_start == c.span.char_start and e.doc_id == c.span.doc_id
            for e in f.evidence for c in claims if c.id in ids
        )
        for f in findings
    )
    return "weak" if risky else "present"


def _eval_rows() -> list[list[str]]:
    path = ROOT / "eval" / "eval_results.json"
    if not path.exists():
        return []
    data = json.loads(path.read_text())
    return data.get("rows", [])


def _eval_stats() -> dict[str, Any]:
    path = ROOT / "eval" / "eval_results.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text()).get("stats", {})


# --------------------------------------------------------------------------

def build_model(
    bundles,
    gaps: list[Finding],
    today: date,
    upload_checks: list[list[str]] | None = None,
    last_upload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    contracts: list[dict[str, Any]] = []
    obligations: list[dict[str, Any]] = []
    findings_out: list[dict[str, Any]] = []
    coverage_rows: list[list[str]] = []
    injections: list[dict[str, Any]] = []

    all_docs = {d.id: d for b in bundles for d in b.docs}
    titles = {d.id: b.contract.counterparty or b.contract.title
              for b in bundles for d in b.docs}

    gaps_by_contract: dict[str, list[Finding]] = {}
    for gap in gaps:
        for cid in gap.contract_ids:
            gaps_by_contract.setdefault(cid, []).append(gap)

    for bundle in bundles:
        contract = bundle.contract
        result = bundle.result()
        doc_text, base = _combined_document(bundle.docs)
        registry = QuoteRegistry(bundle.docs, base, len(doc_text))

        for claim in bundle.claims:
            if claim.effective:
                registry.add(claim.span,
                             CLAUSE_LABELS.get(claim.clause_type,
                                               claim.clause_type.value),
                             claim.party_favored)

        rules_by_id = {r.id: r for r in bundle.rules}
        term_end = None
        if contract.effective_date:
            months = 12
            for c in bundle.claims:
                if c.effective and c.clause_type == CT.TERM and c.fields.get("months"):
                    months = int(c.fields["months"])
                    break
            renewal = None
            for c in bundle.claims:
                if c.effective and c.clause_type == CT.AUTO_RENEWAL \
                        and c.fields.get("months"):
                    renewal = int(c.fields["months"])
                    break
            term_end, _, _ = resolve_term_end(contract.effective_date, months,
                                              renewal, today)

        for ob in bundle.obligations:
            days = ob.days_remaining(today)
            overdue = days < 0
            headline = ob.kind == "notice" and ob.anchor == "term_end"
            obligations.append({
                "id": f"{contract.id}:{ob.rule_id}:{ob.due_date.isoformat()}",
                "cid": contract.id,
                "kind": OBLIGATION_KIND_LABEL.get(ob.kind, ob.kind.title()),
                "due": ob.due_date.isoformat(),
                "flag": "MISSED — ALREADY RENEWED" if (overdue and headline)
                        else ("OVERDUE" if overdue else ""),
                "urgent": bool(overdue or (headline and days <= 90)),
                "plain": _obligation_sentence(ob, contract, term_end, today),
                "meta": (f"owed by {'you' if ob.owed_by == 'us' else ob.owed_by}"
                         f" · computed, not read off the page"),
                "owner": "Procurement" if ob.owed_by == "us" else "Legal",
                "value": contract.annual_value,
                "derivation": _derivation(ob, rules_by_id, registry),
            })

        own_findings = list(bundle.findings) + gaps_by_contract.get(contract.id, [])
        for finding in own_findings:
            for span in finding.evidence:
                if span.doc_id not in base:
                    other = all_docs.get(span.doc_id)
                    if other is not None:
                        registry.attach_external(
                            span, other, titles.get(span.doc_id, "another contract"))
            ev = [qid for qid in
                  (registry.add(span, "Evidence") if span.doc_id in base
                   else registry.by_key.get(
                       (span.doc_id, span.char_start, span.char_end))
                   for span in finding.evidence)
                  if qid]
            findings_out.append({
                "id": finding.id,
                "cid": contract.id,
                "sev": finding.severity,
                "kind": FINDING_KIND_LABEL.get(finding.kind, finding.kind),
                "ev": ev,
                "title": finding.title,
                "ex": finding.explanation,
                "todo": _todo(finding),
                "owner": OWNER_BY_KIND.get(finding.kind, "Legal"),
                "playbook": finding.metadata.get("playbook", ""),
            })

        for report, doc in zip(bundle.firewall_reports, bundle.docs):
            for ind in report.indicators:
                injections.append({
                    "vector": _VECTOR_LABEL.get(ind.kind, ind.kind),
                    "where": (f"{doc.filename}"
                              + (f" · page {ind.page}" if ind.page else "")),
                    "detector": ind.detail,
                    "payload": ind.excerpt[:260],
                    "cid": contract.id,
                })

        asym = result.asymmetry
        profile = result.risk
        coverage_rows.append(
            [contract.id] + [
                _coverage_cell(bundle.claims, own_findings, ctype)
                for ctype, _ in COVERAGE_COLUMNS
            ]
        )

        governing = next(
            (c.fields.get("note") or c.span.quote[:60]
             for c in bundle.claims
             if c.effective and c.clause_type == CT.GOVERNING_LAW), "not stated")
        amendments = sum(1 for d in bundle.docs
                         if d.contract_type == ContractType.AMENDMENT)

        contracts.append({
            "id": contract.id,
            "party": contract.counterparty or contract.title,
            "kind": CONTRACT_KIND.get(contract.contract_type, "Agreement"),
            "side": SIDE.get(contract.our_role, "Mutual"),
            "value": contract.annual_value or 0,
            "effective": contract.effective_date.isoformat()
                         if contract.effective_date else "",
            "termEnd": term_end.isoformat() if term_end else "",
            "renewalEnd": term_end.isoformat() if term_end else "",
            "governing": governing,
            "owner": "Procurement" if contract.our_role == OurRole.BUYER else "Legal",
            "note": _contract_note(bundle, amendments),
            "file": " · ".join(d.filename for d in bundle.docs),
            "doc": doc_text + "".join(registry.extra),
            "risk": profile.overall if profile else 0,
            "riskCaption": _plural(len(own_findings), "thing worth your time",
                                   "things worth your time"),
            "power": str(round((asym.index if asym else 0) * 100)),
            "powerNote": (f"{len(asym.their_rights)} discretionary rights theirs, "
                          f"{len(asym.our_rights)} yours") if asym else "",
            "grounding": bundle.grounding_rate,
            "quotes": registry.entries,
            "axes": [
                {
                    "axis": axis.axis,
                    "score": axis.score,
                    "contrib": [
                        [f"+{c.points}", c.reason,
                         _quote_id_for_clause(registry, bundle, c.clause_id)]
                        for c in axis.contributions
                    ],
                }
                for axis in (profile.axes if profile else [])
            ],
        })

    return {
        "asOf": today.isoformat(),
        "contracts": contracts,
        "obligations": sorted(obligations, key=lambda o: o["due"]),
        "findings": findings_out,
        "gaps": [_gap_row(g) for g in gaps],
        "injections": injections,
        "eval": _eval_rows(),
        "evalStats": _eval_stats(),
        "coverageCols": [label for _, label in COVERAGE_COLUMNS],
        "coverage": coverage_rows,
        "uploadChecks": upload_checks or [],
        "lastUpload": last_upload or {},
        "stats": _stats(bundles, gaps, findings_out, today),
    }


_VECTOR_LABEL = {
    "invisible_text": "Text the same colour as the page",
    "tiny_font": "Text too small to read",
    "offscreen_text": "Text printed off the page",
    "metadata_payload": "Instruction in the file metadata",
    "injection_language": "Instruction aimed at an automated reader",
}


def _quote_id_for_clause(registry: QuoteRegistry, bundle, clause_id) -> str | None:
    if not clause_id:
        return None
    for claim in bundle.claims:
        if claim.id == clause_id:
            key = (claim.span.doc_id, claim.span.char_start, claim.span.char_end)
            return registry.by_key.get(key)
    return None


def _contract_note(bundle, amendments: int) -> str:
    parts = [_plural(len(bundle.docs), "document", "documents")]
    if amendments:
        parts.append(f"{amendments} amendment folded in" if amendments == 1
                     else f"{amendments} amendments folded in")
    superseded = sum(1 for c in bundle.claims if not c.effective)
    if superseded:
        parts.append(f"{superseded} superseded clause"
                     + ("" if superseded == 1 else "s") + " kept underneath")
    return ". ".join(parts).capitalize() + "."


def _todo(finding: Finding) -> str:
    if finding.kind == "missing_clause":
        return "Ask for the clause to be added before signing or at renewal."
    if finding.kind == "backtoback_gap":
        return ("Raise the upstream commitment to match, or lower what you "
                "promised downstream.")
    if finding.kind == "injection":
        return "Request a clean copy from the counterparty. Do not auto-process."
    if finding.kind == "asymmetry":
        asks = finding.metadata.get("asks") or []
        return ("Ask for reciprocity on: " + "; ".join(asks[:3])) if asks \
            else "Ask for reciprocity on these rights."
    return "Read the wording and decide whether to push back at renewal."


def _gap_row(gap: Finding) -> dict[str, str]:
    meta = gap.metadata
    dim = {
        "uptime": "Uptime a month",
        "breach_notice": "Telling you about an incident",
        "deletion": "Data deletion",
        "liability": "Limit on liability",
        "subprocessor_notice": "Notice of a new subprocessor",
    }.get(meta.get("dimension", ""), meta.get("dimension", ""))
    unit = {"uptime": "%", "liability": "", "breach_notice": "h",
            "deletion": " days", "subprocessor_notice": " days"}.get(
        meta.get("dimension", ""), "")
    inn, out = meta.get("inbound_value"), meta.get("outbound_value")
    if meta.get("dimension") == "breach_notice":
        inn, out = (inn or 0) * 24, (out or 0) * 24
    fmt = (lambda v: f"{v:,.0f}" if meta.get("dimension") == "liability"
           else f"{v:g}{unit}")
    return {
        "dim": dim,
        "received": f"{fmt(inn)} from {meta.get('inbound_contract', '')}",
        "promised": f"{fmt(out)} to {meta.get('outbound_contract', '')}",
        "gap": gap.title,
        "exposure": _money(meta.get("exposed_revenue")) + " of customer revenue",
    }


def _stats(bundles, gaps, findings_out, today: date) -> dict[str, Any]:
    obligations = [o for b in bundles for o in b.obligations]
    missed = [o for o in obligations if o.days_remaining(today) < 0]
    next90 = [o for o in obligations if 0 <= o.days_remaining(today) <= 90]
    held = sum(1 for b in bundles
               if any(r.quarantined for r in b.firewall_reports))
    absence = sum(1 for f in findings_out if not f["ev"])
    spans = sum(len(c) for c in
                [[q for q in b.claims] for b in bundles])
    return {
        "contracts": len(bundles),
        "documents": sum(len(b.docs) for b in bundles),
        "missed": len(missed),
        "next90": len(next90),
        "held": held,
        "findings": len(findings_out),
        "absenceFindings": absence,
        "gaps": len(gaps),
        "injections": sum(len(r.indicators) for b in bundles
                          for r in b.firewall_reports),
        "groundedSpans": spans,
        "groundingRate": min([b.grounding_rate for b in bundles], default=1.0),
        "committedValue": sum(b.contract.annual_value or 0 for b in bundles
                              if b.contract.our_role == OurRole.BUYER),
        "exposedRevenue": sum(
            {g.metadata.get("outbound_contract"): g.metadata.get("exposed_revenue") or 0
             for g in gaps}.values()),
    }
