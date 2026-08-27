"""Orchestration. The one place the module order is written down.

    ingest -> firewall -> extract -> verify -> family -> temporal -> risk
           -> findings -> AnalysisResult

Only `extract` touches a model. Everything after `verify` is deterministic,
which is why the same documents always produce the same report.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import date

from api import extract as extract_mod
from api import firewall, risk, temporal
from api.family import build_contract, resolve_supersession
from api.findings import (
    detect_adversarial,
    detect_silence,
    find_gaps,
    measure_asymmetry,
)
from api.schemas import (
    AnalysisResult,
    ClauseClaim,
    ClauseType,
    Contract,
    Document,
    Finding,
    FirewallReport,
    Obligation,
    OurRole,
    TemporalRule,
)
from api.verify import verify_claims, verify_findings, verify_rules


@dataclass
class ContractBundle:
    """One contract: its documents, its analysis, its findings."""

    contract: Contract
    docs: list[Document]
    claims: list[ClauseClaim] = field(default_factory=list)
    rules: list[TemporalRule] = field(default_factory=list)
    obligations: list[Obligation] = field(default_factory=list)
    findings: list[Finding] = field(default_factory=list)
    firewall_reports: list[FirewallReport] = field(default_factory=list)
    unresolved: list[str] = field(default_factory=list)
    grounding_rate: float = 1.0
    dropped: int = 0
    grounding: "extract_mod.GroundingStats" = field(
        default_factory=lambda: extract_mod.GroundingStats())

    # ---- persistence ---------------------------------------------------
    # A bundle is the entire analysis of one contract. Storing it means a fresh
    # process serves uploaded contracts without re-extracting -- no tokens, no
    # latency, and uploads stop disappearing when the container recycles.

    def to_payload(self) -> dict:
        return {
            "version": 1,
            "contract": self.contract.model_dump(mode="json"),
            "documents": [d.model_dump(mode="json") for d in self.docs],
            "claims": [c.model_dump(mode="json") for c in self.claims],
            "rules": [r.model_dump(mode="json") for r in self.rules],
            "obligations": [o.model_dump(mode="json") for o in self.obligations],
            "findings": [f.model_dump(mode="json") for f in self.findings],
            "firewall": [r.model_dump(mode="json") for r in self.firewall_reports],
            "unresolved": self.unresolved,
            "grounding_rate": self.grounding_rate,
            "dropped": self.dropped,
            "grounding": {
                "exact": self.grounding.exact,
                "whitespace": self.grounding.whitespace,
                "fuzzy": self.grounding.fuzzy,
                "dropped": self.grounding.dropped,
            },
        }

    @classmethod
    def from_payload(cls, payload: dict) -> "ContractBundle":
        stats = extract_mod.GroundingStats(**(payload.get("grounding") or {}))
        return cls(
            contract=Contract.model_validate(payload["contract"]),
            docs=[Document.model_validate(d) for d in payload["documents"]],
            claims=[ClauseClaim.model_validate(c) for c in payload["claims"]],
            rules=[TemporalRule.model_validate(r) for r in payload["rules"]],
            obligations=[Obligation.model_validate(o) for o in payload["obligations"]],
            findings=[Finding.model_validate(f) for f in payload["findings"]],
            firewall_reports=[FirewallReport.model_validate(r)
                              for r in payload.get("firewall", [])],
            unresolved=payload.get("unresolved", []),
            grounding_rate=payload.get("grounding_rate", 1.0),
            dropped=payload.get("dropped", 0),
            grounding=stats,
        )

    def result(self) -> AnalysisResult:
        report, _ = measure_asymmetry(self.claims, self.contract)
        return AnalysisResult(
            contract=self.contract,
            clauses=self.claims,
            obligations=self.obligations,
            findings=self.findings,
            risk=risk.score(self.claims, self.contract),
            asymmetry=report,
            firewall=self.firewall_reports,
            grounding_rate=self.grounding_rate,
            dropped_claims=self.dropped,
        )


SEVERITY_ORDER = {"critical": 0, "high": 1, "medium": 2, "low": 3, "info": 4}


def analyze_contract(
    docs: list[Document],
    *,
    title: str,
    counterparty: str,
    our_role: OurRole,
    our_party: str,
    today: date,
    annual_value: float | None = None,
    contract_id: str | None = None,
    doc_paths: dict[str, str] | None = None,
    use_cache: bool = True,
    on_event=None,
) -> ContractBundle:
    """`on_event(dict)` reports each stage as it completes.

    The stages are the pipeline's own, in order, so a caller watching the
    events is watching the real work rather than a scripted approximation.
    """
    emit = on_event or (lambda event: None)
    contract_id = contract_id or f"k_{uuid.uuid4().hex[:8]}"
    doc_index = {d.id: d for d in docs}
    paths = doc_paths or {}

    # 1. firewall -- before any model sees the text
    reports = []
    for doc in docs:
        report = firewall.inspect(doc, paths.get(doc.id))
        reports.append(report)
        emit({"type": "firewall", "doc_id": doc.id, "filename": doc.filename,
              "quarantined": report.quarantined,
              "indicators": [{"kind": i.kind, "detail": i.detail,
                              "excerpt": i.excerpt[:160]}
                             for i in report.indicators]})

    # 2. extract, per document
    claims_by_doc: dict[str, list[ClauseClaim]] = {}
    all_rules: list[TemporalRule] = []
    stats = extract_mod.GroundingStats()
    for doc in docs:
        def _chunk(i, total, start, end, cached, _doc=doc):
            emit({"type": "reading", "doc_id": _doc.id, "chunk": i + 1,
                  "chunks": total, "start": start, "end": end,
                  "cached": cached})

        def _wait(seconds, _doc=doc):
            emit({"type": "throttled", "doc_id": _doc.id,
                  "seconds": round(seconds)})

        raw = extract_mod.call_model(doc, our_party, use_cache=use_cache,
                                     on_chunk=_chunk, on_wait=_wait)
        claims, s1 = extract_mod.ground_clauses(
            raw, doc, contract_id, our_party, our_role)
        rules, s2 = extract_mod.ground_rules(
            raw, doc, contract_id, our_party, our_role)
        for claim in claims:
            emit({"type": "clause", "doc_id": doc.id, "id": claim.id,
                  "clause_type": claim.clause_type.value,
                  "favors": claim.party_favored,
                  "start": claim.span.char_start, "end": claim.span.char_end,
                  "quote": claim.span.quote[:220],
                  "fields": {k: v for k, v in claim.fields.items()
                             if k != "grounding"}})
        for rule in rules:
            emit({"type": "rule", "doc_id": doc.id, "kind": rule.kind,
                  "anchor": rule.anchor, "offset_days": rule.offset_days,
                  "start": rule.span.char_start, "end": rule.span.char_end,
                  "quote": rule.span.quote[:220]})
        claims_by_doc[doc.id] = claims
        all_rules.extend(rules)
        stats = stats.merge(s1).merge(s2)
        if annual_value is None and raw.annual_value:
            annual_value = raw.annual_value

    # 3. verify -- ungrounded output is discarded here, permanently
    flat = [c for cs in claims_by_doc.values() for c in cs]
    verified, claim_report = verify_claims(flat, doc_index)
    kept_ids = {c.id for c in verified}
    claims_by_doc = {
        doc_id: [c for c in cs if c.id in kept_ids] for doc_id, cs in claims_by_doc.items()
    }
    all_rules, rule_report = verify_rules(all_rules, doc_index)
    # Grounding rate spans BOTH gates: the model's quote had to be located in
    # the document, and the resulting span had to verify as an exact substring.
    emit({"type": "verified",
          "kept": claim_report.kept + rule_report.kept,
          "discarded": claim_report.dropped + rule_report.dropped + stats.dropped,
          "exact": stats.exact, "realigned": stats.fuzzy})
    surfaced = claim_report.kept + rule_report.kept
    attempted = stats.total
    grounding_rate = 1.0 if attempted == 0 else surfaced / attempted

    # 4. family -- resolve amendments into effective values
    claims, _lineage = resolve_supersession(claims_by_doc, docs)
    contract = build_contract(contract_id, title, docs, claims, counterparty,
                              our_role, annual_value)

    emit({"type": "family", "documents": len(docs),
          "superseded": sum(1 for c in claims if not c.effective)})

    # 5. temporal -- compute real dates from relative rules
    initial_months, renewal_months = _term_shape(claims, all_rules)
    obligations, unresolved = temporal.materialize(
        all_rules, contract, today,
        initial_term_months=initial_months, renewal_months=renewal_months,
    )

    for ob in obligations:
        emit({"type": "deadline", "kind": ob.kind, "anchor": ob.anchor,
              "due": ob.due_date.isoformat(), "owed_by": ob.owed_by,
              "days": ob.days_remaining(today),
              "description": ob.description,
              "derivation": ob.derivation})
    for reason in unresolved:
        emit({"type": "unresolved", "reason": reason})

    # 6. findings
    findings: list[Finding] = []
    findings += detect_silence(claims, contract, contract.contract_type)
    findings += detect_adversarial(claims, contract)
    _, asym_findings = measure_asymmetry(claims, contract)
    findings += asym_findings
    for report, doc in zip(reports, docs):
        findings += firewall.to_findings(report, doc, contract_id)

    findings, _ = verify_findings(findings, doc_index)
    findings.sort(key=lambda f: SEVERITY_ORDER[f.severity])
    for finding in findings:
        emit({"type": "finding", "kind": finding.kind,
              "severity": finding.severity, "title": finding.title,
              "evidenced": bool(finding.evidence)})

    return ContractBundle(
        contract=contract, docs=docs, claims=claims, rules=all_rules,
        obligations=obligations, findings=findings, firewall_reports=reports,
        unresolved=unresolved, grounding_rate=grounding_rate,
        dropped=stats.dropped + claim_report.dropped + rule_report.dropped,
        grounding=stats,
    )


def _term_shape(
    claims: list[ClauseClaim], rules: list[TemporalRule]
) -> tuple[int, int | None]:
    """Initial term length and renewal period, from extracted fields."""
    initial = 12
    for c in claims:
        if c.effective and c.clause_type == ClauseType.TERM and c.fields.get("months"):
            initial = int(c.fields["months"])
            break
    renewal = None
    for c in claims:
        if c.effective and c.clause_type == ClauseType.AUTO_RENEWAL and c.fields.get("months"):
            renewal = int(c.fields["months"])
            break
    if renewal is None:
        for r in rules:
            if r.kind == "renewal":
                renewal = temporal.parse_recurrence_months(r.recurrence)
                if renewal:
                    break
    return initial, renewal


# --------------------------------------------------------------------------
# portfolio
# --------------------------------------------------------------------------

def analyze_portfolio(bundles: list[ContractBundle]) -> list[Finding]:
    """Cross-contract analysis. Only possible over a structured layer."""
    gaps = find_gaps([(b.contract, b.claims) for b in bundles])
    docs = {d.id: d for b in bundles for d in b.docs}
    verified, _ = verify_findings(gaps, docs)
    return verified


def upcoming_deadlines(
    bundles: list[ContractBundle], today: date, within_days: int = 365
) -> list[dict]:
    """The screen that turns a PDF parser into a product."""
    rows: list[dict] = []
    for bundle in bundles:
        for obligation in bundle.obligations:
            days = obligation.days_remaining(today)
            if days > within_days:
                continue
            rows.append({
                "contract_id": bundle.contract.id,
                "contract": bundle.contract.title,
                "counterparty": bundle.contract.counterparty,
                "kind": obligation.kind,
                "anchor": obligation.anchor,
                "due_date": obligation.due_date.isoformat(),
                "days_remaining": days,
                "overdue": days < 0,
                "owed_by": obligation.owed_by,
                "description": obligation.description,
                "derivation": obligation.derivation,
                "annual_value": bundle.contract.annual_value,
            })
    rows.sort(key=lambda r: r["days_remaining"])
    return rows
