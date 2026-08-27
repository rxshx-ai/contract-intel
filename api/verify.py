"""The grounding gate. Invariant 1.

Everything the model produces passes through here before it can reach a user.
A claim whose quote is not literally present in the source document is
DISCARDED, not flagged, not shown with a warning. Ungrounded output is
structurally unable to reach the UI, which is what makes
`hallucination_rate = 0` a property of the architecture rather than a claim
about the model.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from api.schemas import ClauseClaim, Document, Finding, Span, TemporalRule


@dataclass
class VerificationReport:
    kept: int = 0
    dropped: int = 0
    reasons: list[str] = field(default_factory=list)

    @property
    def total(self) -> int:
        return self.kept + self.dropped

    @property
    def grounding_rate(self) -> float:
        return 1.0 if self.total == 0 else self.kept / self.total


def _span_ok(span: Span, docs: dict[str, Document]) -> tuple[bool, str]:
    doc = docs.get(span.doc_id)
    if doc is None:
        return False, f"span references unknown document {span.doc_id}"
    if span.char_start < 0 or span.char_end > len(doc.text):
        return False, f"span [{span.char_start}:{span.char_end}] outside document bounds"
    if span.char_start >= span.char_end:
        return False, "span is empty or inverted"
    if not span.is_grounded_in(doc.text):
        return False, f"quote is not the document text at that offset: {span.quote[:60]!r}"
    return True, ""


def verify_claims(
    claims: list[ClauseClaim], docs: dict[str, Document]
) -> tuple[list[ClauseClaim], VerificationReport]:
    report = VerificationReport()
    kept: list[ClauseClaim] = []
    for claim in claims:
        ok, why = _span_ok(claim.span, docs)
        if ok:
            kept.append(claim)
            report.kept += 1
        else:
            report.dropped += 1
            report.reasons.append(f"{claim.clause_type.value}: {why}")
    return kept, report


def verify_rules(
    rules: list[TemporalRule], docs: dict[str, Document]
) -> tuple[list[TemporalRule], VerificationReport]:
    report = VerificationReport()
    kept: list[TemporalRule] = []
    for rule in rules:
        ok, why = _span_ok(rule.span, docs)
        if ok:
            kept.append(rule)
            report.kept += 1
        else:
            report.dropped += 1
            report.reasons.append(f"{rule.kind}: {why}")
    return kept, report


def verify_findings(
    findings: list[Finding], docs: dict[str, Document]
) -> tuple[list[Finding], VerificationReport]:
    """Findings we generate ourselves are held to the same standard.

    `missing_clause` is exempt: it asserts the absence of text, so it has
    nothing to quote (invariant 5).
    """
    report = VerificationReport()
    kept: list[Finding] = []
    for finding in findings:
        if finding.kind == "missing_clause":
            kept.append(finding)
            report.kept += 1
            continue
        failures = [why for ok, why in (_span_ok(s, docs) for s in finding.evidence) if not ok]
        if failures:
            report.dropped += 1
            report.reasons.append(f"{finding.id}: {failures[0]}")
        else:
            kept.append(finding)
            report.kept += 1
    return kept, report
