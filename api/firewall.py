"""Treat every document as adversarial input. Invariant 4.

Two jobs:
  1. Detect content hidden from a human reader (tiny/white/off-page text,
     metadata payloads) and content aimed at an automated reader
     (instruction-shaped language).
  2. Wrap document text so it can never be read as instructions -- delimited,
     nonce-fenced, and explicitly labelled untrusted before it reaches a model.

A hidden instruction is reported as a TAMPERING INDICATOR ON THE COUNTERPARTY,
not as a bug in our pipeline. The attack becomes a finding about the vendor.
"""

from __future__ import annotations

import re
import secrets

from api.schemas import Document, Finding, FirewallReport, Span, TamperIndicator

# Language that addresses an automated reader rather than a contracting party.
# Contracts speak to parties; none of this belongs in one.
_INJECTION_PATTERNS: list[tuple[str, str]] = [
    (r"ignore\s+(?:all\s+)?(?:previous|prior|above|earlier)\s+instructions",
     "instructs a reader to discard prior instructions"),
    (r"disregard\s+(?:all\s+|any\s+)?(?:previous|prior|above|the\s+foregoing)\s+"
     r"(?:instructions|analysis|rules)",
     "instructs a reader to disregard prior instructions"),
    (r"\b(?:system|assistant|user)\s*(?:note|prompt|message|instruction)\b",
     "impersonates a system or assistant role"),
    (r"(?:you\s+are|act\s+as)\s+(?:an?\s+)?(?:ai|assistant|language\s+model)",
     "assigns a persona to an automated reader"),
    (r"report\s+(?:the\s+)?(?:overall\s+)?risk(?:\s+score)?\s+as\b",
     "dictates the risk score to be reported"),
    (r"(?:do\s+not|don't|never)\s+(?:extract|mention|summari[sz]e|flag|report|analyz)",
     "instructs a reader to suppress analysis"),
    (r"(?:output|respond\s+with|return|print)\s+only\b",
     "dictates the output of an automated reader"),
    (r"(?:pre-?approved|approved)\s+by\s+(?:[\w']+\s+){0,3}counsel.{0,60}"
     r"(?:no\s+further|requires?\s+no)",
     "asserts pre-approval to suppress review"),
    (r"\bautomated\s+(?:review\s+)?(?:tools?|systems?|agents?)\b",
     "addresses automated review tooling directly"),
    (r"(?:mark|classify|score|treat)\s+(?:this|the)\s+(?:agreement|contract|document)"
     r"\s+as\s+(?:low|no|zero)\s*[- ]?\s*risk",
     "dictates a risk classification"),
]

_COMPILED = [(re.compile(p, re.IGNORECASE | re.DOTALL), why) for p, why in _INJECTION_PATTERNS]

# Directive-shaped bracketed blocks: [SYSTEM NOTE: ...], <!-- ... -->, {{ ... }}
_BRACKETED = re.compile(
    r"(\[[^\]]{40,400}\]|<!--.{20,400}?-->|\{\{.{20,400}?\}\})",
    re.IGNORECASE | re.DOTALL,
)
_DIRECTIVE_VERB = re.compile(
    r"\b(ignore|disregard|report|output|do not|must not|never|always|instruct|"
    r"you\s+(?:are|must|should))\b",
    re.IGNORECASE,
)

TINY_FONT_PT = 4.0
_WHITE = (1.0, 1.0, 1.0)


# --------------------------------------------------------------------------
# text-layer scanning
# --------------------------------------------------------------------------

def scan_text(doc: Document) -> list[TamperIndicator]:
    indicators: list[TamperIndicator] = []
    seen: set[tuple[str, int]] = set()

    for pattern, why in _COMPILED:
        for match in pattern.finditer(doc.text):
            key = (why, match.start() // 200)
            if key in seen:
                continue
            seen.add(key)
            indicators.append(
                TamperIndicator(
                    kind="injection_language",
                    detail=why,
                    excerpt=_excerpt(doc.text, match.start(), match.end()),
                    page=doc.page_for(match.start()),
                )
            )

    for match in _BRACKETED.finditer(doc.text):
        block = match.group(1)
        if len(_DIRECTIVE_VERB.findall(block)) < 2:
            continue
        if any(abs(match.start() - _offset_of(doc.text, i.excerpt)) < 300 for i in indicators):
            continue
        indicators.append(
            TamperIndicator(
                kind="injection_language",
                detail="bracketed block containing imperative directives",
                excerpt=_excerpt(doc.text, match.start(), match.end()),
                page=doc.page_for(match.start()),
            )
        )
    return indicators


def _excerpt(text: str, start: int, end: int, pad: int = 90) -> str:
    lo, hi = max(0, start - pad), min(len(text), end + pad)
    return text[lo:hi].replace("\n", " ").strip()


def _offset_of(text: str, excerpt: str) -> int:
    idx = text.find(excerpt[:40])
    return idx if idx != -1 else -10_000


# --------------------------------------------------------------------------
# layout-layer scanning (things a human literally cannot see)
# --------------------------------------------------------------------------

def scan_pdf_layers(path: str) -> list[TamperIndicator]:
    """Find text rendered invisibly: sub-4pt, white-on-white, or off-page."""
    try:
        import pdfplumber
    except ImportError:  # pragma: no cover
        return []

    indicators: list[TamperIndicator] = []
    with pdfplumber.open(path) as pdf:
        for pageno, page in enumerate(pdf.pages, start=1):
            tiny, invisible, offpage = [], [], []
            for ch in page.chars:
                text = ch.get("text", "")
                if not text.strip():
                    continue
                if (ch.get("size") or 99) < TINY_FONT_PT:
                    tiny.append(text)
                if _is_white(ch.get("non_stroking_color")):
                    invisible.append(text)
                x0, top = ch.get("x0", 0), ch.get("top", 0)
                if x0 < -1 or top < -1 or x0 > page.width + 1 or top > page.height + 1:
                    offpage.append(text)

            for chars, kind, detail in (
                (tiny, "tiny_font", f"text rendered below {TINY_FONT_PT}pt"),
                (invisible, "invisible_text", "text rendered in the page background colour"),
                (offpage, "offscreen_text", "text positioned outside the page boundary"),
            ):
                if len(chars) >= 20:
                    indicators.append(
                        TamperIndicator(
                            kind=kind, detail=detail,
                            excerpt="".join(chars)[:300].strip(), page=pageno,
                        )
                    )

        meta = " ".join(str(v) for v in (pdf.metadata or {}).values())
        if meta and any(p.search(meta) for p, _ in _COMPILED):
            indicators.append(
                TamperIndicator(
                    kind="metadata_payload",
                    detail="instruction-shaped text embedded in PDF metadata",
                    excerpt=meta[:300],
                )
            )
    return indicators


def _is_white(color) -> bool:
    if color is None:
        return False
    if isinstance(color, (int, float)):
        return float(color) >= 0.99
    try:
        vals = [float(c) for c in color]
    except (TypeError, ValueError):
        return False
    return len(vals) in (1, 3) and all(v >= 0.99 for v in vals)


# --------------------------------------------------------------------------
# public entry point
# --------------------------------------------------------------------------

def inspect(doc: Document, path: str | None = None) -> FirewallReport:
    indicators = scan_text(doc)
    if path and path.lower().endswith(".pdf"):
        indicators += scan_pdf_layers(path)
    # Hidden or instruction-shaped content means the document is quarantined:
    # we still analyze it, but every downstream claim is flagged.
    quarantined = any(
        i.kind in ("injection_language", "invisible_text", "tiny_font",
                   "offscreen_text", "metadata_payload")
        for i in indicators
    )
    return FirewallReport(doc_id=doc.id, indicators=indicators, quarantined=quarantined)


def to_findings(report: FirewallReport, doc: Document, contract_id: str) -> list[Finding]:
    """Turn the attack into a finding about the counterparty."""
    findings: list[Finding] = []
    for n, ind in enumerate(report.indicators):
        idx = doc.text.find(ind.excerpt[:60])
        if idx == -1:
            idx = 0
        span = Span(
            doc_id=doc.id, char_start=idx,
            char_end=min(idx + len(ind.excerpt), len(doc.text)),
            quote=doc.text[idx : min(idx + len(ind.excerpt), len(doc.text))],
        )
        findings.append(
            Finding(
                id=f"inj_{doc.id}_{n}",
                kind="injection",
                severity="critical",
                title=f"Hidden instruction detected in counterparty document ({ind.kind})",
                explanation=(
                    f"This document contains content that {ind.detail}. Contract text "
                    f"addresses contracting parties; language addressed to an automated "
                    f"reviewer is not contractual and its presence indicates a deliberate "
                    f"attempt to manipulate automated review. Treat as a tampering "
                    f"indicator on the counterparty and escalate to human review."
                ),
                evidence=[span],
                contract_ids=[contract_id],
                metadata={"indicator_kind": ind.kind, "page": ind.page},
            )
        )
    return findings


# --------------------------------------------------------------------------
# spotlighting -- how untrusted text reaches the model
# --------------------------------------------------------------------------

def wrap_untrusted(text: str) -> tuple[str, str]:
    """Fence document text with an unguessable nonce.

    The model is told: everything between the fences is DATA quoted from a
    third party. It is never concatenated into the instruction context, and a
    document cannot close the fence because it cannot guess the nonce.
    """
    nonce = secrets.token_hex(8)
    fence = f"<<<UNTRUSTED_DOCUMENT_{nonce}>>>"
    # Defensive: strip any pre-existing fence-shaped tokens from the document.
    cleaned = re.sub(r"<<<\s*/?UNTRUSTED_DOCUMENT_[0-9a-f]*\s*>>>", "[REDACTED-FENCE]", text)
    return f"{fence}\n{cleaned}\n{fence}", fence
