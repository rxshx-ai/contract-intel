"""Calendar: when the document arrived, and every date inside it.

Three sources, kept distinct because they carry different weight:

  system   — when we ingested the file. Provenance, not contract content.
  computed — deadlines derived by temporal.py from relative wording. These are
             the ones that matter, and none of them appear in the document.
  quoted   — dates written literally in the text. Grounded to a span, so a
             user can see the sentence the date came from.

A literal date is NOT the same thing as a deadline, and the calendar says which
is which. Most contract calendars conflate them, which is how "the contract
says 31 December" turns into a missed notice window that was actually 60 days
earlier.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Literal

from api.family import parse_date
from api.schemas import Document, Span

Source = Literal["system", "computed", "quoted"]

# Literal dates as contracts actually write them.
_DATE_PATTERNS = [
    re.compile(r"\b\d{1,2}\s+(?:January|February|March|April|May|June|July|August|"
               r"September|October|November|December)\s+\d{4}\b", re.I),
    re.compile(r"\b(?:January|February|March|April|May|June|July|August|September|"
               r"October|November|December)\s+\d{1,2},?\s+\d{4}\b", re.I),
    re.compile(r"\b\d{4}-\d{2}-\d{2}\b"),
    re.compile(r"\b\d{1,2}/\d{1,2}/\d{4}\b"),
]

KIND_LABEL = {
    "uploaded": "Document received",
    "effective": "Agreement starts",
    "term_end": "Current term ends",
    "notice": "Notice deadline",
    "renewal": "Renews",
    "report": "Report due",
    "payment": "Payment due",
    "expiry": "Obligations lapse",
    "cure": "Cure period",
    "mentioned": "Date written in the document",
}


@dataclass
class CalendarEvent:
    id: str
    date: date
    kind: str
    source: Source
    title: str
    detail: str
    contract_id: str
    contract: str
    file: str | None = None
    quote: str | None = None
    start: int | None = None
    end: int | None = None
    page: int | None = None
    actionable: bool = False
    meta: dict[str, Any] = field(default_factory=dict)

    def days_from(self, today: date) -> int:
        return (self.date - today).days

    def to_dict(self, today: date) -> dict[str, Any]:
        return {
            "id": self.id,
            "date": self.date.isoformat(),
            "kind": self.kind,
            "label": KIND_LABEL.get(self.kind, self.kind),
            "source": self.source,
            "title": self.title,
            "detail": self.detail,
            "contract_id": self.contract_id,
            "contract": self.contract,
            "file": self.file,
            "quote": self.quote,
            "start": self.start,
            "end": self.end,
            "page": self.page,
            "actionable": self.actionable,
            "days": self.days_from(today),
            "overdue": self.days_from(today) < 0,
            "meta": self.meta,
        }


# --------------------------------------------------------------------------

def literal_dates(doc: Document) -> list[tuple[date, Span]]:
    """Every date written in the document, each with the span it came from."""
    found: dict[tuple[int, int], tuple[date, Span]] = {}
    for pattern in _DATE_PATTERNS:
        for match in pattern.finditer(doc.text):
            parsed = parse_date(match.group(0))
            if parsed is None:
                continue
            span = Span(doc_id=doc.id, char_start=match.start(),
                        char_end=match.end(), quote=match.group(0))
            found[(match.start(), match.end())] = (parsed, span)
    return [found[k] for k in sorted(found)]


def _line_around(doc: Document, start: int, end: int) -> str:
    lo = max(0, doc.text.rfind("\n", 0, start) + 1)
    hi = doc.text.find("\n", end)
    hi = len(doc.text) if hi == -1 else hi
    return doc.text[lo:hi].strip()


def build_calendar(bundles, today: date) -> list[CalendarEvent]:
    events: list[CalendarEvent] = []

    for bundle in bundles:
        contract = bundle.contract
        name = contract.counterparty or contract.title
        cid = contract.id

        # 1. provenance: when each file arrived
        for doc in bundle.docs:
            if doc.ingested_at is None:
                continue
            stamp = doc.ingested_at
            events.append(CalendarEvent(
                id=f"{cid}:upload:{doc.id}",
                date=stamp.date() if isinstance(stamp, datetime) else stamp,
                kind="uploaded", source="system",
                title=f"{doc.filename} received",
                detail=(f"{len(doc.text):,} characters, {len(doc.pages)} page(s)"
                        + (", read by OCR" if doc.used_ocr else "")),
                contract_id=cid, contract=name, file=doc.filename,
                meta={"sha256": doc.sha256[:12]},
            ))

        # 2. contract-level milestones
        if contract.effective_date:
            events.append(CalendarEvent(
                id=f"{cid}:effective", date=contract.effective_date,
                kind="effective", source="computed",
                title=f"{name} agreement starts",
                detail="Effective Date, resolved across the document family.",
                contract_id=cid, contract=name,
            ))

        # 3. computed deadlines -- the ones with consequences
        for ob in bundle.obligations:
            rule = next((r for r in bundle.rules if r.id == ob.rule_id), None)
            doc = next((d for d in bundle.docs
                        if rule and d.id == rule.span.doc_id), None)
            events.append(CalendarEvent(
                id=f"{cid}:ob:{ob.rule_id}:{ob.due_date}", date=ob.due_date,
                kind=ob.kind, source="computed",
                title=f"{KIND_LABEL.get(ob.kind, ob.kind)} — {name}",
                detail=ob.description or "",
                contract_id=cid, contract=name,
                file=doc.filename if doc else None,
                quote=rule.span.quote if rule else None,
                start=rule.span.char_start if rule else None,
                end=rule.span.char_end if rule else None,
                actionable=True,
                meta={"anchor": ob.anchor, "owed_by": ob.owed_by,
                      "derivation": ob.derivation},
            ))

        # 4. dates written in the text
        for doc in bundle.docs:
            for parsed, span in literal_dates(doc):
                events.append(CalendarEvent(
                    id=f"{cid}:lit:{doc.id}:{span.char_start}", date=parsed,
                    kind="mentioned", source="quoted",
                    title=f"Date written in {doc.filename}",
                    detail=_line_around(doc, span.char_start, span.char_end)[:220],
                    contract_id=cid, contract=name, file=doc.filename,
                    quote=span.quote, start=span.char_start, end=span.char_end,
                    page=doc.page_for(span.char_start),
                ))

    events.sort(key=lambda e: (e.date, e.kind))
    return events


def in_range(events: list[CalendarEvent], start: date | None,
             end: date | None) -> list[CalendarEvent]:
    return [e for e in events
            if (start is None or e.date >= start) and (end is None or e.date <= end)]


def by_month(events: list[CalendarEvent], today: date) -> list[dict[str, Any]]:
    """Grouped for a month-by-month view."""
    months: dict[str, list[CalendarEvent]] = {}
    for event in events:
        months.setdefault(event.date.strftime("%Y-%m"), []).append(event)
    out = []
    for key in sorted(months):
        bucket = months[key]
        out.append({
            "month": key,
            "label": date(int(key[:4]), int(key[5:]), 1).strftime("%B %Y"),
            "count": len(bucket),
            "actionable": sum(1 for e in bucket if e.actionable),
            "events": [e.to_dict(today) for e in bucket],
        })
    return out


def summary(events: list[CalendarEvent], today: date) -> dict[str, Any]:
    actionable = [e for e in events if e.actionable]
    return {
        "total": len(events),
        "actionable": len(actionable),
        "overdue": sum(1 for e in actionable if e.days_from(today) < 0),
        "next_30": sum(1 for e in actionable if 0 <= e.days_from(today) <= 30),
        "next_90": sum(1 for e in actionable if 0 <= e.days_from(today) <= 90),
        "documents": sum(1 for e in events if e.kind == "uploaded"),
        "written_in_documents": sum(1 for e in events if e.source == "quoted"),
        "computed": sum(1 for e in events if e.source == "computed"),
    }
