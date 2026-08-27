"""The retrieval layer: verified records AND verbatim document passages.

Two indexes, one interface:

  * RECORDS  — the extracted layer (clauses, computed deadlines, findings,
               absences, gaps). Already verified, already comparable.
  * PASSAGES — the contract text itself, chunked at clause boundaries, each
               chunk carrying its absolute offsets.

Records answer "what is the liability cap" and "what is missing". Passages
answer "what does it actually say about X" for wording no extractor captured
as a clause. Both are grounded by construction: a passage IS a slice of the
document, so anything retrieved can be quoted with real offsets.

Groq serves no embedding model, so ranking is BM25 plus structured boosts.
`Retriever.search` is the seam: swapping in vectors means changing the ranking
inside it and nothing above it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any, Literal

from api.bm25 import BM25, tokenize
from api.chunking import section_boundaries
from api.schemas import Document

PASSAGE_CHARS = 700
PASSAGE_OVERLAP = 120
MIN_PASSAGE_CHARS = 80


@dataclass
class Passage:
    """A verbatim slice of a real document."""

    id: str
    contract_id: str
    contract: str
    doc_id: str
    file: str
    start: int
    end: int
    text: str
    page: int | None = None
    section: str | None = None

    def citation(self) -> dict[str, Any]:
        return {
            "record_id": self.id,
            "contract_id": self.contract_id,
            "contract": self.contract,
            "kind": "passage",
            "title": self.section or f"{self.file} passage",
            "quote": self.text,
            "file": self.file,
            "start": self.start,
            "end": self.end,
            "page": self.page,
        }

    def for_model(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "contract": self.contract,
            "kind": "passage",
            "title": self.section or self.file,
            "detail": f"Verbatim text from {self.file}"
                      + (f", page {self.page}" if self.page else ""),
            "verbatim_quote": self.text,
        }


def _section_label(text: str, start: int) -> str | None:
    """Nearest preceding heading, so a passage can say where it came from."""
    head = text.rfind("\n", 0, start)
    window = text[max(0, start - 400):start]
    for line in reversed(window.split("\n")):
        line = line.strip()
        if 3 < len(line) < 90 and (line[0].isdigit() or line.isupper()):
            return line
    return None


def _merged_sections(text: str) -> list[tuple[int, int]]:
    """Section spans, with short ones merged FORWARD into the next.

    A bare heading ("4. TERMINATION") is 16 characters, and BM25 length
    normalisation makes it outrank the clause underneath it. Dropping short
    chunks fixed the ranking but lost text -- "3.4 Late Payment. Overdue
    amounts accrue interest at 1.5% per month." is 70 characters and became
    unretrievable. Merging keeps every character in exactly one passage.
    """
    bounds = section_boundaries(text)
    spans: list[tuple[int, int]] = []
    pending: int | None = None
    for lo, hi in zip(bounds, bounds[1:]):
        start = lo if pending is None else pending
        if hi - start < MIN_PASSAGE_CHARS:
            pending = start           # too small on its own; carry it forward
            continue
        spans.append((start, hi))
        pending = None
    if pending is not None and pending < len(text):
        if spans:                     # tail is short: fold into the last span
            spans[-1] = (spans[-1][0], len(text))
        else:
            spans.append((pending, len(text)))
    return spans


def chunk_document(doc: Document, contract_id: str, contract: str) -> list[Passage]:
    """Split at clause boundaries, then window long sections. Offsets are
    absolute into the canonical document text and never re-derived."""
    text = doc.text
    passages: list[Passage] = []

    for lo, hi in _merged_sections(text):
        cursor = lo
        while cursor < hi:
            end = min(hi, cursor + PASSAGE_CHARS)
            if end < hi:                      # prefer a sentence boundary
                dot = text.rfind(". ", cursor + PASSAGE_CHARS // 2, end)
                if dot > cursor:
                    end = dot + 1
            chunk = text[cursor:end]
            if chunk.strip():
                passages.append(Passage(
                    id=f"p{len(passages) + 1}:{doc.id}",
                    contract_id=contract_id, contract=contract, doc_id=doc.id,
                    file=doc.filename, start=cursor, end=end, text=chunk,
                    page=doc.page_for(cursor),
                    section=_section_label(text, cursor),
                ))
            if end >= hi:
                break
            cursor = max(cursor + 1, end - PASSAGE_OVERLAP)
    return passages


# --------------------------------------------------------------------------

@dataclass
class Hit:
    kind: Literal["record", "passage"]
    id: str
    score: float
    payload: Any

    def for_model(self) -> dict[str, Any]:
        return self.payload.for_model()

    def citation(self) -> dict[str, Any]:
        return self.payload.citation()


class Retriever:
    """One search surface over both indexes."""

    def __init__(self, bundles, gaps, today: date):
        from api.ask import Index, build_records

        self.today = today
        self.records = build_records(bundles, gaps, today)
        self.record_index = Index(self.records)

        self.passages: list[Passage] = []
        for bundle in bundles:
            name = bundle.contract.counterparty or bundle.contract.title
            for doc in bundle.docs:
                self.passages.extend(
                    chunk_document(doc, bundle.contract.id, name))
        self.passage_bm25 = BM25([p.text + " " + (p.section or "") + " " + p.contract
                                  for p in self.passages])
        self.by_id: dict[str, Any] = {r.id: r for r in self.records}
        self.by_id.update({p.id: p for p in self.passages})

    # ---- search --------------------------------------------------------

    def search(
        self,
        query: str,
        k: int = 8,
        contract_id: str | None = None,
        include: tuple[str, ...] = ("record", "passage"),
    ) -> list[Hit]:
        hits: list[Hit] = []

        if "record" in include:
            for record, score in self.record_index.rank(
                query, top_k=k * 2, contract_id=contract_id
            ):
                hits.append(Hit("record", record.id, score, record))

        if "passage" in include:
            terms = tokenize(query)
            for i, passage in enumerate(self.passages):
                if contract_id and passage.contract_id != contract_id:
                    continue
                score = self.passage_bm25.score(i, terms)
                if score > 0:
                    # Records are verified conclusions; passages are raw text.
                    # Prefer a record when both answer, but keep passages
                    # available for wording no clause type covers.
                    hits.append(Hit("passage", passage.id, score * 0.75, passage))

        hits.sort(key=lambda h: -h.score)
        return _diversify(hits, k)

    def get(self, record_id: str) -> Any | None:
        return self.by_id.get(record_id)

    def citations(self, ids: list[str]) -> list[dict[str, Any]]:
        """Resolve ids to citations. Unknown ids are dropped, never invented."""
        out = []
        for rid in ids:
            item = self.by_id.get(rid)
            if item is not None:
                out.append(item.citation())
        return out

    @property
    def stats(self) -> dict[str, int]:
        return {"records": len(self.records), "passages": len(self.passages)}


def _diversify(hits: list[Hit], k: int, per_contract: int = 4) -> list[Hit]:
    """Stop one contract's passages crowding out every other contract.

    Comparison questions fail badly without this: the best-matching contract
    fills the whole result set and the model never sees what to compare against.
    """
    seen: dict[str, int] = {}
    out: list[Hit] = []
    for hit in hits:
        cid = getattr(hit.payload, "contract_id", "")
        if seen.get(cid, 0) >= per_contract:
            continue
        seen[cid] = seen.get(cid, 0) + 1
        out.append(hit)
        if len(out) >= k:
            break
    return out
