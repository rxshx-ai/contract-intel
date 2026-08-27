"""Split contracts at clause boundaries for extraction.

Two reasons, one of which is not about rate limits:

1. Free-tier TPM ceilings make a whole-contract request impossible.
2. Open-weight models extract better from small windows. Recall on a 26-clause
   MSA in one shot is materially worse than on six focused chunks.

Chunks carry their absolute offset into the canonical document text, so a quote
found in a chunk still resolves against the full document. Offsets never become
chunk-relative -- that would break every Span in the system.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from api.schemas import Document

# "11. LIMITATION OF LIABILITY", "11.1 Fees.", "SECTION 4", "Article VII"
_HEADING = re.compile(
    r"^(?:(?:SECTION|ARTICLE|Section|Article)\s+)?"
    r"(?:\d{1,2}(?:\.\d{1,2})*\.?|[IVXL]{1,5}\.)\s+\S",
    re.MULTILINE,
)

DEFAULT_MAX_CHARS = 1800
OVERLAP_CHARS = 320


@dataclass(frozen=True)
class Chunk:
    text: str
    start: int          # absolute offset into Document.text
    end: int
    index: int
    total: int

    @property
    def label(self) -> str:
        return f"part {self.index + 1} of {self.total}"


def section_boundaries(text: str) -> list[int]:
    """Offsets where a numbered section or article begins."""
    return sorted({0, *(m.start() for m in _HEADING.finditer(text)), len(text)})


def chunk_document(
    doc: Document,
    max_chars: int = DEFAULT_MAX_CHARS,
    overlap: int = OVERLAP_CHARS,
) -> list[Chunk]:
    """Group whole sections into chunks under `max_chars`.

    A section longer than the limit is emitted alone rather than cut mid-clause;
    splitting a clause in half loses it from both chunks.
    """
    text = doc.text
    if len(text) <= max_chars:
        return [Chunk(text=text, start=0, end=len(text), index=0, total=1)]

    bounds = section_boundaries(text)
    spans: list[tuple[int, int]] = []
    cursor = bounds[0]
    for nxt in bounds[1:]:
        if nxt - cursor >= max_chars:
            spans.append((cursor, nxt))
            cursor = nxt
    if cursor < len(text):
        spans.append((cursor, len(text)))

    # Merge trailing tiny spans into their predecessor.
    merged: list[tuple[int, int]] = []
    for start, end in spans:
        if merged and (end - start) < max_chars * 0.25 and \
                (end - merged[-1][0]) <= max_chars * 1.4:
            merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))

    total = len(merged)
    chunks: list[Chunk] = []
    for i, (start, end) in enumerate(merged):
        # Overlap backwards so a clause straddling a boundary appears whole in
        # the later chunk. Duplicates are removed after grounding.
        lo = max(0, start - overlap) if i > 0 else start
        chunks.append(Chunk(text=text[lo:end], start=lo, end=end,
                            index=i, total=total))
    return chunks


def estimate_tokens(text: str) -> int:
    """Legal English tokenizes densely (numbers, parentheses, capitals)."""
    return int(len(text) / 3.4) + 1
