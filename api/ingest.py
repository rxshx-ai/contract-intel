"""PDF -> canonical text with reliable character offsets.

THE RULE: `Document.text` is canonical and immutable. Every offset in the
system indexes into it. Normalization (ligatures, hyphenation, whitespace)
happens HERE, once, before any offset exists. No downstream module may
re-normalize text -- doing so silently invalidates every Span in the system.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
import uuid

from api.schemas import ContractType, Document, PageMark, Span

PAGE_SEP = "\n\n"

# Ligatures survive PDF text extraction and break exact-substring matching
# when a model helpfully "corrects" them in its quote.
_LIGATURES = {
    "ﬀ": "ff", "ﬁ": "fi", "ﬂ": "fl",
    "ﬃ": "ffi", "ﬄ": "ffl", "ﬅ": "st", "ﬆ": "st",
}
_QUOTES = {
    "‘": "'", "’": "'", "‚": "'", "‛": "'",
    "“": '"', "”": '"', "„": '"',
    "–": "-", "—": "-", "−": "-", " ": " ",
}

_HYPHEN_BREAK = re.compile(r"(\w)-\n(\w)")
_TRAILING_WS = re.compile(r"[ \t]+\n")
_MANY_BLANKS = re.compile(r"\n{3,}")
_MANY_SPACES = re.compile(r"[ \t]{2,}")


def normalize(raw: str) -> str:
    """Canonicalize page text. MUST run before any offset is computed."""
    text = unicodedata.normalize("NFKC", raw)
    for src, dst in {**_LIGATURES, **_QUOTES}.items():
        text = text.replace(src, dst)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _HYPHEN_BREAK.sub(r"\1\2", text)   # "termi-\nnation" -> "termination"
    text = _TRAILING_WS.sub("\n", text)
    text = _MANY_SPACES.sub(" ", text)
    text = _MANY_BLANKS.sub("\n\n", text)
    return text.strip()


def _guess_type(filename: str, text: str) -> ContractType:
    """Title block wins over body text: an MSA that merely *mentions* an Order
    Form is still an MSA."""
    checks = [
        (ContractType.AMENDMENT, ("amendment", "addendum")),
        (ContractType.DPA, ("data processing agreement", "data protection addendum")),
        (ContractType.NDA, ("non-disclosure", "nondisclosure", "confidentiality agreement")),
        (ContractType.SOW, ("statement of work", "scope of work")),
        (ContractType.ORDER_FORM, ("order form", "order schedule")),
        (ContractType.MSA, ("master services agreement", "master service agreement",
                            "master subscription agreement", "terms of service")),
    ]
    title = text[:200].lower()
    name = filename.lower()
    body = text[:4000].lower()
    for haystack in (title, name, body):
        for ctype, needles in checks:
            if any(n in haystack for n in needles):
                return ctype
    return ContractType.UNKNOWN


def _needs_ocr(pages: list[str]) -> bool:
    """A text layer that is mostly empty means a scanned document."""
    if not pages:
        return True
    chars = sum(len(p.strip()) for p in pages)
    return chars / max(len(pages), 1) < 120


def _ocr_pages(path: str) -> tuple[list[str], list[float]] | None:
    """Best-effort OCR. Returns None when the toolchain is unavailable."""
    try:
        import pytesseract  # type: ignore
        from pdf2image import convert_from_path  # type: ignore
    except ImportError:
        return None
    texts, confs = [], []
    for image in convert_from_path(path, dpi=200):
        data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
        words = [w for w in data["text"] if w.strip()]
        scores = [int(c) for c, w in zip(data["conf"], data["text"]) if w.strip() and c != "-1"]
        texts.append(" ".join(words))
        confs.append(sum(scores) / len(scores) / 100 if scores else 0.0)
    return texts, confs


def _assemble(page_texts: list[str], confs: list[float | None]) -> tuple[str, list[PageMark]]:
    """Join normalized pages, recording exact offset ranges for each."""
    parts: list[str] = []
    marks: list[PageMark] = []
    cursor = 0
    for i, page in enumerate(page_texts):
        clean = normalize(page)
        if i > 0:
            parts.append(PAGE_SEP)
            cursor += len(PAGE_SEP)
        start = cursor
        parts.append(clean)
        cursor += len(clean)
        marks.append(
            PageMark(page=i + 1, char_start=start, char_end=cursor,
                     ocr_confidence=confs[i] if i < len(confs) else None)
        )
    return "".join(parts), marks


def ingest_pdf(path: str, filename: str | None = None) -> Document:
    import pdfplumber

    name = filename or path.rsplit("/", 1)[-1]
    raw_pages: list[str] = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            raw_pages.append(page.extract_text() or "")

    used_ocr = False
    confs: list[float | None] = [None] * len(raw_pages)
    if _needs_ocr(raw_pages):
        result = _ocr_pages(path)
        if result is not None:
            raw_pages, ocr_confs = result
            confs = list(ocr_confs)
            used_ocr = True

    text, marks = _assemble(raw_pages, confs)
    with open(path, "rb") as fh:
        digest = hashlib.sha256(fh.read()).hexdigest()

    return Document(
        id=f"doc_{uuid.uuid4().hex[:10]}",
        filename=name,
        text=text,
        pages=marks,
        contract_type=_guess_type(name, text),
        used_ocr=used_ocr,
        sha256=digest,
    )


def ingest_text(raw: str, filename: str = "pasted.txt") -> Document:
    """Same canonical path for plain text, so tests never touch a PDF."""
    text, marks = _assemble([raw], [None])
    return Document(
        id=f"doc_{uuid.uuid4().hex[:10]}",
        filename=filename,
        text=text,
        pages=marks,
        contract_type=_guess_type(filename, text),
        sha256=hashlib.sha256(raw.encode()).hexdigest(),
    )


# --------------------------------------------------------------------------
# span location -- the bridge from model output back to grounded offsets
# --------------------------------------------------------------------------

_WS = re.compile(r"\s+")


def find_span(doc: Document, quote: str, hint: int = 0) -> Span | None:
    """Locate a model-supplied quote in the document and return exact offsets.

    The model returns quote TEXT only; it never supplies offsets, because
    language models cannot count characters reliably. We recover the offsets
    ourselves, which is what makes grounding trustworthy rather than asserted.

    Falls back to whitespace-insensitive matching, since models routinely
    reflow line breaks inside a quoted passage.
    """
    if not quote or not quote.strip():
        return None
    text = doc.text

    idx = text.find(quote, hint)
    if idx == -1 and hint:
        idx = text.find(quote)
    if idx != -1:
        return Span(doc_id=doc.id, char_start=idx, char_end=idx + len(quote), quote=quote)

    # whitespace-insensitive: build a regex from the quote's non-space tokens
    tokens = [re.escape(t) for t in _WS.split(quote.strip()) if t]
    if not tokens:
        return None
    match = re.search(r"\s+".join(tokens), text)
    if match is None:
        return None
    return Span(
        doc_id=doc.id,
        char_start=match.start(),
        char_end=match.end(),
        quote=text[match.start() : match.end()],  # the REAL text, not the model's
    )


# --------------------------------------------------------------------------
# fuzzy recovery -- for models that paraphrase inside a "verbatim" quote
# --------------------------------------------------------------------------

FUZZY_THRESHOLD = 0.85
_ANCHOR_WORDS = 5


def _anchor_candidates(tokens: list[str]) -> list[list[str]]:
    """Try the start, middle and end of the quote as anchors, in that order."""
    n = len(tokens)
    if n <= _ANCHOR_WORDS:
        return [tokens]
    mid = max(0, (n - _ANCHOR_WORDS) // 2)
    return [
        tokens[:_ANCHOR_WORDS],
        tokens[mid : mid + _ANCHOR_WORDS],
        tokens[-_ANCHOR_WORDS:],
    ]


def find_span_fuzzy(
    doc: Document, quote: str, threshold: float = FUZZY_THRESHOLD
) -> Span | None:
    """Locate a near-miss quote and return the DOCUMENT's text for that range.

    Weaker models drop a word, fix a typo, or expand an abbreviation inside
    text they were told to copy verbatim. Dropping those extractions costs
    recall for no safety benefit, because we never surface the model's string:
    the returned Span always quotes the document itself.

    Invariant 1 is therefore untouched -- verify.py still checks
    `text[start:end] == quote`, and that holds by construction here. The risk
    this guards against is different: anchoring to the WRONG passage. Hence a
    high similarity floor and a length-plausibility check.
    """
    import difflib

    if not quote or not quote.strip():
        return None
    text = doc.text
    tokens = [t for t in _WS.split(quote.strip()) if t]
    if not tokens:
        return None

    target_len = len(quote)
    window = int(target_len * 1.6) + 40
    best: tuple[float, int, int] | None = None

    for anchor in _anchor_candidates(tokens):
        pattern = re.compile(r"\s+".join(re.escape(t) for t in anchor), re.IGNORECASE)
        for match in pattern.finditer(text):
            lo = max(0, match.start() - window)
            hi = min(len(text), match.start() + window)
            region = text[lo:hi]
            matcher = difflib.SequenceMatcher(None, region, quote, autojunk=False)
            # Derive BOTH bounds from the alignment, so a document phrase that is
            # longer than the model's quote ("forty-five (45) days" vs "45 days")
            # is not truncated.
            blocks = [b for b in matcher.get_matching_blocks() if b.size >= 4]
            if not blocks:
                continue
            start = lo + blocks[0].a
            end = lo + blocks[-1].a + blocks[-1].size
            candidate = text[start:end]
            if not candidate.strip():
                continue
            if not (0.5 * target_len <= len(candidate) <= 2.0 * target_len):
                continue
            ratio = difflib.SequenceMatcher(None, candidate, quote,
                                            autojunk=False).ratio()
            if ratio >= threshold and (best is None or ratio > best[0]):
                best = (ratio, start, end)

    if best is None:
        return None
    _, start, end = best
    return Span(doc_id=doc.id, char_start=start, char_end=end, quote=text[start:end])


def locate(doc: Document, quote: str) -> tuple[Span | None, str]:
    """Exact -> whitespace-insensitive -> fuzzy. Returns (span, how)."""
    span = find_span(doc, quote)
    if span is not None:
        return span, "exact" if span.quote == quote else "whitespace"
    span = find_span_fuzzy(doc, quote)
    return (span, "fuzzy") if span is not None else (None, "dropped")
