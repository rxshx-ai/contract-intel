# Contract Risk & Obligation Intelligence

Contract analysis where **every claim is quoted, every number is computed, and
ungrounded output cannot reach the screen**.

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python eval/make_fixtures.py --seed-cache   # seed the offline demo
.venv/bin/python eval/demo_run.py                     # the five demo beats
.venv/bin/uvicorn api.main:app --reload --port 8077   # then open localhost:8077
```

No API key is needed for the demo — extractions are cached by document hash.
Set `ANTHROPIC_API_KEY` to analyze new documents.

## What it is for

The commercial value is in **dates and obligations**, not risk scores. The
target buyer is procurement/finance at a 200–800 person company whose current
system is a Drive folder and someone's memory, and who is paying right now for
something they meant to cancel. Risk scoring exists to make one contract's
problems legible in five seconds; the auto-renewal deadline is what gets paid
for.

Positioning: *it never replaces legal review — it decides what gets reviewed,
and it never lets a date slip.*

## The five invariants

Every design decision descends from these.

1. **Nothing surfaces that isn't quoted.** Every claim carries a verbatim span
   with character offsets. `verify.py` runs an exact-substring check; anything
   that fails is **discarded, not flagged**. Ungrounded output is structurally
   unable to render, which is why `hallucination_rate = 0` is a property of the
   architecture rather than a claim about the model.
2. **The LLM never does arithmetic or logic.** It converts language into
   structured claims. Dates, scores, comparisons and aggregations are
   deterministic Python. The model returns *quotes only* — never offsets,
   because language models cannot count characters; `find_span` recovers them.
3. **One module talks to the model** (`extract.py`). Everything downstream is
   pure and tested with zero network.
4. **Document text is untrusted input**, fenced with an unguessable nonce and
   never concatenated into the instruction context.
5. **Absence is a finding.** A missing liability cap is worse than a bad one.

## What it does that other tools don't

| Feature | Why it's hard elsewhere |
|---|---|
| **Temporal obligation compiler** | Contracts contain *rules*, not dates: "60 days prior to the end of the then-current Term". The rule is extracted; the date is computed, with an auditable derivation chain. |
| **Silence detection** | You cannot quote a clause that doesn't exist. Keyword search can't see it and extractors never mention it. Playbook diff per contract type. |
| **Back-to-back gap analysis** | Requires normalized structured fields across a *portfolio*. Impossible for a chatbot or single-document analyzer. |
| **Contract family + supersession** | A contract is a stack (MSA → Order Form → amendments). Analyze the MSA alone and you report a cap that stopped being true two years ago. |
| **Prompt-injection firewall** | Documents are adversarially authored. Hidden instructions are reported as a *tampering indicator on the counterparty* — the attack becomes a finding about the vendor. |
| **Power asymmetry index** | Comparative, not absolute, so it's honest — and it yields the negotiation ask list for free. |

## Architecture

```
ingest → firewall → extract → verify → family → temporal → risk → findings
```

`schemas.py` is the frozen contract between every module. `temporal`, `risk`,
`family` and everything in `findings/` are pure consumers of the extracted
layer — which is what makes them testable without an API key and what makes the
whole reasoning layer reproducible.

| Module | Responsibility |
|---|---|
| `api/ingest.py` | PDF/text → canonical text + offsets. Normalizes **once**, before any offset exists. |
| `api/firewall.py` | Invisible/tiny/off-page text, metadata payloads, injection language; nonce fencing. |
| `api/extract.py` | The only model caller. Structured output via `messages.parse`, cached by SHA-256. |
| `api/verify.py` | The grounding gate. Drops anything unquotable. |
| `api/family.py` | Document graph, amendment supersession, lineage. |
| `api/temporal.py` | Rules → dated obligations with derivation chains. |
| `api/risk.py` | Party-aware rubric; every point names its clause. |
| `api/findings/` | silence · asymmetry · adversarial · backtoback · termination |
| `api/pipeline.py` | Orchestration; the module order written down once. |

## Verification

```bash
.venv/bin/python -m pytest tests/ -q      # 109 tests, one touches the network path
.venv/bin/python eval/run_eval.py --self  # harness self-check (must score 1.00)
.venv/bin/python eval/run_eval.py         # live accuracy (needs ANTHROPIC_API_KEY)
```

The property test `span.quote == text[start:end]` is invariant 1 mechanized and
runs against every line of every fixture. `eval/run_eval.py` prints
per-clause-type precision/recall/F1 and **shows the weak types** — a table with
three bad rows is more credible than one with none.

## Security posture

| Control | Status |
|---|---|
| Indirect prompt injection | Implemented — nonce fencing + classifier, pre-inference |
| Hidden-text tampering detection | Implemented — sub-4pt, background-colour, off-page, metadata |
| Append-only audit log | Implemented — every extraction and every view |
| Row-level tenant isolation | Schema-level from the first migration |
| PII/pricing redaction before egress | **Not implemented** — design only |
| Per-tenant encryption keys | **Not implemented** — design only |

## Known gaps

Honest list, since the deliverable is decision support:

- **Extraction has never run live here** — no `ANTHROPIC_API_KEY` was available.
  `extract.py` is written against the documented `messages.parse` API but is
  unexercised; the demo runs on hand-authored fixtures seeded into the cache.
  This is the one thing to verify first.
- **OCR is best-effort** — requires `pytesseract` + `pdf2image`, absent here.
  Falls back to the text layer and reports `used_ocr=False` rather than failing.
- Supersession matches on clause *type*, not section number, so an amendment
  that changes one of several same-type clauses supersedes the wrong one.
- Defined-term resolution, cross-references, and incorporation-by-reference are
  not implemented.
- No connectors (DocuSign/Drive/email) — the real go-to-market blocker.
