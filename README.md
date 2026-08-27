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
To analyze new documents:

```bash
export GROQ_API_KEY=gsk_...
.venv/bin/python eval/smoke_groq.py      # verify the provider works
.venv/bin/python eval/run_eval.py        # score it against the gold fixtures
```

**Model:** `openai/gpt-oss-120b` on Groq — 131k context, 65k max output, strict
`json_schema` support, ~500 tok/s. Override with `GROQ_MODEL`. The provider
lives entirely in `api/llm.py`; swapping it touches one file, because
`extract.py` is the only module that talks to a model (invariant 3).

Note the extraction cache is keyed by `(document SHA-256, party, model)`, so
after changing `GROQ_MODEL` re-run `eval/make_fixtures.py --seed-cache`.

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
   because language models cannot count characters; `locate()` recovers them.
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

## Grounding an open-weight model

Open models paraphrase inside text they were told to copy verbatim — dropping a
word, normalizing "forty-five (45) days" to "45 days". Discarding those costs
recall for no safety benefit, so `locate()` escalates through three stages and
records which one hit:

| Stage | What it handles |
|---|---|
| `exact` | byte-identical substring |
| `whitespace` | model reflowed line breaks inside the quote |
| `realigned` | model dropped/altered words — recovered by difflib alignment, ≥0.85 similarity + length plausibility |
| `discarded` | not in the document. Dropped. |

**This does not weaken invariant 1.** A recovered Span always carries the
*document's* text, never the model's, so `verify.py`'s exact-substring check
still passes by construction. The risk it guards against is different —
anchoring to the wrong passage — which is what the similarity floor is for.
Provenance is reported, not hidden: `/portfolio/stats` and the eval harness both
print the exact/reflowed/realigned/discarded split.

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
| `api/llm.py` | Groq client; Pydantic → strict `json_schema` rewrite. The only provider-aware file. |
| `api/extract.py` | The only model caller. Structured output, temperature 0, cached by SHA-256. |
| `api/verify.py` | The grounding gate. Drops anything unquotable. |
| `api/family.py` | Document graph, amendment supersession, lineage. |
| `api/temporal.py` | Rules → dated obligations with derivation chains. |
| `api/risk.py` | Party-aware rubric; every point names its clause. |
| `api/findings/` | silence · asymmetry · adversarial · backtoback · termination |
| `api/pipeline.py` | Orchestration; the module order written down once. |

## Verification

```bash
.venv/bin/python -m pytest tests/ -q      # 124 tests, none touch the network
.venv/bin/python eval/run_eval.py --self  # harness self-check (must score 1.00)
.venv/bin/python eval/smoke_groq.py       # one real Groq call (needs GROQ_API_KEY)
.venv/bin/python eval/run_eval.py         # live accuracy against gold fixtures
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

- **Extraction has never run live here** — no `GROQ_API_KEY` was available.
  `api/llm.py` is written against Groq's documented strict `json_schema` API and
  its schema rewriting is unit-tested, but no real completion has been made. The
  demo runs on hand-authored fixtures seeded into the cache. Run
  `eval/smoke_groq.py` first; it is built for exactly this check.
- **Grounding rate on a real open-weight model is unmeasured.** The fuzzy
  recovery stage exists because `gpt-oss-120b` is expected to paraphrase more
  than a frontier model would, but the actual exact/realigned/discarded split is
  unknown until the smoke test runs. If it comes back below ~80%, try
  `GROQ_MODEL=moonshotai/kimi-k2-instruct` or tighten the prompt.
- **No chunking.** Documents over ~380k characters are rejected with a clear
  error rather than silently truncated.
- **OCR is best-effort** — requires `pytesseract` + `pdf2image`, absent here.
  Falls back to the text layer and reports `used_ocr=False` rather than failing.
- Supersession matches on clause *type*, not section number, so an amendment
  that changes one of several same-type clauses supersedes the wrong one.
- Defined-term resolution, cross-references, and incorporation-by-reference are
  not implemented.
- No connectors (DocuSign/Drive/email) — the real go-to-market blocker.
