# Contract Risk & Obligation Intelligence

Contract analysis where **every claim is quoted, every number is computed, and
ungrounded output cannot reach the screen**.

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python eval/seed_cache.py                   # install the shipped extractions
.venv/bin/python -m pytest tests/ -q                  # 237 tests, no network needed
.venv/bin/python eval/demo_run.py                     # the five demo beats
.venv/bin/uvicorn api.main:app --reload --port 8077   # then open localhost:8077
```

`demo_cache/` holds **real `openai/gpt-oss-120b` output** for the six demo
documents, committed so a clone runs the whole product — analysis, calendar,
retrieval, agent — with no API key. The live cache stays untracked, because it
fills with extractions of whatever you upload and those are your documents.

No API key is needed for the demo — extractions are cached by document hash.
To analyse new documents, or re-extract the corpus:

```bash
export GROQ_API_KEY=gsk_...
.venv/bin/python eval/smoke_groq.py        # one call, verifies the provider
.venv/bin/python eval/extract_all.py --force  # re-extract the corpus (~8 min)
.venv/bin/python eval/run_eval.py --cached    # score it against gold
```

**Free-tier limits shape the design.** Groq free tier allows 8,000 TPM, and the
`max_tokens` reservation counts against it — so a big reservation alone will 413
a small request. `api/chunking.py` splits contracts at clause boundaries and
`llm.TokenBudget` throttles to the window. A cold run of the six-document corpus
is 9 requests over ~8 minutes; everything after that reads the cache.

**Model:** `openai/gpt-oss-120b` on Groq — 131k context, strict `json_schema`
support, ~500 tok/s. Override with `GROQ_MODEL`. The provider lives entirely in
`api/llm.py`; swapping it touches one file, because `extract.py` is the only
module that talks to a model (invariant 3).

**Measured accuracy** (real extraction vs. hand-authored gold, `eval/run_eval.py --cached`):

| | value |
|---|---|
| micro-average precision / recall / F1 | **0.84 / 0.91 / 0.88** |
| clause types covered | 28 |
| grounding rate | **100%** (86 spans, 0 discarded) |
| hallucination rate | 0, by construction |

Weakest types are `payment_terms` and `auto_renewal` (F1 0.67) — both
precision failures where the model splits one clause into two overlapping
quotes, not invented content.

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

## Making an open-weight model behave

Four things were needed to get `gpt-oss-120b` to a usable extraction, each
found by a real failure rather than anticipated:

1. **Every wire field must be nullable.** Groq strict mode marks all fields
   required, and a model with nothing to say emits `null`. A single
   `currency: str = "USD"` cost the *entire chunk* with
   `'/currency' … expected string, but got null`.
2. **No enums on the wire.** The model writes `"party_favored": "Customer"`
   where the schema said `"us"`, and strict mode rejects the whole response
   over one nested value. Enums are plain strings on the wire and
   `normalize_party()` maps them back — resolving role words against which side
   of the paper we are on, so "Customer" means *us* when we are the buyer and
   *the counterparty* when we are the seller.
3. **The prompt must enumerate valid values and demand the numeric fields.**
   Without an explicit field table the model returned quotes with every field
   null, which silently zeroes the entire reasoning layer. Without an enumerated
   anchor list it invented `quarter_end`, `anniversary`, and `audit` — two of
   which turned out to be legitimate and are now first-class anchors.
4. **Unknown anchors become `event`, not a drop.** An obligation we cannot put
   on a calendar is still a real obligation; `temporal.py` reports it as
   conditional. Dropping it would hide exactly what this product exists to find.

### Grounding

Open models also paraphrase inside text they were told to copy verbatim.
`locate()` escalates through three stages and records which one hit:

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
| `api/llm.py` | Groq client, strict-schema rewrite, TPM throttle, retry. The only provider-aware file. |
| `api/chunking.py` | Splits contracts at clause boundaries; chunks keep absolute offsets. |
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

## Ask

Questions answered from the **extracted layer**, never from the raw contract.

A document is analysed once at upload into verified clauses, computed
deadlines, findings and absences. Ask retrieves over those records, so:

- an answer can only be assembled from things already verified — invariant 1
  extends to Q&A for free;
- the model emits **record IDs, not quotes**, so it cannot fabricate a
  citation. Any id it invents is dropped before the answer is returned;
- portfolio questions work ("where are we exposed across contracts?") because
  the records are structured and comparable. Retrieval over PDF chunks cannot
  answer that;
- when the records do not cover the question, it says so and names what is
  missing rather than filling the gap from general knowledge.

Retrieval is BM25 over record text plus structured boosts (a renewal question
ranks the `term_end` obligation, not a quarterly report that happens to share
the word "deadline"). Groq serves no embedding model; for a few hundred short,
highly structured records this is also more precise than vectors. Swapping in
embeddings means changing `Index.rank()` and nothing else.

```
POST /ask   question=... [&contract_id=...]
```

## Retrieval layer (RAG)

Two indexes behind one search surface (`api/rag.py`):

| Index | What it holds |
|---|---|
| **Records** | the extracted layer — clauses, computed deadlines, findings, absences, gaps. Already verified and comparable. |
| **Passages** | the contract text, chunked at clause boundaries, each carrying absolute offsets. |

Records answer *"what is the liability cap"*; passages answer wording no clause
type covers — "Overdue amounts accrue interest at 1.5% per month" was never
extracted as a clause but is retrievable. Both are grounded: a passage IS a
slice of the document, so anything retrieved can be quoted with real offsets.
Two properties are tested: **no text is lost** between passages, and short
sections are merged forward rather than dropped (a bare 16-char heading
outranks real clauses under BM25 length normalisation).

Ranking is BM25 plus structured boosts, with per-contract diversification so a
comparison question does not get one contract's passages filling every slot.
Groq serves no embedding model; swapping in vectors means changing
`Retriever.search` and nothing above it. `GET /rag/search?q=...`

## Agent

`api/agent.py` plans, calls tools, and answers with citations.
`POST /agent/ask`

- **Plans first.** The plan is a separate structured call, not a tool — forcing
  `tool_choice` to a named function makes gpt-oss emit a call to an internal
  channel name (`commentary`, `json`) that Groq then rejects. The plan is shown
  to the user.
- **Comparison is a tool, not a judgement.** `compare` returns a computed table
  across contracts with the wording behind every figure. Party-aware: a 99.99%
  uptime commitment is a protection from a supplier and an exposure to a
  customer, so ranking flips with which side we are on and never mixes the two
  into one league table.
- **Silence is reported.** A contract that states nothing appears under
  `not_stated`, never as a good value.
- **Citations cannot be faked.** `finish` returns record ids; ids that do not
  resolve are dropped.
- **Bounded.** Repeat calls reuse the earlier result, steps are capped, and
  every call goes through the shared TPM budget.
- **Salvage.** When a finish call is emitted under the wrong tool name, the
  arguments are recovered rather than losing a completed run. This fires in
  practice — the UI labels it "recovered from a malformed tool call".

Tools: `search` · `compare` · `deadlines` · `calendar` · `contract_facts` ·
`exit_cost` · `finish`

## Calendar

`api/calendar.py`, `GET /calendar` — three kinds of date, deliberately kept apart:

| Source | Meaning |
|---|---|
| `system` | when we ingested the file |
| `computed` | deadlines derived from relative wording. None appear in any document |
| `quoted` | dates written literally in the text, grounded to a span |

**A date in the text is not a deadline.** Conflating them is how "the contract
says 31 December" becomes a notice window that actually closed 60 days earlier.
Only computed events are marked actionable.

## Front end

The UI is the Claude Design export, wired to the API rather than rebuilt:

```
web/app/design.dc.html   pristine export, never edited
web/app/wire.py          transforms it into a live page
web/app/index.html       generated — do not edit by hand
```

`python web/app/wire.py` applies ~32 assertive patches and fails loudly if the
design is re-exported and a patched region moved. It also node-syntax-checks
the result, because a patch that concatenates a `//` comment onto one line
silently comments out the rest of the method — which is how it broke first time.

What wiring involved beyond swapping data in:

- the design's ~36KB mock corpus is **deleted**, not renamed. It has no business
  shipping in a page whose whole claim is that everything on screen is quoted
  from a real file;
- every hardcoded figure in the design's prose ("Nineteen things across six
  contracts", "96.4%", "Measured on 148 contracts") became a binding computed
  from real data;
- `quote()` now uses the offsets the verifier checked instead of `indexOf`, so
  repeated wording highlights the right occurrence;
- exit cost comes from `/contracts/{id}/termination-cost` instead of the
  design's invented constants;
- React is vendored locally, so the page never reaches for a CDN mid-demo.

`/raw` still serves the minimal test harness.

## Test upload

`contracts/meridian_msa.pdf` (generated by `eval/make_test_contract.py`) is a
4-page vendor MSA the model has never seen, built so a correct analysis has to
find specific things:

| It should find | Because |
|---|---|
| renewal notice deadline **2027-10-04** | the document contains no dates — 120 days before a 24-month term from 2026-02-01 |
| 4 new flow-down gaps vs. the Acme contract | 99.5% vs 99.99% uptime, 120h vs 24h breach notice, 180d vs 30d deletion, $25k vs $5M cap |
| missing confidentiality + termination-for-cause | deliberately omitted |
| 6 dark patterns | unilateral amendment, 120-day window, sole-discretion pricing, fee acceleration, one-way exit, uneconomic remedy ($25k cap + exclusive Singapore venue) |
| liability 70 / operational 75 | $25k cap against $216k annual value |

First upload runs live extraction (~3 min on free tier, with automatic
chunk-splitting if a section overflows the output budget). Re-uploading the same
file is **instant** — the cache is keyed by content hash — so it is safe to
demo live.

Uploaded contracts live in memory; a server restart reloads only the demo
portfolio. The extraction cache survives, so re-uploading is immediate.

## Storage and vectors

Both are optional. Unset, the service runs on SQLite with lexical retrieval —
which is what the tests and the offline demo use.

| Variable | Effect when set |
|---|---|
| `DATABASE_URL` | Postgres instead of SQLite |
| `PINECONE_API_KEY` | Semantic retrieval fused with BM25 |

### Postgres

`api/store.py` speaks both engines behind one interface. The interesting change
is not the engine — it is that **the analysis now survives a restart**.
Previously `_state["bundles"]` lived only in memory, so an uploaded contract
died with the process. Contracts are stored in their analysed form, so a fresh
process rehydrates without re-extracting and **without spending a token**.

Verified against real Postgres 15: upload in one process, restart, contract and
its claims/findings/deadlines all present, quotes still grounded.

```bash
docker compose up -d
export DATABASE_URL=postgresql://postgres:postgres@localhost:5433/contract_intel
```

Every query filters on `tenant_id`, and the tenant is bound at construction
rather than passed per call — an argument you must remember to pass is an
isolation bug waiting to happen. There is a test that one tenant cannot read
another's contracts.

### Pinecone

`api/vectors.py`. Vectors sit **alongside** BM25, not in place of it, because
each fails where the other works:

- **BM25** — exact terms and numbers. "99.9%", "forty-five (45) days". An
  embedding will happily rank 99.99% next to 99.9%, which is precisely the
  distinction a contract question turns on.
- **Vectors** — paraphrase. "what if they go bust" finds insolvency and
  termination wording that shares no words with the question. BM25 returns
  nothing.

They are fused with **Reciprocal Rank Fusion**, which uses position only. BM25
scores are unbounded and Pinecone's are bounded; blending them numerically
would need a calibration that drifts the moment either side changes.

Embeddings use **Pinecone integrated inference** — text is upserted and Pinecone
embeds it server-side. That keeps the credential count at one, which matters
because Groq serves no embedding model and the alternative was adding OpenAI or
Cohere purely to make vectors work.

```bash
export PINECONE_API_KEY=...
curl -X POST localhost:8077/vectors/sync    # index the extracted layer
curl localhost:8077/system                  # what this process is backed by
```

Without a key the module is inert — every method is a no-op, so callers never
branch on availability, and retrieval falls back to BM25 alone.

**What is verified, and what is not.** The lexical half runs in the product
today. The fusion path is driven in tests by a stub vector index, which covers
the ranking merge, grounding of fused results, scope filtering, the
`include` filter, fallback when the vector store returns nothing, and dropping
ids the vector store does not recognise. What has **never executed** is the
Pinecone HTTP call itself — no key has been available. Expect index creation
and the first `/vectors/sync` to need debugging.

**Is a vector DB warranted here?** Honestly, at 200 items for 4 contracts, no —
BM25 answers in microseconds. At ~15,000 items for 300 contracts it is still not
a scale problem. Vectors earn their place for *paraphrase recall*, not speed.

## Deploying to AWS

```bash
aws configure                      # you must do this; I will not handle credentials
open -a Docker                     # daemon must be running
export GROQ_API_KEY=gsk_...        # optional: demo works without it
./deploy/aws-apprunner.sh
```

**App Runner, single instance.** Not Lambda, and not an autoscaled ECS service —
this app keeps analysis state in memory (`api/main.py` `_state`), writes SQLite
to the working directory, and caches extractions on local disk. Two replicas
would answer from two different sets of contracts. Externalising that state to
Postgres and S3 is real work, not a deployment flag.

What the script does, all from the CLI: creates the ECR repository, builds the
image **pinned to `linux/amd64`** (App Runner is x86_64; an Apple Silicon
default build fails to start with an exec-format error), pushes it, creates the
IAM role App Runner needs to pull from ECR, creates or redeploys the service,
waits for `RUNNING`, and prints the HTTPS URL.

The image runs `eval/seed_cache.py` at build time and then asserts the demo
loads, so a broken image fails the build rather than deploying. `GROQ_API_KEY`
is passed as a runtime environment variable and never baked into the layer.

Rough cost: App Runner bills provisioned memory continuously (~$10/month at
2 GB) and vCPU only while serving requests, so an idle demo is cheap. Pause it
between sessions:

```bash
aws apprunner pause-service --region us-east-1 --service-arn <arn>
```

Cheaper alternatives if you would rather manage the box: EC2 `t4g.small`
(~$12/month, and arm64 means a native build on an Apple Silicon Mac) or a
Lightsail container service at a flat $10/month — both need you to handle TLS
and process supervision yourself.

Set `DATABASE_URL` to an RDS instance (`deploy/aws-rds.sh` creates one) before
deploying anything you will show people. Without it the service uses SQLite on
the container's ephemeral disk and loses uploads on every recycle.

## Known gaps

Honest list, since the deliverable is decision support:

- **Precision is 0.84** — the model sometimes emits one clause as two
  overlapping quotes. `_dedupe()` catches same-type overlaps within a document
  but not near-duplicates the model files under different clause types.
- **Recall is 0.91** — five gold clauses were missed across four documents.
- **The fuzzy realignment stage has never fired on real output.** All 86 live
  spans matched exactly. It is tested with synthetically mangled quotes, so it
  works, but it is insurance rather than a load-bearing path today.
- **Only 4 of 6 documents are scored** by the eval; the Order Form and
  Amendment have gold fixtures but are not in `EVAL_SET`.
- **Free tier makes cold extraction slow** (~8 min for six documents). A paid
  tier with a higher TPM removes the throttling entirely — set `GROQ_TPM_LIMIT`.
- Supersession matches on clause *type*, not section number.
- **OCR is best-effort** — requires `pytesseract` + `pdf2image`, absent here.
  Falls back to the text layer and reports `used_ocr=False` rather than failing.
- Supersession matches on clause *type*, not section number, so an amendment
  that changes one of several same-type clauses supersedes the wrong one.
- Defined-term resolution, cross-references, and incorporation-by-reference are
  not implemented.
- No connectors (DocuSign/Drive/email) — the real go-to-market blocker.
