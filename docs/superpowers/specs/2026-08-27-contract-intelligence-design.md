# Contract Risk & Obligation Intelligence — Design & Build Plan

Date: 2026-08-27
Window: 36 hours, team of 3
Stack: Python 3.11 / FastAPI / SQLite / Claude API / single-file HTML front end

---

## 1. Positioning

**One-liner:** It never replaces legal review — it decides what gets reviewed, and it never lets a date slip.

The commercial value is in **dates and obligations**, not risk scores. Risk scoring exists to make one contract's problems legible in five seconds on stage; the auto-renewal deadline is what a customer pays for. Build accordingly.

**Primary buyer:** procurement/finance at a 200–800 person company with 200–400 vendor agreements and 1–2 in-house lawyers.
**Incumbent competitor:** a Google Drive folder and someone's memory.

## 2. Non-goals

Explicitly out of scope, and we say so out loud:

- Legal advice or a sign/don't-sign recommendation. Decision support only.
- Contract authoring, redlining workflow, e-signature.
- DocuSign / Drive / email connectors. Named as the real go-to-market blocker in the pitch; not built.
- Multi-tenant SaaS infrastructure. Single-tenant, one SQLite file.
- Contract negotiation chat. A grounded Q&A tab is a stretch goal, never the core.

## 3. The five invariants

Every design decision below descends from these. They are also the pitch.

1. **Nothing surfaces that isn't quoted.** Every claim carries a verbatim span with character offsets. A verifier does an exact-substring check against source text; anything that fails is *discarded, not displayed*. Ungrounded output is structurally impossible to render.
2. **The LLM never does arithmetic or logic.** It converts language into structured claims. Dates, scores, comparisons, and aggregations are deterministic Python over those claims. Reproducible, unit-testable, auditable.
3. **One module talks to the model.** `extract.py`. Everything downstream consumes typed objects and is testable with fixtures and zero network.
4. **Document text is untrusted input.** It is never concatenated into the instruction context.
5. **Absence is a finding.** A missing liability cap is worse than a bad one. `missing_clause` is the only finding kind permitted to carry no evidence span.

## 4. Architecture

```
contract-intel/
  api/
    main.py          FastAPI app, routes, static mount
    schemas.py       ← THE CONTRACT BETWEEN ALL MODULES. Frozen at T+2.
    db.py            SQLite, inline schema, no ORM
    ingest.py        PDF → text + reliable char offsets; OCR fallback
    firewall.py      invisible-text + injection detection (pre-LLM)
    extract.py       Claude → ClauseClaim / TemporalRule  [only LLM caller]
    verify.py        exact-substring grounding check; drop-on-fail
    temporal.py      TemporalRule → materialized Obligation + derivation chain
    family.py        document graph; supersession resolution
    risk.py          rubric → RiskAxis scores with per-clause contributions
    findings/
      silence.py     playbook diff → missing_clause findings
      asymmetry.py   unilateral-rights tally → power index
      adversarial.py dark-pattern library match
      backtoback.py  cross-contract flow-down gap analysis
      termination.py exit-cost calculator
  eval/
    cuad_slice.py    held-out sample loader
    run_eval.py      per-clause-type P/R/F1 + grounding rate
  web/
    index.html       ~200 lines, no framework, no build step
  contracts/         sample PDFs incl. one poisoned
  tests/
```

**Dependency shape:** `ingest → firewall → extract → verify` produces the structured layer. `temporal`, `family`, `risk`, and everything in `findings/` are pure consumers of that layer. They can be written against fixtures before extraction works, which is what makes the parallel plan in §7 viable.

## 5. Data model (`schemas.py`)

Freeze this first. Everything else is downstream of it.

```python
class Span(BaseModel):
    doc_id: str
    char_start: int
    char_end: int
    quote: str                    # MUST be exact substring of doc text

class ClauseClaim(BaseModel):
    id: str
    contract_id: str
    clause_type: ClauseType       # enum, CUAD-aligned
    party_favored: Literal["us", "counterparty", "mutual", "na"]
    span: Span
    fields: dict                  # type-specific payload (cap_amount, notice_days, ...)
    confidence: float
    superseded_by: str | None = None

class TemporalRule(BaseModel):
    id: str
    kind: Literal["renewal","notice","expiry","payment","report","cure"]
    anchor: Literal["effective_date","term_end","invoice_date","breach_date"]
    offset_days: int              # negative = before anchor
    recurrence: str | None        # ISO 8601 duration, e.g. "P12M"
    condition: str | None         # quoted, never paraphrased
    consequence: str
    span: Span

class Obligation(BaseModel):      # materialized by temporal.py, never by the LLM
    rule_id: str
    due_date: date
    owed_by: Literal["us", "counterparty"]
    derivation: list[str]         # human-readable chain, shown in UI

class Finding(BaseModel):         # unified output of every analysis module
    id: str
    kind: Literal["risky_clause","missing_clause","adversarial_pattern",
                  "backtoback_gap","injection","asymmetry"]
    severity: Literal["critical","high","medium","low","info"]
    title: str
    explanation: str
    evidence: list[Span]          # empty ONLY when kind == "missing_clause"
    contract_ids: list[str]
```

Unifying every analysis module behind one `Finding` type means the UI, the API, and the alerting path are each written once.

## 6. API surface

The eventual real UI is just a client of this, so it gets designed properly even though the test UI is throwaway.

| Method | Path | Returns |
|---|---|---|
| POST | `/documents` | doc_id, extracted text, firewall report |
| POST | `/contracts` | group doc_ids into a family, resolve supersession |
| GET | `/contracts/{id}/clauses` | effective clauses with lineage |
| GET | `/contracts/{id}/obligations` | materialized dates + derivation chains |
| GET | `/contracts/{id}/risk` | axis scores + traces |
| GET | `/contracts/{id}/findings` | all findings, severity-ranked |
| POST | `/contracts/{id}/termination-cost` | `{exit_date}` → itemized exit cost |
| GET | `/portfolio/deadlines` | cross-contract, sorted by days remaining |
| GET | `/portfolio/gaps` | back-to-back flow-down mismatches |
| GET | `/portfolio/deadlines.ics` | calendar subscription |

## 7. Build sequence

Three people: **P** (pipeline), **R** (reasoning), **X** (platform/security/eval). Hours are wall-clock from T+0.

### Block 0 — T+0 → T+2 · ALL HANDS · Schema freeze
Write `schemas.py` together, plus a fixture file of 3 hand-written `ClauseClaim` sets representing a realistic MSA. Nobody writes another line until this is agreed.

*Why all hands:* R and X spend the next 12 hours writing code against these types without needing extraction to work. Getting the schema wrong at T+2 costs the whole team at T+14.

### Block 1 — T+2 → T+9 · Parallel
- **P:** `ingest.py`. PDF → text with **character offsets that survive** page joins, headers/footers, ligatures, and hyphenation. Everything depends on offsets being correct; this is the highest-risk hour of the project. OCR fallback with per-page confidence.
- **R:** `temporal.py` + `risk.py` against fixtures. Pure functions, full unit tests, zero network.
- **X:** `db.py`, `main.py` skeleton with stubbed routes returning fixtures, `web/index.html` rendering those fixtures.

**T+9 gate:** the UI displays fixture data end-to-end through real routes. The seam is proven before the hard part lands.

### Block 2 — T+9 → T+16
- **P:** `extract.py` + `verify.py`. Per-clause-type prompts, structured output, exact-substring verification, drop-on-fail. Log `grounding_rate`.
- **R:** `findings/silence.py` (playbooks for MSA/NDA/SOW/DPA) and `findings/asymmetry.py`.
- **X:** `firewall.py` — invisible-text detection (font < 4pt, text-colour ≈ background, off-canvas, PDF metadata payloads) + injection classifier + spotlighting delimiters. Author the poisoned sample contract.

**T+16 gate — the real one:** a genuine PDF goes in, grounded clauses come out, UI shows them. If this slips past T+20, cut Block 4 entirely.

### Block 3 — T+16 → T+22
- **P:** accuracy pass. Prompt iteration against CUAD, per-type failure triage.
- **R:** `findings/adversarial.py` + `findings/termination.py`.
- **X:** `eval/run_eval.py` → the P/R/F1 table. Alerts: ICS export + digest renderer.

### Block 4 — T+22 → T+28 · The differentiators
- **P + R:** `family.py` (document graph, supersession) then `findings/backtoback.py`. Back-to-back depends on family only for effective values; if family slips, run back-to-back on unresolved clauses and say so.
- **X:** portfolio views, `/portfolio/deadlines` + `/portfolio/gaps`.

### Block 5 — T+28 → T+33 · Freeze
Code freeze at T+30. Seed the demo dataset: 6 contracts forming one deliberate back-to-back gap, one amendment chain, one poisoned document, one contract missing its liability cap. Rehearse the demo three times, end to end, on the machine and network you'll present on.

### Block 6 — T+33 → T+36
Slides, README, buffer. Do not write code in this block.

## 8. Cut lines

In order of what dies first:

1. Grounded Q&A tab *(never started unless everything else is done)*
2. `family.py` supersession → degrade to flat extraction with a visible "amendments not resolved" banner
3. `findings/termination.py`
4. OCR fallback → require text-layer PDFs, note it as known
5. `findings/adversarial.py`

**Never cut:** grounding verification, silence detection, the eval table, the firewall. Those four are the entire differentiation.

## 9. Testing

- **Unit, no network:** `verify`, `temporal`, `risk`, `family`, and all of `findings/` run against fixture `ClauseClaim` sets. This is the bulk of the suite and it must stay fast.
- **Property test:** for every extraction, `span.quote == doc_text[span.char_start:span.char_end]`. This single assertion is invariant 1, mechanized.
- **Golden file:** one full PDF → committed expected JSON. Detects prompt-tuning regressions.
- **Adversarial:** the poisoned contract must produce an `injection` finding and must not alter the risk score.
- **Integration:** one real PDF through `/analyze`. Exactly one network-touching test.

## 10. Eval plan

Corpus: CUAD (510 contracts, 41 expert-labelled clause types, CC-BY). Hold out a slice never used for prompt tuning.

Report on the slide:

| Metric | Target |
|---|---|
| Per-clause-type precision / recall / F1 | table, all types shown including the bad ones |
| Grounding rate | > 95% |
| **Hallucination rate** | **0, by construction — the verifier makes it unrepresentable** |
| Median latency per contract | reported honestly |

Show the weak clause types too. A table with three bad rows is far more credible than one with none.

## 11. Security posture

| Control | Implementation |
|---|---|
| Indirect prompt injection | Document text passed as delimited untrusted data with spotlighting; never concatenated into instructions. Pre-inference classifier. |
| Document tampering | Invisible-text detection reported as a **tampering indicator on the counterparty** — an attack reframed as a product feature. |
| Confidentiality at rest | Per-tenant key, encrypted blobs, row-level isolation in the schema from day one. |
| Minimisation | Pricing/PII redaction with a reversible token map before any external model call. |
| Provider retention | Zero-retention configuration, stated explicitly. |
| Auditability | Append-only log of every extraction and every view; signed export bundles. |

In 36 hours the firewall and the audit log are real; the rest is schema-level and honestly labelled as such.

## 12. Demo script — five beats, one per judging criterion

1. **Money** *(business viability)* — a real auto-renewal. The contract contains no dates; show the derived notice deadline and the derivation chain that produced it. "You are paying for something right now that you meant to cancel."
2. **Absence** *(the surprise beat)* — "and here is the clause that isn't there at all." Nobody in the room expects a tool to find a missing clause.
3. **Chain** *(technical depth — the moat)* — the back-to-back gap across three contracts. "You promised three customers 99.99%. Your provider gives you 99.9%. You are underwriting the difference."
4. **Attack** *(security)* — upload the poisoned contract. Naive path returns "Risk: LOW ✅". Ours quarantines it: "Adversarial content detected — 3 hidden instructions in this vendor's document."
5. **Proof** *(AI rigor)* — the CUAD table and the verifier. "Everything we showed you is quoted from your contract. Every number was computed, not generated. Here is the accuracy."

Under a minute per beat.

## 13. Risks

| Risk | Mitigation |
|---|---|
| **Character offsets drift in `ingest.py`** — silently corrupts every downstream quote | Highest-risk item in the project. Property test from hour one. P works on nothing else until it holds. |
| T+16 gate slips | Cut Block 4 immediately, don't negotiate. Blocks 1–3 alone are a complete demo. |
| Live API calls fail on stage | Pre-computed results cached for every demo contract. Demo runs from cache by default with a `--live` flag. |
| Schema churn after T+2 | The freeze is the mitigation. Additive changes only after Block 1. |
| CUAD eval takes longer than budgeted | Score 3 clause types thoroughly rather than 41 shallowly. A narrow honest table beats a broad fabricated one. |

---

## Appendix — feature inventory

| Feature | Module | Block | Effort |
|---|---|---|---|
| Grounded extraction + verifier | `extract`, `verify` | 2 | 6h |
| Temporal obligation compiler | `temporal` | 1 | 3h |
| Party-aware risk rubric | `risk` | 1 | 3h |
| **Silence detection** | `findings/silence` | 2 | 2h |
| **Power asymmetry index** | `findings/asymmetry` | 2 | 1.5h |
| Prompt-injection firewall | `firewall` | 2 | 3h |
| Adversarial clause library | `findings/adversarial` | 3 | 2h |
| Termination cost calculator | `findings/termination` | 3 | 2.5h |
| Eval harness | `eval/` | 3 | 2h |
| Alerts (ICS + digest) | `main` | 3 | 1.5h |
| Contract family + supersession | `family` | 4 | 4h |
| **Back-to-back gap analysis** | `findings/backtoback` | 4 | 4h |
| Portfolio views | `main`, `web` | 4 | 2h |

Bold rows are the three highest impressiveness-per-hour items on the list.
