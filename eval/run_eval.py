"""Extraction accuracy against the hand-authored gold standard.

Scores a REAL model extraction against `contracts/fixtures/*.json`. The
fixtures are the gold labels; nothing here reads the cache those fixtures
seeded, because scoring a fixture against itself measures nothing.

Reported metrics:
  - per-clause-type precision / recall / F1
  - grounding rate (share of model claims that located in the source)
  - hallucination rate (0 by construction -- verify.py discards the rest)

Weak clause types are printed, not hidden. A table with three bad rows is more
credible than one with none.

    python eval/run_eval.py            # live extraction (needs ANTHROPIC_API_KEY)
    python eval/run_eval.py --self     # self-check of the harness on fixtures
"""

from __future__ import annotations

import pathlib
import sys
import time
from collections import defaultdict

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from api.extract import RawExtraction, call_model, ground_clauses
from api.llm import MODEL
from api.ingest import ingest_text
from api.schemas import ClauseClaim, Document
from api.verify import verify_claims

ROOT = pathlib.Path(__file__).resolve().parents[1]
CONTRACTS = ROOT / "contracts"
FIXTURES = CONTRACTS / "fixtures"
OUR_PARTY = "Contoso Systems Ltd."

# Held out from prompt tuning.
EVAL_SET = [
    "msa_northwind.txt",
    "customer_msa_acme.txt",
    "nda_helios.txt",
    "poisoned_msa_vertex.txt",
]

OVERLAP_THRESHOLD = 0.3


def _iou(a: ClauseClaim, b: ClauseClaim) -> float:
    lo = max(a.span.char_start, b.span.char_start)
    hi = min(a.span.char_end, b.span.char_end)
    inter = max(0, hi - lo)
    union = ((a.span.char_end - a.span.char_start)
             + (b.span.char_end - b.span.char_start) - inter)
    return inter / union if union else 0.0


def match(pred: list[ClauseClaim], gold: list[ClauseClaim]) -> tuple[int, int, int]:
    """Greedy one-to-one match on (clause_type, span overlap)."""
    unmatched = list(gold)
    tp = 0
    for p in pred:
        best, best_score = None, 0.0
        for g in unmatched:
            if g.clause_type != p.clause_type:
                continue
            score = _iou(p, g)
            if score > best_score:
                best, best_score = g, score
        if best is not None and best_score >= OVERLAP_THRESHOLD:
            unmatched.remove(best)
            tp += 1
    return tp, len(pred) - tp, len(unmatched)   # tp, fp, fn


def prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def evaluate(self_check: bool) -> int:
    per_type: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])
    totals = [0, 0, 0]
    grounded_total = dropped_total = 0
    latencies: list[float] = []
    grounding: list = []

    for filename in EVAL_SET:
        doc: Document = ingest_text((CONTRACTS / filename).read_text(), filename)
        gold_raw = RawExtraction.model_validate_json(
            (FIXTURES / filename.replace(".txt", ".json")).read_text()
        )
        gold, _gs = ground_clauses(gold_raw, doc, "gold")

        started = time.time()
        if self_check:
            pred_raw = gold_raw           # harness self-test: must score 1.00
        else:
            pred_raw = call_model(doc, OUR_PARTY, use_cache=False)
        latencies.append(time.time() - started)

        pred, gstats = ground_clauses(pred_raw, doc, "pred")
        pred, report = verify_claims(pred, {doc.id: doc})
        grounded_total += len(pred)
        dropped_total += gstats.dropped + report.dropped
        grounding.append(gstats)

        types = {c.clause_type for c in pred} | {c.clause_type for c in gold}
        for ctype in types:
            p = [c for c in pred if c.clause_type == ctype]
            g = [c for c in gold if c.clause_type == ctype]
            tp, fp, fn = match(p, g)
            bucket = per_type[ctype.value]
            bucket[0] += tp; bucket[1] += fp; bucket[2] += fn
            totals[0] += tp; totals[1] += fp; totals[2] += fn

        print(f"  {filename:30} {len(pred):>3} predicted / {len(gold):>3} gold"
              f"   {latencies[-1]:.1f}s")

    print(f"\n{'clause type':<34}{'P':>7}{'R':>7}{'F1':>7}{'TP':>5}{'FP':>5}{'FN':>5}")
    print("-" * 70)
    rows = sorted(per_type.items(), key=lambda kv: prf(*kv[1])[2])
    for ctype, (tp, fp, fn) in rows:
        p, r, f1 = prf(tp, fp, fn)
        flag = "  <-- weak" if f1 < 0.6 else ""
        print(f"{ctype:<34}{p:>7.2f}{r:>7.2f}{f1:>7.2f}{tp:>5}{fp:>5}{fn:>5}{flag}")

    p, r, f1 = prf(*totals)
    total_claims = grounded_total + dropped_total
    grounding = grounded_total / total_claims if total_claims else 1.0

    print("-" * 70)
    print(f"{'MICRO-AVERAGE':<34}{p:>7.2f}{r:>7.2f}{f1:>7.2f}"
          f"{totals[0]:>5}{totals[1]:>5}{totals[2]:>5}")
    print(f"\n  documents scored      : {len(EVAL_SET)}")
    print(f"  clause types covered  : {len(per_type)}")
    print(f"  grounding rate        : {grounding:.1%}")
    print(f"  ungrounded, discarded : {dropped_total}")
    if grounding:
        merged = grounding[0]
        for g in grounding[1:]:
            merged = merged.merge(g)
        print(f"  span provenance       : {merged.summary()}")
    print(f"  hallucination rate    : 0.000  (by construction -- see api/verify.py)")
    print(f"  median latency        : {sorted(latencies)[len(latencies)//2]:.1f}s/contract")

    if self_check and f1 < 0.999:
        print("\nSELF-CHECK FAILED: harness must score 1.00 against its own fixtures.")
        return 1
    return 0


if __name__ == "__main__":
    self_check = "--self" in sys.argv
    print("HARNESS SELF-CHECK (gold vs gold)\n" if self_check
          else f"LIVE EXTRACTION EVAL - model: {MODEL}\n")
    raise SystemExit(evaluate(self_check))
