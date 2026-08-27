"""One real Groq call, end to end. Run this first after setting GROQ_API_KEY.

    export GROQ_API_KEY=gsk_...
    .venv/bin/python eval/smoke_groq.py            # tiny contract, fast
    .venv/bin/python eval/smoke_groq.py --full     # the real Northwind MSA
"""

from __future__ import annotations

import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from api.extract import call_model, ground_clauses, ground_rules
from api.ingest import ingest_text
from api.llm import MODEL, ExtractionUnavailable
from api.verify import verify_claims

TINY = """MASTER SERVICES AGREEMENT

1. TERM. This Agreement continues for twelve (12) months and renews
automatically for successive twelve (12) month periods unless either party
gives written notice of non-renewal at least sixty (60) days prior to the end
of the then-current Term.

2. LIABILITY. Each party's total aggregate liability shall not exceed fifty
thousand dollars ($50,000).

3. GOVERNING LAW. This Agreement is governed by the laws of Delaware.
"""


def main() -> int:
    full = "--full" in sys.argv
    if full:
        path = pathlib.Path(__file__).resolve().parents[1] / "contracts" / "msa_northwind.txt"
        doc = ingest_text(path.read_text(), "msa_northwind.txt")
    else:
        doc = ingest_text(TINY, "tiny_msa.txt")

    print(f"model    : {MODEL}")
    print(f"document : {doc.filename} ({len(doc.text):,} chars)\n")

    from api.chunking import chunk_document

    chunks = chunk_document(doc)
    print(f"chunks   : {len(chunks)}\n")

    started = time.time()
    try:
        raw = call_model(doc, "Contoso Systems Ltd.", use_cache=False, verbose=True)
    except ExtractionUnavailable as exc:
        print(f"FAILED: {exc}")
        return 1
    elapsed = time.time() - started

    claims, cstats = ground_clauses(raw, doc, "smoke")
    rules, rstats = ground_rules(raw, doc, "smoke")
    stats = cstats.merge(rstats)
    verified, report = verify_claims(claims, {doc.id: doc})

    print(f"latency  : {elapsed:.1f}s")
    print(f"returned : {len(raw.clause_list)} clauses, {len(raw.rule_list)} rules")
    print(f"grounding: {stats.summary()}")
    print(f"verified : {report.kept} kept, {report.dropped} dropped by the verifier")
    if stats.dropped_reasons:
        print("\ndiscarded (model quoted text not in the document):")
        for reason in stats.dropped_reasons[:6]:
            print(f"  - {reason}")

    print(f"\nclauses ({len(verified)}):")
    for c in verified[:14]:
        marker = " ~" if c.fields.get("grounding") == "realigned" else "  "
        print(f" {marker} {c.clause_type.value:32} {str(c.fields)[:44]}")
        print(f"      \"{c.span.quote[:88]}\"")

    print(f"\ntemporal rules ({len(rules)}):")
    for r in rules:
        print(f"    {r.kind:8} anchor={r.anchor:14} offset={r.offset_days:+5}d "
              f"recur={r.recurrence or '-'}")

    if stats.rate < 0.8:
        print(f"\nWARNING: grounding rate {stats.rate:.0%} is low. The model is "
              f"paraphrasing quotes. Consider a stronger model via GROQ_MODEL.")
        return 1
    print(f"\nOK - {MODEL} produces grounded extractions.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
