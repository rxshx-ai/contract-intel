"""Extract the whole demo corpus with the live model and cache the results.

Free tier is 8,000 TPM, so this throttles and takes several minutes. Run once;
everything afterwards reads the cache.
"""

from __future__ import annotations

import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from api.demo import OUR_PARTY, PORTFOLIO
from api.extract import call_model, ground_clauses, ground_rules
from api.ingest import ingest_text
from api.llm import MODEL, ExtractionUnavailable
from api.schemas import OurRole

ROOT = pathlib.Path(__file__).resolve().parents[1]
CONTRACTS = ROOT / "contracts"


def main() -> int:
    force = "--force" in sys.argv
    print(f"model: {MODEL}\n")
    started = time.time()
    failures = 0

    for spec in PORTFOLIO:
        role = spec["our_role"]
        for filename in spec["files"]:
            doc = ingest_text((CONTRACTS / filename).read_text(), filename)
            print(f"{filename}")
            try:
                raw = call_model(doc, OUR_PARTY, use_cache=not force, verbose=True)
            except ExtractionUnavailable as exc:
                print(f"   FAILED: {exc}\n")
                failures += 1
                continue
            claims, cs = ground_clauses(raw, doc, spec["id"], OUR_PARTY, role)
            rules, rs = ground_rules(raw, doc, spec["id"], OUR_PARTY, role)
            stats = cs.merge(rs)
            print(f"   {len(claims)} clauses, {len(rules)} rules | {stats.summary()}")
            for reason in stats.dropped_reasons[:3]:
                print(f"     drop: {reason}")
            print()

    print(f"done in {time.time() - started:.0f}s, {failures} failure(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
