"""Install the shipped demo extractions into the extraction cache.

    python eval/seed_cache.py

`demo_cache/` holds real openai/gpt-oss-120b output for the six demo documents,
committed so a fresh clone can run the whole product with no API key. The live
cache itself stays untracked: it fills up with extractions of whatever you
upload, and those are your documents, not the repository's.

Cache keys include the model name, so after changing GROQ_MODEL these entries
no longer match and the corpus must be re-extracted (eval/extract_all.py).
"""

from __future__ import annotations

import pathlib
import shutil
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from api.demo import OUR_PARTY, PORTFOLIO
from api.extract import MODEL, _cache_path
from api.ingest import ingest_text

ROOT = pathlib.Path(__file__).resolve().parents[1]
SHIPPED = ROOT / "demo_cache"


def main() -> int:
    if not SHIPPED.exists():
        print(f"missing {SHIPPED}")
        return 1

    seeded, missing = 0, []
    for spec in PORTFOLIO:
        for name in spec["files"]:
            source = SHIPPED / f"{name.replace('.txt', '')}.json"
            if not source.exists():
                missing.append(name)
                continue
            doc = ingest_text((ROOT / "contracts" / name).read_text(), name)
            target = _cache_path(doc, OUR_PARTY)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            seeded += 1

    print(f"seeded {seeded} extraction(s) for model {MODEL}")
    if missing:
        print(f"  no shipped extraction for: {', '.join(missing)}")
        print("  run eval/extract_all.py with GROQ_API_KEY set")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
