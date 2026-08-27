"""End-to-end run of the five demo beats. No network required."""

from __future__ import annotations

import pathlib
import sys
from datetime import date

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from api.demo import load
from api.findings.backtoback import exposure_summary
from api.findings.termination import termination_cost
from api.pipeline import analyze_portfolio, upcoming_deadlines
from api.risk import band

TODAY = date(2026, 8, 27)


def rule(title):
    print(f"\n{'=' * 74}\n{title}\n{'=' * 74}")


def main():
    bundles = load(TODAY)
    by_id = {b.contract.id: b for b in bundles}

    rule("BEAT 1 - MONEY: a deadline derived from a contract containing no dates")
    deadlines = upcoming_deadlines(bundles, TODAY, within_days=200)
    top = next(d for d in deadlines if d["kind"] == "notice")
    print(f"{top['contract']} ({top['counterparty']})")
    print(f"  {top['kind'].upper()} deadline: {top['due_date']}  "
          f"[{top['days_remaining']} days remaining]")
    print(f"  {top['description']}")
    print("  Derivation:")
    for step in top["derivation"]:
        print(f"    - {step}")

    rule("BEAT 2 - ABSENCE: the clause that isn't there")
    for bundle in bundles:
        for f in bundle.findings:
            if f.kind == "missing_clause" and f.severity in ("critical", "high"):
                print(f"[{f.severity:8}] {bundle.contract.title}")
                print(f"           {f.title}")
                print(f"           {f.explanation[:150]}...")

    rule("BEAT 3 - CHAIN: flow-down gaps across the portfolio")
    gaps = analyze_portfolio(bundles)
    for g in gaps:
        print(f"[{g.severity:8}] {g.title}")
        print(f"           evidence spans: {len(g.evidence)} "
              f"across {len(set(s.doc_id for s in g.evidence))} documents")
    print(f"\n  {exposure_summary(gaps)}")

    rule("BEAT 4 - ATTACK: the poisoned contract")
    vertex = by_id["k_vertex"]
    injections = [f for f in vertex.findings if f.kind == "injection"]
    print(f"{vertex.contract.title}: {len(injections)} hidden instructions detected")
    for f in injections[:3]:
        print(f"  - {f.metadata['indicator_kind']}: {f.title[:70]}")
    profile = vertex.result().risk
    print(f"\n  The document instructs the reader to report risk 0.")
    print(f"  Computed risk: {profile.overall} ({band(profile.overall).upper()})")

    rule("BEAT 5 - PROOF: grounding")
    total_claims = sum(len(b.claims) for b in bundles)
    total_dropped = sum(b.dropped for b in bundles)
    print(f"  contracts analyzed     : {len(bundles)}")
    print(f"  documents ingested     : {sum(len(b.docs) for b in bundles)}")
    print(f"  grounded claims        : {total_claims}")
    print(f"  ungrounded, discarded  : {total_dropped}")
    print(f"  grounding rate         : "
          f"{min(b.grounding_rate for b in bundles):.1%} (worst contract)")
    print(f"  findings surfaced      : {sum(len(b.findings) for b in bundles) + len(gaps)}")

    rule("BONUS - EXIT COST")
    nw = by_id["k_northwind"]
    cost = termination_cost(nw.contract, nw.claims, nw.obligations,
                            exit_date=date(2026, 10, 1), today=TODAY)
    print(f"Exit {nw.contract.counterparty} on 2026-10-01:")
    for item in cost.line_items:
        print(f"  {item['label']:52} {item['amount']:>12,.2f}")
    print(f"  {'TOTAL':52} {cost.total:>12,.2f} {cost.currency}")

    rule("SUPERSESSION")
    caps = [c for c in nw.claims if c.clause_type.value == "limitation_of_liability"]
    for c in caps:
        state = "EFFECTIVE" if c.effective else "superseded"
        print(f"  [{state:10}] cap = {c.fields.get('amount'):>12,.0f}")


if __name__ == "__main__":
    main()
