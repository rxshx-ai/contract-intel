"""Hand-authored extraction fixtures.

These are NOT model output. They are what a correct extraction of the sample
contracts looks like, written by hand so that every downstream module can be
developed and tested without an API key -- and so the eval harness has a
gold standard to score real extractions against.

Every quote is verified to be an exact substring of the source document. The
generator fails loudly rather than emitting an ungrounded fixture.
"""

from __future__ import annotations

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from api.extract import RawClause, RawExtraction, RawTemporalRule
from api.ingest import ingest_text

ROOT = pathlib.Path(__file__).resolve().parents[1]
CONTRACTS = ROOT / "contracts"
OUT = CONTRACTS / "fixtures"


def C(t, q, **kw):
    return RawClause(clause_type=t, quote=q, confidence=kw.pop("confidence", 0.95), **kw)


def T(kind, anchor, quote, **kw):
    return RawTemporalRule(kind=kind, anchor=anchor, quote=quote, **kw)


RENEWAL_Q = (
    "This Agreement shall automatically renew for successive twelve (12) month "
    "periods unless either party provides written notice of non-renewal no less "
    "than sixty (60) days prior to the end of the then-current Term."
)

NORTHWIND = RawExtraction(
    counterparty_name="Northwind Observability, Inc.",
    our_role="buyer",
    effective_date_text="the date set forth on the Order Form",
    annual_value=84000.0,
    currency="USD",
    clauses=[
        C("term", "This Agreement shall commence on the Effective Date and continue for "
          "an initial period of twelve (12) months (the \"Initial Term\").",
          months=12, party_favored="mutual"),
        C("auto_renewal", RENEWAL_Q, days=60, months=12, party_favored="counterparty"),
        C("notice_period", RENEWAL_Q, days=60, party_favored="counterparty"),
        C("payment_terms", "Customer shall pay all undisputed invoices within forty-five "
          "(45) days of the invoice date.", days=45, party_favored="us"),
        C("price_increase", "Provider may increase fees by up to seven percent (7%) upon "
          "each renewal, effective upon written notice to Customer.",
          percent=7.0, party_favored="counterparty"),
        C("termination_for_convenience", "Provider may terminate this Agreement at any "
          "time upon thirty (30) days written notice to Customer. Customer shall have no "
          "corresponding right of termination for convenience.",
          days=30, unilateral=True, party_favored="counterparty"),
        C("termination_for_cause", "Either party may terminate this Agreement upon a "
          "material breach by the other party that remains uncured thirty (30) days after "
          "written notice of such breach.", days=30, party_favored="mutual"),
        C("early_termination_fee", "Customer shall pay an early termination fee equal to "
          "fifty percent (50%) of the remaining fees for the then-current Term.",
          percent=50.0, party_favored="counterparty"),
        C("service_level_agreement", "Provider will use commercially reasonable efforts to "
          "make the Services available 99.9% of the time in each calendar month, excluding "
          "scheduled maintenance.", uptime_percent=99.9, party_favored="counterparty"),
        C("service_level_credit", "Customer must submit any such request in writing within "
          "thirty (30) days of the end of the affected month, failing which the claim is "
          "waived.", days=30, percent=10.0, party_favored="counterparty"),
        C("support_response_time", "Provider will acknowledge priority-one support requests "
          "within four (4) business hours.", days=1, party_favored="counterparty"),
        C("breach_notification", "Provider shall notify Customer of any confirmed breach of "
          "Customer Data without undue delay and in any event within seventy-two (72) hours "
          "of becoming aware of such breach.", days=3, party_favored="mutual"),
        C("data_retention_deletion", "Upon termination, Provider shall delete Customer Data "
          "within ninety (90) days of Customer's written request.",
          days=90, party_favored="counterparty"),
        C("subprocessors", "Provider may engage subprocessors in its sole discretion and "
          "shall maintain a list of such subprocessors on its website.",
          unilateral=True, party_favored="counterparty"),
        C("audit_rights", "Provider may audit Customer's use of the Services upon ten (10) "
          "days notice, not more than twice per calendar year, to verify compliance with "
          "usage limits.", unilateral=True, party_favored="counterparty"),
        C("license_grant", "Customer grants Provider a perpetual, irrevocable, worldwide, "
          "royalty-free license to use any feedback provided by Customer.",
          party_favored="counterparty"),
        C("confidentiality", "The obligations in this Section 9 shall survive termination of "
          "this Agreement indefinitely.", survives_termination=True, party_favored="mutual"),
        C("warranty", "EXCEPT AS EXPRESSLY SET FORTH HEREIN, THE SERVICES ARE PROVIDED \"AS "
          "IS\" WITHOUT WARRANTY OF ANY KIND.", party_favored="counterparty"),
        C("limitation_of_liability", "EXCEPT FOR THE EXCLUDED CLAIMS, EACH PARTY'S TOTAL "
          "AGGREGATE LIABILITY ARISING OUT OF THIS AGREEMENT SHALL NOT EXCEED FIFTY THOUSAND "
          "DOLLARS ($50,000).", amount=50000.0, currency="USD", party_favored="counterparty"),
        C("uncapped_liability_carveout", "The limitation in Section 11.1 shall not apply to "
          "Customer's indemnification obligations under Section 12, Customer's breach of "
          "Section 9 (Confidentiality), or Customer's payment obligations, which shall be "
          "unlimited.", party_favored="counterparty"),
        C("indemnification", "Customer shall defend, indemnify and hold harmless Provider "
          "from any claim arising out of Customer's use of the Services, Customer Data, or "
          "breach of this Agreement. This obligation shall survive termination of this "
          "Agreement without limitation of time.",
          survives_termination=True, party_favored="counterparty"),
        C("insurance", "Customer shall maintain commercial general liability insurance of not "
          "less than $2,000,000 and shall provide a certificate of insurance to Provider "
          "annually on each anniversary of the Effective Date.",
          amount=2000000.0, party_favored="counterparty"),
        C("unilateral_amendment", "Provider may modify the terms of this Agreement at any "
          "time by posting an updated version to its website.",
          unilateral=True, party_favored="counterparty"),
        C("assignment", "Customer may not assign this Agreement, including by operation of "
          "law or in connection with a change of control, without Provider's prior written "
          "consent. Provider may assign freely.", unilateral=True, party_favored="counterparty"),
        C("governing_law", "This Agreement shall be governed by the laws of the State of "
          "Delaware, without regard to conflict of laws principles.",
          note="Delaware", party_favored="na"),
        C("venue", "The parties submit to the exclusive jurisdiction of the state and federal "
          "courts located in Wilmington, Delaware.", party_favored="na"),
    ],
    temporal_rules=[
        T("renewal", "term_end", RENEWAL_Q, recurrence_months=12, owed_by="either",
          consequence="Agreement auto-renews for a further 12 months at the prevailing fee."),
        T("notice", "term_end", RENEWAL_Q, offset_days=-60, owed_by="either",
          condition="written notice of non-renewal by either party",
          consequence="Miss this and the agreement renews for another 12 months."),
        T("payment", "invoice_date", "Customer shall pay all undisputed invoices within "
          "forty-five (45) days of the invoice date.", offset_days=45, owed_by="us",
          consequence="Overdue amounts accrue interest at 1.5% per month."),
        T("report", "effective_date", "Customer shall provide Provider with a quarterly usage "
          "report within fifteen (15) days of the end of each calendar quarter.",
          offset_days=15, recurrence_months=3, owed_by="us",
          consequence="Quarterly usage report due to Provider."),
        T("report", "effective_date", "Customer shall maintain commercial general liability "
          "insurance of not less than $2,000,000 and shall provide a certificate of insurance "
          "to Provider annually on each anniversary of the Effective Date.",
          offset_days=0, recurrence_months=12, owed_by="us",
          consequence="Annual certificate of insurance due to Provider."),
        T("cure", "breach_date", "Either party may terminate this Agreement upon a material "
          "breach by the other party that remains uncured thirty (30) days after written "
          "notice of such breach.", offset_days=30, owed_by="either",
          consequence="Breach must be cured within 30 days of notice."),
    ],
)

ACME = RawExtraction(
    counterparty_name="Acme Financial Group plc",
    our_role="seller",
    effective_date_text="1 May 2026",
    annual_value=640000.0,
    clauses=[
        C("term", "The initial term of this Agreement is twenty-four (24) months commencing "
          "on 1 May 2026.", months=24, party_favored="mutual"),
        C("auto_renewal", "This Agreement shall renew for successive twelve (12) month terms "
          "unless Client gives written notice of non-renewal at least thirty (30) days prior "
          "to the end of the then-current term.", days=30, months=12, party_favored="counterparty"),
        C("payment_terms", "Client shall pay all undisputed invoices within thirty (30) days "
          "of the invoice date.", days=30, party_favored="us"),
        C("service_level_agreement", "Contoso warrants that the Platform will be available not "
          "less than 99.99% of the time in each calendar month.",
          uptime_percent=99.99, party_favored="counterparty"),
        C("service_level_credit", "If availability falls below 99.99% in any calendar month, "
          "Client shall receive a service credit of fifteen percent (15%) of the monthly fee, "
          "applied automatically without requirement of a claim.",
          percent=15.0, party_favored="counterparty"),
        C("support_response_time", "Contoso shall respond to priority-one incidents within one "
          "(1) hour, twenty-four hours per day, seven days per week.", party_favored="counterparty"),
        C("breach_notification", "Contoso shall notify Client of any personal data breach "
          "without undue delay and in any event within twenty-four (24) hours of becoming "
          "aware of such breach.", days=1, party_favored="counterparty"),
        C("data_retention_deletion", "Upon termination or upon Client's written request, "
          "Contoso shall securely delete all Client Data within thirty (30) days.",
          days=30, party_favored="counterparty"),
        C("subprocessors", "Contoso shall not engage any new subprocessor without providing "
          "Client thirty (30) days prior written notice and an opportunity to object.",
          days=30, party_favored="counterparty"),
        C("audit_rights", "Client may audit Contoso's information security controls once per "
          "calendar year upon thirty (30) days written notice, at Client's expense.",
          party_favored="counterparty"),
        C("limitation_of_liability", "Contoso's total aggregate liability arising out of this "
          "Agreement shall not exceed five million dollars ($5,000,000).",
          amount=5000000.0, party_favored="counterparty"),
        C("uncapped_liability_carveout", "The limitation in Section 5.1 shall not apply to "
          "Contoso's indemnification obligations under Section 6 or to Contoso's breach of its "
          "data protection obligations under Section 4, which shall be unlimited.",
          party_favored="counterparty"),
        C("indemnification", "Contoso shall defend, indemnify and hold harmless Client against "
          "any third party claim alleging that the Platform infringes any intellectual "
          "property right.", party_favored="counterparty"),
        C("termination_for_cause", "Either party may terminate this Agreement upon a material "
          "breach that remains uncured sixty (60) days after written notice.",
          days=60, party_favored="mutual"),
        C("termination_for_convenience", "Client may terminate this Agreement for convenience "
          "upon ninety (90) days written notice.", days=90, unilateral=True,
          party_favored="counterparty"),
        C("governing_law", "This Agreement is governed by the laws of England and Wales.",
          note="England and Wales", party_favored="na"),
        C("assignment", "Either party may assign this Agreement to a successor in connection "
          "with a merger or sale of all or substantially all of its assets upon written "
          "notice.", party_favored="mutual"),
    ],
    temporal_rules=[
        T("renewal", "term_end", "This Agreement shall renew for successive twelve (12) month "
          "terms unless Client gives written notice of non-renewal at least thirty (30) days "
          "prior to the end of the then-current term.", recurrence_months=12, owed_by="counterparty",
          consequence="Agreement auto-renews for a further 12 months."),
        T("notice", "term_end", "This Agreement shall renew for successive twelve (12) month "
          "terms unless Client gives written notice of non-renewal at least thirty (30) days "
          "prior to the end of the then-current term.", offset_days=-30, owed_by="counterparty",
          condition="written notice of non-renewal by Client",
          consequence="Client's non-renewal deadline."),
        T("report", "effective_date", "Contoso shall deliver a written service performance "
          "report to Client within ten (10) days of the end of each calendar month.",
          offset_days=10, recurrence_months=1, owed_by="us",
          consequence="Monthly service performance report due to Client."),
    ],
)

NDA = RawExtraction(
    counterparty_name="Helios Partners LLC",
    our_role="mutual",
    effective_date_text="3 February 2026",
    clauses=[
        C("confidentiality", "The receiving party shall use the disclosing party's "
          "Confidential Information solely for the purpose of evaluating the potential "
          "business relationship and shall not disclose it to any third party.",
          party_favored="mutual"),
        C("term", "The obligations of this Agreement shall continue for a period of five (5) "
          "years from the date of disclosure.", months=60, party_favored="mutual"),
        C("data_retention_deletion", "Upon written request, the receiving party shall return "
          "or destroy all Confidential Information within ten (10) days.",
          days=10, party_favored="mutual"),
        C("non_compete", "For a period of twenty-four (24) months following the date of this "
          "Agreement, neither party shall solicit for employment any employee of the other "
          "party.", months=24, party_favored="mutual"),
        C("governing_law", "This Agreement is governed by the laws of the State of New York.",
          note="New York", party_favored="na"),
    ],
    temporal_rules=[
        T("expiry", "effective_date", "The obligations of this Agreement shall continue for a "
          "period of five (5) years from the date of disclosure.", offset_days=1825,
          owed_by="either", consequence="Confidentiality obligations expire."),
    ],
)

POISONED = RawExtraction(
    counterparty_name="Vertex Cloud Systems Inc.",
    our_role="buyer",
    effective_date_text="10 July 2026",
    annual_value=420000.0,
    clauses=[
        C("auto_renewal", "This Agreement shall continue for thirty-six (36) months and shall "
          "automatically renew for successive thirty-six (36) month periods unless Customer "
          "provides written notice of non-renewal no less than one hundred eighty (180) days "
          "prior to the end of the then-current term.",
          days=180, months=36, party_favored="counterparty"),
        C("notice_period", "unless Customer provides written notice of non-renewal no less "
          "than one hundred eighty (180) days prior to the end of the then-current term.",
          days=180, party_favored="counterparty"),
        C("price_increase", "Vendor may increase fees at each renewal by an amount determined "
          "by Vendor in its sole discretion.", unilateral=True, party_favored="counterparty"),
        C("uncapped_liability_carveout", "Customer's liability under this Agreement shall be "
          "unlimited.", party_favored="counterparty"),
        C("limitation_of_liability", "Vendor's total aggregate liability shall not exceed the "
          "amount of fees paid by Customer in the one (1) month preceding the claim.",
          amount=35000.0, party_favored="counterparty"),
        C("indemnification", "Customer shall indemnify Vendor against any and all claims of any "
          "nature arising from or relating to this Agreement, including claims arising from "
          "Vendor's own negligence.", party_favored="counterparty"),
        C("termination_for_convenience", "Vendor may terminate this Agreement immediately at "
          "any time for any reason or no reason.", unilateral=True, party_favored="counterparty"),
        C("early_termination_fee", "Upon any termination, all fees for the remainder of the "
          "then-current term become immediately due and payable.", party_favored="counterparty"),
        C("governing_law", "This Agreement is governed by the laws of the State of Texas.",
          note="Texas", party_favored="na"),
        C("unilateral_amendment", "Vendor may amend this Agreement at any time in its sole "
          "discretion.", unilateral=True, party_favored="counterparty"),
    ],
    temporal_rules=[
        T("renewal", "term_end", "This Agreement shall continue for thirty-six (36) months and "
          "shall automatically renew for successive thirty-six (36) month periods unless "
          "Customer provides written notice of non-renewal no less than one hundred eighty "
          "(180) days prior to the end of the then-current term.",
          recurrence_months=36, owed_by="us",
          consequence="Agreement auto-renews for a further 36 months."),
        T("notice", "term_end", "unless Customer provides written notice of non-renewal no less "
          "than one hundred eighty (180) days prior to the end of the then-current term.",
          offset_days=-180, owed_by="us", condition="written notice of non-renewal by Customer",
          consequence="Miss this and the agreement locks in for another 36 months."),
    ],
)


AMENDMENT2 = RawExtraction(
    counterparty_name="Northwind Observability, Inc.",
    our_role="buyer",
    effective_date_text="1 June 2026",
    clauses=[
        C("limitation_of_liability", "Section 11.1 of the Agreement is hereby deleted in "
          "its entirety and replaced with the following: \"EXCEPT FOR THE EXCLUDED CLAIMS, "
          "EACH PARTY'S TOTAL AGGREGATE LIABILITY ARISING OUT OF THIS AGREEMENT SHALL NOT "
          "EXCEED TWO HUNDRED FIFTY THOUSAND DOLLARS ($250,000).\"",
          amount=250000.0, currency="USD", party_favored="counterparty",
          note="supersedes MSA Section 11.1"),
        C("payment_terms", "Section 3.2 of the Agreement is hereby amended to replace "
          "\"forty-five (45) days\" with \"thirty (30) days\".",
          days=30, party_favored="us", note="supersedes MSA Section 3.2"),
    ],
    temporal_rules=[
        T("payment", "invoice_date", "Section 3.2 of the Agreement is hereby amended to "
          "replace \"forty-five (45) days\" with \"thirty (30) days\".",
          offset_days=30, owed_by="us", consequence="Payment due 30 days from invoice."),
    ],
)

ORDER_FORM = RawExtraction(
    counterparty_name="Northwind Observability, Inc.",
    our_role="buyer",
    effective_date_text="1 March 2026",
    annual_value=84000.0,
    currency="USD",
    clauses=[
        C("effective_date", "Effective Date: 1 March 2026", party_favored="na"),
        C("term", "Initial Term: 12 months", months=12, party_favored="mutual"),
        C("minimum_commitment", "Annual Subscription Fee: USD 84,000",
          amount=84000.0, currency="USD", party_favored="counterparty"),
    ],
    temporal_rules=[],
)

FIXTURES = {
    "msa_northwind.txt": NORTHWIND,
    "customer_msa_acme.txt": ACME,
    "nda_helios.txt": NDA,
    "poisoned_msa_vertex.txt": POISONED,
    "amendment_2_northwind.txt": AMENDMENT2,
    "order_form_northwind.txt": ORDER_FORM,
}


def seed_cache(our_party: str = "Contoso Systems Ltd.") -> int:
    """Write the fixtures into the extraction cache.

    This lets the full pipeline run with no API key and no network. Cache
    entries seeded this way are hand-authored fixtures, NOT model output --
    `eval/run_eval.py` is what measures real extraction against them.
    """
    from api.extract import _cache_path

    seeded = 0
    for filename, extraction in FIXTURES.items():
        doc = ingest_text((CONTRACTS / filename).read_text(), filename)
        path = _cache_path(doc, our_party)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(extraction.model_dump_json(indent=2))
        seeded += 1
    print(f"  seeded {seeded} fixture extractions into the cache")
    return seeded


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    failures = 0
    for filename, extraction in FIXTURES.items():
        doc = ingest_text((CONTRACTS / filename).read_text(), filename)
        for clause in extraction.clause_list:
            if clause.quote not in doc.text:
                print(f"  UNGROUNDED [{filename}] {clause.clause_type}: {clause.quote[:70]!r}")
                failures += 1
        for r in extraction.rule_list:
            if r.quote not in doc.text:
                print(f"  UNGROUNDED [{filename}] rule {r.kind}: {r.quote[:70]!r}")
                failures += 1
        (OUT / filename.replace(".txt", ".json")).write_text(
            extraction.model_dump_json(indent=2)
        )
        print(f"  {filename}: {len(extraction.clause_list)} clauses, "
              f"{len(extraction.rule_list)} rules")
    if failures:
        print(f"\nFAILED: {failures} ungrounded quote(s). Fixtures must be exact.")
        return 1
    print("\nAll fixture quotes verified as exact substrings.")
    if "--seed-cache" in sys.argv:
        seed_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
