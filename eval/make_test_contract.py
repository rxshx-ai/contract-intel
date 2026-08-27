"""Generate a realistic vendor MSA as a PDF, for testing upload.

Deliberately constructed so that a correct analysis finds:
  - a renewal notice deadline with NO date in the document (temporal compiler)
  - three NEW flow-down gaps against the Acme customer contract
      uptime 99.5% vs 99.99% promised out
      breach notice 120h vs 24h promised out
      data deletion 180d vs 30d promised out
      liability cap 25,000 vs 5,000,000 promised out
  - a liability cap far below annual contract value (risk rubric)
  - five dark patterns: unilateral amendment, 120-day notice window,
    sole-discretion pricing, fee acceleration on exit, one-way termination
  - an uneconomic remedy: a $25k cap with exclusive Singapore venue
  - missing clauses: confidentiality, termination for cause (silence detection)

Nothing about it is fake-friendly: every number is stated the way a real
vendor paper states it, in words with the figure in parentheses.
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer

OUT = pathlib.Path(__file__).resolve().parents[1] / "contracts" / "meridian_msa.pdf"

TITLE = "MASTER SERVICES AGREEMENT"

PREAMBLE = (
    "This Master Services Agreement (this \"Agreement\") is entered into as of "
    "1 February 2026 (the \"Effective Date\") by and between Meridian Cloud "
    "Infrastructure Pte. Ltd., a company incorporated in Singapore (\"Vendor\"), "
    "and Contoso Systems Ltd. (\"Customer\")."
)

SECTIONS: list[tuple[str, list[str]]] = [
    ("1. DEFINITIONS", [
        "1.1 \"Services\" means the managed cloud infrastructure and container "
        "orchestration platform made available by Vendor under this Agreement.",
        "1.2 \"Customer Data\" means all data submitted to the Services by or on "
        "behalf of Customer.",
        "1.3 \"Term\" means the Initial Term together with any Renewal Term.",
    ]),
    ("2. TERM AND RENEWAL", [
        "2.1 Initial Term. This Agreement commences on the Effective Date and "
        "continues for an initial period of twenty-four (24) months (the "
        "\"Initial Term\").",
        "2.2 Automatic Renewal. Upon expiry of the Initial Term, this Agreement "
        "shall automatically renew for successive twenty-four (24) month periods "
        "unless Customer provides written notice of non-renewal no less than one "
        "hundred twenty (120) days prior to the end of the then-current Term.",
    ]),
    ("3. FEES AND PAYMENT", [
        "3.1 Fees. Customer shall pay an annual subscription fee of USD 216,000, "
        "invoiced annually in advance.",
        "3.2 Payment Terms. Customer shall pay all invoices within thirty (30) "
        "days of the invoice date. Fees are non-refundable in all circumstances.",
        "3.3 Fee Adjustment. Vendor may adjust the fees payable at each renewal "
        "by an amount determined by Vendor in its sole discretion, effective upon "
        "written notice to Customer.",
        "3.4 Late Payment. Overdue amounts shall accrue interest at two percent "
        "(2%) per month, compounded monthly.",
    ]),
    ("4. TERMINATION", [
        "4.1 Termination for Convenience. Vendor may terminate this Agreement at "
        "any time upon forty-five (45) days written notice to Customer. Customer "
        "shall have no right to terminate this Agreement for convenience during "
        "the Initial Term or any Renewal Term.",
        "4.2 Effect of Termination. Upon any termination of this Agreement, all "
        "fees for the remainder of the then-current Term shall become immediately "
        "due and payable in full.",
    ]),
    ("5. SERVICE LEVELS", [
        "5.1 Availability. Vendor shall use reasonable efforts to make the "
        "Services available not less than 99.5% of the time in each calendar "
        "month, excluding scheduled maintenance windows.",
        "5.2 Service Credits. Where availability falls below 99.5% in a calendar "
        "month, Customer may claim a service credit of five percent (5%) of the "
        "monthly fee. Customer must submit any such claim in writing within "
        "fifteen (15) days of the end of the affected month, failing which the "
        "claim is irrevocably waived.",
        "5.3 Support. Vendor shall use reasonable efforts to acknowledge "
        "priority-one incidents within twelve (12) business hours.",
    ]),
    ("6. DATA PROTECTION", [
        "6.1 Security. Vendor shall implement such technical and organisational "
        "measures as Vendor considers appropriate in its sole discretion.",
        "6.2 Breach Notification. Vendor shall notify Customer of any confirmed "
        "unauthorised access to Customer Data within five (5) days of Vendor "
        "confirming such access.",
        "6.3 Data Deletion. Following termination, Vendor shall delete Customer "
        "Data within one hundred eighty (180) days of receipt of Customer's "
        "written request.",
        "6.4 Subprocessors. Vendor may appoint, replace or remove subprocessors "
        "in its sole discretion without notice to Customer.",
    ]),
    ("7. AUDIT AND REPORTING", [
        "7.1 Vendor Audit Rights. Vendor may audit Customer's use of the Services "
        "upon five (5) days notice to verify compliance with applicable usage "
        "limits. Customer shall provide all reasonable cooperation at Customer's "
        "sole cost and expense.",
        "7.2 Customer Reporting. Customer shall deliver to Vendor a written "
        "utilisation report within twenty (20) days of the end of each calendar "
        "quarter.",
        "7.3 Insurance. Customer shall maintain commercial general liability "
        "insurance of not less than USD 3,000,000 and shall furnish a certificate "
        "of insurance to Vendor annually on each anniversary of the Effective Date.",
    ]),
    ("8. INTELLECTUAL PROPERTY", [
        "8.1 Ownership. Vendor retains all right, title and interest in and to the "
        "Services. Customer retains all right, title and interest in Customer Data.",
        "8.2 Feedback. Customer grants to Vendor a perpetual, irrevocable, "
        "worldwide, royalty-free and fully sublicensable licence to use, modify "
        "and commercialise any feedback or suggestions provided by Customer.",
    ]),
    ("9. LIMITATION OF LIABILITY", [
        "9.1 Cap. Subject to Section 9.2, the total aggregate liability of Vendor "
        "arising out of or in connection with this Agreement shall not exceed "
        "twenty-five thousand dollars (USD 25,000).",
        "9.2 Excluded Claims. The limitation set out in Section 9.1 shall not "
        "apply to Customer's indemnification obligations under Section 10, or to "
        "Customer's payment obligations, in respect of which Customer's liability "
        "shall be unlimited.",
        "9.3 Consequential Loss. Neither party shall be liable for any indirect, "
        "incidental, special or consequential loss.",
    ]),
    ("10. INDEMNIFICATION", [
        "10.1 Customer shall defend, indemnify and hold harmless Vendor and its "
        "affiliates from and against any and all claims arising out of or relating "
        "to Customer's use of the Services or Customer Data, including claims "
        "arising in whole or in part from Vendor's own negligence. This obligation "
        "shall survive termination of this Agreement indefinitely.",
    ]),
    ("11. GENERAL", [
        "11.1 Amendment. Vendor may modify the terms of this Agreement at any time "
        "in its sole discretion by publishing an updated version to Vendor's "
        "website. Customer's continued use of the Services following publication "
        "constitutes acceptance of the modified terms.",
        "11.2 Assignment. Customer shall not assign or transfer this Agreement, "
        "whether by operation of law, merger, or in connection with a change of "
        "control of Customer, without the prior written consent of Vendor, such "
        "consent to be granted or withheld in Vendor's sole discretion. Vendor may "
        "assign this Agreement freely and without notice.",
        "11.3 Governing Law. This Agreement shall be governed by and construed in "
        "accordance with the laws of the Republic of Singapore.",
        "11.4 Venue. The parties irrevocably submit to the exclusive jurisdiction "
        "of the courts of Singapore in respect of any dispute arising out of this "
        "Agreement.",
        "11.5 Entire Agreement. This Agreement constitutes the entire agreement "
        "between the parties with respect to its subject matter.",
    ]),
]

SIGNATURE = (
    "IN WITNESS WHEREOF, the parties have executed this Agreement as of the "
    "Effective Date.<br/><br/>"
    "MERIDIAN CLOUD INFRASTRUCTURE PTE. LTD.<br/>"
    "By: ______________________&nbsp;&nbsp;&nbsp;Name: A. Tanaka&nbsp;&nbsp;&nbsp;"
    "Title: Chief Revenue Officer<br/><br/>"
    "CONTOSO SYSTEMS LTD.<br/>"
    "By: ______________________&nbsp;&nbsp;&nbsp;Name: ____________&nbsp;&nbsp;&nbsp;"
    "Title: ____________"
)


def build() -> pathlib.Path:
    styles = getSampleStyleSheet()
    title = ParagraphStyle("t", parent=styles["Title"], fontName="Times-Bold",
                           fontSize=14, spaceAfter=4, alignment=TA_CENTER)
    heading = ParagraphStyle("h", parent=styles["Heading2"], fontName="Times-Bold",
                             fontSize=10.5, spaceBefore=11, spaceAfter=4)
    body = ParagraphStyle("b", parent=styles["BodyText"], fontName="Times-Roman",
                          fontSize=9.5, leading=13.5, alignment=TA_JUSTIFY,
                          spaceAfter=6)

    story = [Paragraph(TITLE, title), Spacer(1, 10), Paragraph(PREAMBLE, body),
             Spacer(1, 4)]
    for name, clauses in SECTIONS:
        story.append(Paragraph(name, heading))
        for clause in clauses:
            story.append(Paragraph(clause, body))
    story += [PageBreak(), Paragraph("12. EXECUTION", heading),
              Paragraph(SIGNATURE, body)]

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setFont("Times-Roman", 8)
        canvas.drawCentredString(LETTER[0] / 2, 0.55 * inch,
                                 f"Meridian Cloud Infrastructure Pte. Ltd. — "
                                 f"Confidential — Page {doc.page}")
        canvas.restoreState()

    SimpleDocTemplate(
        str(OUT), pagesize=LETTER,
        leftMargin=0.95 * inch, rightMargin=0.95 * inch,
        topMargin=0.85 * inch, bottomMargin=0.85 * inch,
        title="Master Services Agreement", author="Meridian Cloud Infrastructure Pte. Ltd.",
    ).build(story, onFirstPage=footer, onLaterPages=footer)
    return OUT


if __name__ == "__main__":
    path = build()
    from api.firewall import inspect
    from api.ingest import ingest_pdf

    doc = ingest_pdf(str(path))
    report = inspect(doc, str(path))
    print(f"wrote {path} ({path.stat().st_size:,} bytes)")
    print(f"pages          : {len(doc.pages)}")
    print(f"text extracted : {len(doc.text):,} chars")
    print(f"detected type  : {doc.contract_type.value}")
    print(f"firewall       : {'CLEAN' if report.clean else 'quarantined'}")
    from api.chunking import chunk_document
    print(f"chunks         : {len(chunk_document(doc))}")
