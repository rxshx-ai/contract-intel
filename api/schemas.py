"""Frozen data contract between every module.

Invariants encoded here:
  1. Every claim carries a verbatim Span. verify.py enforces exact-substring.
  2. Obligations are materialized by temporal.py, never emitted by the LLM.
  5. `missing_clause` is the only Finding kind permitted to carry no evidence.
"""

from __future__ import annotations

from datetime import date, datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

Party = Literal["us", "counterparty", "mutual", "na"]
Severity = Literal["critical", "high", "medium", "low", "info"]


class ClauseType(str, Enum):
    """CUAD-aligned vocabulary, trimmed to the types that drive our findings."""

    # commercial
    PAYMENT_TERMS = "payment_terms"
    PRICE_INCREASE = "price_increase"
    MINIMUM_COMMITMENT = "minimum_commitment"
    MOST_FAVORED_NATION = "most_favored_nation"
    # term & exit
    EFFECTIVE_DATE = "effective_date"
    TERM = "term"
    AUTO_RENEWAL = "auto_renewal"
    NOTICE_PERIOD = "notice_period"
    TERMINATION_CONVENIENCE = "termination_for_convenience"
    TERMINATION_CAUSE = "termination_for_cause"
    EARLY_TERMINATION_FEE = "early_termination_fee"
    CURE_PERIOD = "cure_period"
    # liability
    LIABILITY_CAP = "limitation_of_liability"
    UNCAPPED_CARVEOUT = "uncapped_liability_carveout"
    INDEMNIFICATION = "indemnification"
    INSURANCE = "insurance"
    WARRANTY = "warranty"
    # service
    SLA = "service_level_agreement"
    SLA_CREDIT = "service_level_credit"
    SUPPORT_RESPONSE = "support_response_time"
    # ip & data
    IP_ASSIGNMENT = "ip_assignment"
    LICENSE_GRANT = "license_grant"
    CONFIDENTIALITY = "confidentiality"
    DATA_PROTECTION = "data_protection"
    DATA_RETENTION_DELETION = "data_retention_deletion"
    BREACH_NOTIFICATION = "breach_notification"
    SUBPROCESSORS = "subprocessors"
    AUDIT_RIGHTS = "audit_rights"
    # control
    GOVERNING_LAW = "governing_law"
    VENUE = "venue"
    ASSIGNMENT = "assignment"
    CHANGE_OF_CONTROL = "change_of_control"
    UNILATERAL_AMENDMENT = "unilateral_amendment"
    NON_COMPETE = "non_compete"
    EXCLUSIVITY = "exclusivity"
    FORCE_MAJEURE = "force_majeure"
    SURVIVAL = "survival"


class ContractType(str, Enum):
    MSA = "msa"
    NDA = "nda"
    SOW = "sow"
    DPA = "dpa"
    ORDER_FORM = "order_form"
    AMENDMENT = "amendment"
    UNKNOWN = "unknown"


class OurRole(str, Enum):
    """Which side of the paper we are on. Risk is meaningless without this."""

    BUYER = "buyer"      # we receive the service, we pay
    SELLER = "seller"    # we provide the service, we get paid
    MUTUAL = "mutual"    # e.g. a mutual NDA


# --------------------------------------------------------------------------
# grounding
# --------------------------------------------------------------------------

class Span(BaseModel):
    """A verbatim pointer into a source document. The unit of grounding."""

    doc_id: str
    char_start: int
    char_end: int
    quote: str

    def is_grounded_in(self, text: str) -> bool:
        """Invariant 1, mechanized. verify.py is the only caller that matters."""
        return text[self.char_start : self.char_end] == self.quote


# --------------------------------------------------------------------------
# documents
# --------------------------------------------------------------------------

class PageMark(BaseModel):
    """Maps a char offset range back to a page, so the UI can jump to it."""

    page: int
    char_start: int
    char_end: int
    ocr_confidence: float | None = None


class Document(BaseModel):
    id: str
    filename: str
    text: str
    ingested_at: datetime | None = None   # when we first read this file
    pages: list[PageMark] = Field(default_factory=list)
    contract_type: ContractType = ContractType.UNKNOWN
    used_ocr: bool = False
    sha256: str = ""

    def page_for(self, char_offset: int) -> int | None:
        for pm in self.pages:
            if pm.char_start <= char_offset < pm.char_end:
                return pm.page
        return None


class TamperIndicator(BaseModel):
    """Something in the document was hidden from a human reader."""

    kind: Literal[
        "invisible_text", "tiny_font", "offscreen_text",
        "metadata_payload", "injection_language",
    ]
    detail: str
    excerpt: str
    page: int | None = None


class FirewallReport(BaseModel):
    doc_id: str
    indicators: list[TamperIndicator] = Field(default_factory=list)
    quarantined: bool = False

    @property
    def clean(self) -> bool:
        return not self.indicators


# --------------------------------------------------------------------------
# extracted layer
# --------------------------------------------------------------------------

class ClauseClaim(BaseModel):
    id: str
    contract_id: str
    clause_type: ClauseType
    party_favored: Party = "na"
    span: Span
    fields: dict[str, Any] = Field(default_factory=dict)
    confidence: float = 0.0
    superseded_by: str | None = None
    # set by family.py when an amendment overrides this clause
    supersedes: str | None = None

    @property
    def effective(self) -> bool:
        return self.superseded_by is None


class TemporalRule(BaseModel):
    """A date *rule*, not a date. Contracts rarely contain absolute deadlines."""

    id: str
    contract_id: str
    kind: Literal["renewal", "notice", "expiry", "payment", "report", "cure"]
    anchor: Literal[
        # resolvable against the contract calendar
        "effective_date", "term_end", "signature_date",
        "anniversary", "month_end", "quarter_end",
        # event-driven: no date until the event happens
        "invoice_date", "breach_date", "event",
    ]
    offset_days: int = 0          # negative = before the anchor
    recurrence: str | None = None  # ISO-8601 duration, e.g. "P12M"
    condition: str | None = None   # quoted, never paraphrased
    consequence: str = ""
    owed_by: Literal["us", "counterparty", "either"] = "us"
    span: Span


class Obligation(BaseModel):
    """Materialized by temporal.py. The LLM never produces one of these."""

    rule_id: str
    contract_id: str
    kind: str
    anchor: str = "effective_date"
    due_date: date
    owed_by: Literal["us", "counterparty", "either"]
    description: str
    derivation: list[str] = Field(default_factory=list)
    consequence_if_missed: str = ""

    def days_remaining(self, today: date) -> int:
        return (self.due_date - today).days


# --------------------------------------------------------------------------
# analysis layer
# --------------------------------------------------------------------------

class RiskContribution(BaseModel):
    clause_id: str | None
    points: int
    reason: str


class RiskAxis(BaseModel):
    axis: Literal["financial", "lockin", "liability", "compliance", "operational"]
    score: int = 0  # 0-100, higher = worse for us
    contributions: list[RiskContribution] = Field(default_factory=list)


class RiskProfile(BaseModel):
    contract_id: str
    our_role: OurRole
    axes: list[RiskAxis] = Field(default_factory=list)

    @property
    def overall(self) -> int:
        """Worst-axis, not mean. A single catastrophic axis is not averaged away."""
        return max((a.score for a in self.axes), default=0)


class Finding(BaseModel):
    """Unified output of every analysis module. Written once, rendered once."""

    id: str
    kind: Literal[
        "risky_clause", "missing_clause", "adversarial_pattern",
        "backtoback_gap", "injection", "asymmetry",
    ]
    severity: Severity
    title: str
    explanation: str
    evidence: list[Span] = Field(default_factory=list)
    contract_ids: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _absence_is_the_only_unevidenced_finding(self) -> Finding:
        if not self.evidence and self.kind != "missing_clause":
            raise ValueError(
                f"Finding kind '{self.kind}' must carry evidence; only "
                f"'missing_clause' may be unevidenced (invariant 5)."
            )
        return self


class AsymmetryReport(BaseModel):
    contract_id: str
    our_rights: list[Span] = Field(default_factory=list)
    their_rights: list[Span] = Field(default_factory=list)

    @property
    def index(self) -> float:
        """0.0 = perfectly balanced, 1.0 = fully one-sided against us."""
        total = len(self.our_rights) + len(self.their_rights)
        if total == 0:
            return 0.0
        return len(self.their_rights) / total


class TerminationCost(BaseModel):
    contract_id: str
    exit_date: date
    line_items: list[dict[str, Any]] = Field(default_factory=list)
    total: float = 0.0
    currency: str = "USD"
    notes: list[str] = Field(default_factory=list)


class Contract(BaseModel):
    """A contract is a *family* of documents, not a single PDF."""

    id: str
    title: str
    counterparty: str = ""
    our_role: OurRole = OurRole.BUYER
    contract_type: ContractType = ContractType.UNKNOWN
    doc_ids: list[str] = Field(default_factory=list)
    effective_date: date | None = None
    term_end: date | None = None
    annual_value: float | None = None
    currency: str = "USD"


class AnalysisResult(BaseModel):
    """Everything the UI needs for one contract, in one payload."""

    contract: Contract
    clauses: list[ClauseClaim] = Field(default_factory=list)
    obligations: list[Obligation] = Field(default_factory=list)
    findings: list[Finding] = Field(default_factory=list)
    risk: RiskProfile | None = None
    asymmetry: AsymmetryReport | None = None
    firewall: list[FirewallReport] = Field(default_factory=list)
    grounding_rate: float = 1.0
    dropped_claims: int = 0
