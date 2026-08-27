"""Ask: questions answered from the extracted layer, never from the raw contract.

When a document is uploaded it is already turned into verified clauses,
computed deadlines, findings and absences. Ask retrieves over THOSE RECORDS.

Three consequences, and they are the whole point:

  * An answer can only be assembled from things that were already verified, so
    invariant 1 extends to Q&A for free.
  * The model never sees the contract text as a wall to summarise. It receives
    a handful of structured records and picks which ones answer the question --
    it emits record IDs, not quotes, so it *cannot* fabricate a citation.
  * Questions that span the portfolio ("which contracts auto-renew before
    March?") are answerable, because the records are structured and comparable.
    Retrieval over raw PDF chunks cannot do that.

Retrieval is BM25 over the record text plus structured boosts. Groq serves no
embedding model; for a few hundred short, highly structured records lexical
retrieval is also more precise than vectors. Swapping in embeddings later means
changing `rank()` and nothing else.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Iterable

from pydantic import BaseModel

from api.llm import ExtractionUnavailable, complete_json
from api.schemas import OurRole

TOP_K = 12
MAX_QUOTE_CHARS = 320

_WORD = re.compile(r"[a-z0-9$%.]+")
_STOP = {
    "the", "a", "an", "of", "to", "in", "for", "on", "and", "or", "is", "are",
    "we", "our", "us", "i", "do", "does", "did", "what", "which", "how", "when",
    "who", "whom", "any", "all", "with", "that", "this", "it", "be", "by", "at",
    "from", "as", "if", "can", "have", "has", "there", "their", "they",
}


def tokenize(text: str) -> list[str]:
    return [t for t in _WORD.findall(text.lower()) if t not in _STOP and len(t) > 1]


# --------------------------------------------------------------------------

@dataclass
class Record:
    """One verified thing we know. The unit of both retrieval and citation."""

    id: str
    contract_id: str
    contract: str
    kind: str          # clause | obligation | absence | finding | gap | fact
    title: str
    body: str
    quote: str | None = None
    src_file: str | None = None
    src_start: int | None = None
    src_end: int | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def searchable(self) -> str:
        return " ".join(filter(None, [self.contract, self.kind, self.title,
                                      self.body, self.quote or ""]))

    def for_model(self) -> dict[str, Any]:
        out = {
            "id": self.id,
            "contract": self.contract,
            "kind": self.kind,
            "title": self.title,
            "detail": self.body,
        }
        if self.quote:
            out["verbatim_quote"] = self.quote[:MAX_QUOTE_CHARS]
        return out

    def citation(self) -> dict[str, Any]:
        return {
            "record_id": self.id,
            "contract_id": self.contract_id,
            "contract": self.contract,
            "kind": self.kind,
            "title": self.title,
            "quote": self.quote,
            "file": self.src_file,
            "start": self.src_start,
            "end": self.src_end,
        }


# --------------------------------------------------------------------------
# building the index from the analysis output
# --------------------------------------------------------------------------

def build_records(bundles, gaps, today: date) -> list[Record]:
    from api.viewmodel import CLAUSE_LABELS, FINDING_KIND_LABEL

    records: list[Record] = []
    for bundle in bundles:
        contract = bundle.contract
        name = contract.counterparty or contract.title
        cid = contract.id

        role = {OurRole.BUYER: "we buy from them",
                OurRole.SELLER: "we sell to them",
                OurRole.MUTUAL: "mutual"}.get(contract.our_role, "")
        # Which side of the paper we are on. Without it, "how fast must our
        # PROVIDER tell us about a breach" retrieves the clause where WE promise
        # a customer 24 hours -- same subject, opposite direction, confidently
        # wrong answer.
        #
        # This is carried in METADATA and applied as a ranking boost, not
        # written into the searchable text. Stuffing "supplier vendor provider"
        # into every buyer-side record made the word match all of them equally
        # and buried the clause that actually answered the question.
        side_label = {OurRole.BUYER: "their obligation to us (supplier contract)",
                      OurRole.SELLER: "our obligation to them (customer contract)",
                      OurRole.MUTUAL: "mutual"}.get(contract.our_role, "")
        facts = [f"{name} is a {contract.contract_type.value.upper()} where {role}."]
        if contract.effective_date:
            facts.append(f"Effective date {contract.effective_date.isoformat()}.")
        if contract.annual_value:
            facts.append(f"Annual value {contract.annual_value:,.0f} {contract.currency}.")
        profile = bundle.result().risk
        if profile:
            facts.append(f"Overall attention score {profile.overall} out of 100.")
        records.append(Record(
            id=f"{cid}:fact", contract_id=cid, contract=name, kind="fact",
            title=f"Key facts about the {name} agreement",
            body=" ".join(facts),
            meta={"annual_value": contract.annual_value,
                  "role": contract.our_role.value, "side": contract.our_role.value},
        ))

        for claim in bundle.claims:
            if not claim.effective:
                continue
            label = CLAUSE_LABELS.get(claim.clause_type, claim.clause_type.value)
            fields = ", ".join(f"{k}={v}" for k, v in claim.fields.items()
                               if k != "grounding")
            records.append(Record(
                id=claim.id, contract_id=cid, contract=name, kind="clause",
                title=f"{label} — {name}",
                body=f"{label}. {claim.clause_type.value}. {side_label}. "
                     f"Favours {claim.party_favored}. {fields}",
                quote=claim.span.quote,
                src_file=_file_of(bundle, claim.span.doc_id),
                src_start=claim.span.char_start, src_end=claim.span.char_end,
                meta={"clause_type": claim.clause_type.value,
                      "side": contract.our_role.value, **claim.fields},
            ))

        rules = {r.id: r for r in bundle.rules}
        for ob in bundle.obligations:
            days = ob.days_remaining(today)
            rule = rules.get(ob.rule_id)
            records.append(Record(
                id=f"{cid}:ob:{ob.rule_id}:{ob.due_date}", contract_id=cid,
                contract=name, kind="obligation",
                title=f"{ob.kind} deadline {ob.due_date.isoformat()} — {name}",
                body=(f"A {ob.kind} obligation falls due {ob.due_date.isoformat()}, "
                      f"{abs(days)} days {'ago' if days < 0 else 'from now'}. "
                      f"Owed by {ob.owed_by}. {ob.description} "
                      f"Derivation: {' | '.join(ob.derivation)}"),
                quote=rule.span.quote if rule else None,
                src_file=_file_of(bundle, rule.span.doc_id) if rule else None,
                src_start=rule.span.char_start if rule else None,
                src_end=rule.span.char_end if rule else None,
                meta={"due": ob.due_date.isoformat(), "days": days,
                      "obligation_kind": ob.kind, "anchor": ob.anchor,
                      "side": contract.our_role.value},
            ))

        for finding in bundle.findings:
            kind = "absence" if finding.kind == "missing_clause" else "finding"
            records.append(Record(
                id=finding.id, contract_id=cid, contract=name, kind=kind,
                title=finding.title,
                body=(f"{FINDING_KIND_LABEL.get(finding.kind, finding.kind)}, "
                      f"severity {finding.severity}. {finding.explanation}"),
                quote=finding.evidence[0].quote if finding.evidence else None,
                src_file=(_file_of(bundle, finding.evidence[0].doc_id)
                          if finding.evidence else None),
                src_start=finding.evidence[0].char_start if finding.evidence else None,
                src_end=finding.evidence[0].char_end if finding.evidence else None,
                meta={"severity": finding.severity, "finding_kind": finding.kind,
                      "side": contract.our_role.value},
            ))

    for gap in gaps:
        meta = gap.metadata
        records.append(Record(
            id=gap.id, contract_id=gap.contract_ids[0] if gap.contract_ids else "",
            contract=" / ".join(filter(None, [meta.get("outbound_contract"),
                                              meta.get("inbound_contract")])),
            kind="gap", title=gap.title,
            body=f"Cross-contract flow-down gap. {gap.explanation}",
            quote=gap.evidence[0].quote if gap.evidence else None,
            meta={"dimension": meta.get("dimension"),
                  "outbound": meta.get("outbound_value"),
                  "inbound": meta.get("inbound_value")},
        ))
    return records


def _file_of(bundle, doc_id: str) -> str | None:
    for doc in bundle.docs:
        if doc.id == doc_id:
            return doc.filename
    return None


# --------------------------------------------------------------------------
# retrieval
# --------------------------------------------------------------------------

class Index:
    """Okapi BM25 over record text. No dependencies, deterministic, testable."""

    K1 = 1.5
    B = 0.75

    def __init__(self, records: Iterable[Record]):
        self.records = list(records)
        self.docs = [tokenize(r.searchable) for r in self.records]
        self.lengths = [len(d) for d in self.docs]
        self.avg_len = (sum(self.lengths) / len(self.lengths)) if self.lengths else 0.0
        self.freqs = [Counter(d) for d in self.docs]
        self.df: Counter[str] = Counter()
        for doc in self.docs:
            self.df.update(set(doc))
        self.n = len(self.docs)

    def _idf(self, term: str) -> float:
        df = self.df.get(term, 0)
        return math.log(1 + (self.n - df + 0.5) / (df + 0.5))

    def score(self, i: int, terms: list[str]) -> float:
        total = 0.0
        freq, length = self.freqs[i], self.lengths[i] or 1
        for term in terms:
            f = freq.get(term, 0)
            if not f:
                continue
            denom = f + self.K1 * (1 - self.B + self.B * length / (self.avg_len or 1))
            total += self._idf(term) * (f * (self.K1 + 1)) / denom
        return total

    def rank(self, question: str, top_k: int = TOP_K,
             contract_id: str | None = None) -> list[tuple[Record, float]]:
        terms = tokenize(question)
        lowered = question.lower()
        scored: list[tuple[Record, float]] = []
        for i, record in enumerate(self.records):
            if contract_id and record.contract_id != contract_id:
                continue
            score = self.score(i, terms)
            score += _boost(record, lowered)
            if score > 0:
                scored.append((record, score))
        scored.sort(key=lambda pair: -pair[1])
        return scored[:top_k]


_TIME_WORDS = ("when", "deadline", "due", "date", "soon", "days", "month",
               "expire", "expiry")
_RENEWAL_WORDS = ("renew", "renewal", "auto-renew", "notice", "cancel",
                  "terminate", "get out", "exit", "non-renewal")
_ABSENCE_WORDS = ("missing", "absent", "without", "lack", "no ", "not have",
                  "isn't", "does not", "silent", "nothing about")
_RISK_WORDS = ("risk", "worst", "worry", "concern", "dangerous", "problem",
               "attention", "exposed", "exposure", "uncovered", "underwrit",
               "liable", "on the hook")
_CROSS_WORDS = ("across", "portfolio", "chain", "flow-down", "flow down",
                "downstream", "upstream", "both sides", "supplier", "customers",
                "everything", "all contracts")
_MONEY_WORDS = ("cost", "pay", "fee", "price", "spend", "value", "much",
                "worth", "annual")


# Words that pin a question to one side of the relationship.
_SUPPLIER_WORDS = ("supplier", "provider", "vendor", "they give us",
                   "we buy", "our provider", "upstream")
_CUSTOMER_WORDS = ("customer", "client", "we promise", "we sell", "we give",
                   "downstream", "we owe")


def _boost(record: Record, question: str) -> float:
    """Structured nudges.

    Lexical similarity alone answers the wrong question for "what have we
    missed" or "where are we exposed", where the answer is a record TYPE rather
    than a matching word. These boosts encode that.
    """
    boost = 0.0
    kind = record.kind
    if any(w in question for w in _TIME_WORDS) and kind == "obligation":
        boost += 2.0
    if any(w in question for w in _RENEWAL_WORDS):
        if kind == "obligation" and record.meta.get("anchor") == "term_end":
            boost += 6.0          # the renewal deadline, not a quarterly report
        elif kind == "obligation":
            boost += 0.5
        elif record.meta.get("clause_type") in ("auto_renewal", "notice_period",
                                                "termination_for_convenience"):
            boost += 3.0
    if any(w in question for w in _ABSENCE_WORDS) and kind == "absence":
        boost += 3.0
    if any(w in question for w in _RISK_WORDS) and kind in ("finding", "gap"):
        boost += 3.0
    if any(w in question for w in _CROSS_WORDS) and kind == "gap":
        boost += 5.0
    if any(w in question for w in _MONEY_WORDS) and kind == "fact":
        boost += 1.5
    side = record.meta.get("side")
    if side:
        if any(w in question for w in _SUPPLIER_WORDS):
            boost += 6.0 if side == "buyer" else -5.0
        elif any(w in question for w in _CUSTOMER_WORDS):
            boost += 6.0 if side == "seller" else -5.0
    if record.contract:
        for token in record.contract.lower().replace("/", " ").split():
            if len(token) > 3 and token.strip(",.") in question:
                boost += 3.0
                break
    return boost


# --------------------------------------------------------------------------
# answering
# --------------------------------------------------------------------------

class RawAnswer(BaseModel):
    answer: str | None = None
    cited_record_ids: list[str] | None = None
    sufficient: bool | None = None
    missing: str | None = None


SYSTEM = """You answer questions about contracts using ONLY the records supplied.

Each record was extracted from a real document and verified character-for-
character against it. You are NOT reading the contracts; you are reading a set
of already-verified findings about them.

RULES
1. Use only the records given. If they do not contain the answer, set
   sufficient=false and put what is missing in `missing`. Never guess, and never
   fill a gap from general knowledge of how contracts usually work.
2. Do not quote text yourself. Cite the record ids you relied on in
   `cited_record_ids`; the quotes are attached to those records and are shown
   to the user automatically.
3. Cite every record your answer depends on. Cite nothing you did not use.
4. If several contracts are relevant, cover them rather than picking one.
   "Which of our suppliers..." is asking about all of them.
5. Answer in plain language, in two or three sentences. Give the specific
   figure, date or clause the records contain rather than describing it. Say
   which contract each fact came from when more than one is involved.
6. Dates and totals in the records were computed by code. Repeat them exactly;
   do not recalculate.
7. Direction matters. "Our provider / supplier / vendor" means a contract
   where WE BUY -- their obligation to us. "Our customer / client" means a
   contract where WE SELL -- our obligation to them. Each record says which it
   is. Answering with the wrong side is the same subject and the opposite
   meaning, so check before you answer.
8. You are decision support, not a lawyer. State what the documents say. Do not
   advise whether to sign, and do not opine on enforceability.
"""


@dataclass
class Answer:
    question: str
    answer: str
    citations: list[dict[str, Any]]
    sufficient: bool
    missing: str = ""
    considered: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "answer": self.answer,
            "citations": self.citations,
            "sufficient": self.sufficient,
            "missing": self.missing,
            "considered": self.considered,
        }


_GENERIC_NAME_TOKENS = {
    "the", "and", "ltd", "limited", "inc", "llc", "plc", "gmbh", "corp",
    "company", "systems", "group", "partners", "holdings", "cloud",
    "services", "financial", "technologies", "international",
}


def infer_contract(question: str, retriever) -> str | None:
    """The contract a question names, if it names exactly one.

    Matches on distinctive tokens of the counterparty name -- "northwind",
    "acme" -- ignoring the corporate furniture ("Systems", "Ltd", "Group")
    that several counterparties share. Returns None when the question names
    none or more than one, so an ambiguous question still searches everything.
    """
    lowered = f" {question.lower()} "
    matched: set[str] = set()
    for record in getattr(retriever, "records", []):
        if not record.contract_id or not record.contract:
            continue
        # A cross-contract gap names BOTH parties under one contract id, so it
        # makes every such question look ambiguous. Skip them here.
        if record.kind == "gap":
            continue
        for token in record.contract.lower().replace("/", " ").split():
            token = token.strip(",.()")
            if len(token) <= 3 or token in _GENERIC_NAME_TOKENS:
                continue
            if f" {token} " in lowered or f" {token}'" in lowered:
                matched.add(record.contract_id)
                break
    return matched.pop() if len(matched) == 1 else None


def keyword_answer(question: str, retriever, contract_id: str | None = None,
                   reason: str = "") -> Answer:
    """What we can say without the model: the matching passages, unsynthesised.

    Retrieval does not need the model -- BM25 and the stored vectors are local.
    So a model outage costs the user the SUMMARY, not the search. Returning the
    matches with an honest banner is more useful than an error, and it cannot
    be mistaken for an answer because nothing has been written.
    """
    contract_id = contract_id or infer_contract(question, retriever)
    hits = retriever.search(question, k=TOP_K, contract_id=contract_id)
    citations = [hit.citation() for hit in hits][:6]

    if not citations:
        text = ("The model is unavailable, and nothing in the contracts matches "
                "those words either.")
    else:
        where = ", ".join(sorted({c.get("contract") or "" for c in citations
                                  if c.get("contract")})[:3])
        text = (f"The model is unavailable, so this has not been read or "
                f"summarised. These {len(citations)} passages match your words"
                + (f", from {where}" if where else "") + ". "
                f"Every one is quoted from your documents.")

    return Answer(
        question=question, answer=text, citations=citations,
        sufficient=False,
        missing=reason or "the model is not answering right now",
        considered=len(hits),
    )


def ask(
    question: str,
    retriever,
    contract_id: str | None = None,
    top_k: int = TOP_K,
) -> Answer:
    """Answer from retrieved, already-verified material.

    `retriever` is the hybrid Retriever, so Ask and the agent search the same
    way. Ask used to run BM25 over records only, which meant the two surfaces
    could give different answers to the same question -- and Ask missed
    anything phrased differently from the contract ("tell us" vs "notify").
    """
    # A question that names a contract is asking about THAT contract. Without
    # this, "can we get out of the northwind deal early" retrieved Vertex's
    # termination clause -- a confident answer about the wrong agreement.
    contract_id = contract_id or infer_contract(question, retriever)

    hits = retriever.search(question, k=top_k, contract_id=contract_id)
    if not hits:
        return Answer(
            question=question,
            answer="Nothing in the extracted contracts touches on that.",
            citations=[], sufficient=False,
            missing="No verified record matched the question.", considered=0,
        )

    by_id = {hit.id: hit.payload for hit in hits}
    payload = [hit.payload.for_model() for hit in hits]
    user = (
        f"Question: {question}\n\n"
        f"Records ({len(payload)}), each already verified against its source "
        f"document:\n"
        + "\n".join(
            f"- id={r['id']} | {r['contract']} | {r['kind']} | {r['title']}\n"
            f"    {r['detail']}"
            + (f"\n    quote: \"{r['verbatim_quote']}\"" if r.get("verbatim_quote") else "")
            for r in payload
        )
    )

    result = complete_json(SYSTEM, user, RawAnswer, schema_name="contract_answer",
                           max_tokens=1200)

    # The model returns ids, not quotes. Anything it invents is dropped here, so
    # a fabricated citation cannot reach the user.
    cited = [by_id[rid].citation()
             for rid in (result.cited_record_ids or []) if rid in by_id]

    sufficient = bool(result.sufficient) and bool(result.answer)
    answer_text = (result.answer or "").strip()
    if not sufficient and not answer_text:
        answer_text = "That is not something the extracted records can answer."

    return Answer(
        question=question, answer=answer_text, citations=cited,
        sufficient=sufficient, missing=(result.missing or "").strip(),
        considered=len(hits),
    )
