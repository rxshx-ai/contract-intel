"""FastAPI surface.

The throwaway test UI is just a client of this. The API is designed properly
because the real interface, when it is designed, will consume exactly these
routes.
"""

from __future__ import annotations

import json
import os
import tempfile
from datetime import date
from pathlib import Path

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Query
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles

from api import demo
from api.store import Store, is_postgres
from api.vectors import VectorIndex
from api.llm import MODEL as llm_model_name, ExtractionUnavailable
from api.findings.backtoback import exposure_summary
from api.findings.termination import termination_cost
from api.ingest import ingest_pdf, ingest_text
from api.pipeline import (ContractBundle, analyze_contract,
                          analyze_portfolio, upcoming_deadlines)
from api.risk import band
from api.schemas import OurRole

def llm_model() -> str:
    return llm_model_name


def _spill(raw: bytes) -> str:
    """Write upload bytes to a real file and CLOSE it.

    pdfplumber reads from the path, so the buffer must be flushed first --
    ingesting inside the NamedTemporaryFile block yields "No /Root object!"
    on a file that is still half in memory.
    """
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(raw)
        tmp.flush()
        return tmp.name


ROOT = Path(__file__).resolve().parents[1]
TENANT = "demo"          # single-tenant demo; every query still filters on it
ACTOR = "demo@contoso.example"

_store = Store(tenant=TENANT)
_vectors = VectorIndex(namespace=os.environ.get("TENANT", "demo"))
_state: dict = {"bundles": [], "gaps": [], "today": date.today(),
                "upload_checks": [], "last_upload": {}, "ask_index": None,
                "retriever": None}


@asynccontextmanager
async def lifespan(_app: FastAPI):
    boot(_state["today"])
    yield


app = FastAPI(title="Contract Risk & Obligation Intelligence", version="0.1.0",
              lifespan=lifespan)


def _today() -> date:
    return _state["today"]


def _bundle(contract_id: str):
    for bundle in _state["bundles"]:
        if bundle.contract.id == contract_id:
            return bundle
    raise HTTPException(status_code=404, detail=f"unknown contract {contract_id}")


def _refresh_portfolio() -> None:
    _state["gaps"] = analyze_portfolio(_state["bundles"])
    _state["ask_index"] = None      # rebuilt lazily on the next question
    _state["retriever"] = None


def boot(today: date) -> None:
    """Restore analysed contracts from the database; seed the demo if empty.

    Uploads used to live only in `_state`, so a restart lost them. They are now
    rehydrated from storage without re-extracting -- no tokens spent, and the
    service can be restarted (or replaced) without losing a user's work.
    """
    _state["today"] = today
    restored = []
    for contract_id in _store.contract_ids():
        payload = _store.get_contract(contract_id)
        if not payload or "documents" not in payload:
            continue                       # pre-persistence row; ignore
        try:
            restored.append(ContractBundle.from_payload(payload))
        except Exception:                  # a schema change, not a user error
            continue

    if restored:
        _state["bundles"] = restored
        _state["restored"] = True
    else:
        _state["bundles"] = demo.load(today)
        _state["restored"] = False
        for bundle in _state["bundles"]:
            persist(bundle)
    _refresh_portfolio()
    _store.audit("system", "boot", None,
                 f"{len(_state['bundles'])} contracts "
                 f"({'restored' if restored else 'seeded'})")


def persist(bundle) -> None:
    """Store the analysis, and index it for semantic search."""
    _store.save_contract(bundle.contract, json.dumps(bundle.to_payload()))
    for doc in bundle.docs:
        _store.save_document(doc, contract_id=bundle.contract.id)


def load_demo(today: date) -> None:
    """Reset to the seeded corpus. Used by tests."""
    _state["today"] = today
    _state["bundles"] = demo.load(today)
    _state["restored"] = False
    _refresh_portfolio()
    for bundle in _state["bundles"]:
        persist(bundle)


# --------------------------------------------------------------------------
# documents
# --------------------------------------------------------------------------

@app.post("/documents")
async def upload_document(file: UploadFile = File(...)):
    """Ingest one document. Firewall runs here, before any model call."""
    from api.firewall import inspect

    raw = await file.read()
    name = file.filename or "upload.txt"
    if name.lower().endswith(".pdf"):
        tmp_path = _spill(raw)
        try:
            doc = ingest_pdf(tmp_path, name)
            report = inspect(doc, tmp_path)
        finally:
            os.unlink(tmp_path)
    else:
        doc = ingest_text(raw.decode("utf-8", errors="replace"), name)
        report = inspect(doc)

    _store.save_document(doc)
    _store.audit(ACTOR, "upload", doc.id, doc.filename)
    return {
        "doc_id": doc.id,
        "filename": doc.filename,
        "contract_type": doc.contract_type.value,
        "chars": len(doc.text),
        "pages": len(doc.pages),
        "used_ocr": doc.used_ocr,
        "sha256": doc.sha256,
        "firewall": report.model_dump(),
    }


@app.post("/contracts")
async def create_contract(
    files: list[UploadFile] = File(...),
    title: str = Form("Uploaded contract"),
    counterparty: str = Form(""),
    our_role: str = Form("buyer"),
    our_party: str = Form(demo.OUR_PARTY),
    annual_value: float | None = Form(None),
):
    """Group documents into one contract family and analyze it end to end."""
    docs, paths = [], {}
    for upload in files:
        raw = await upload.read()
        name = upload.filename or "upload.txt"
        if name.lower().endswith(".pdf"):
            tmp_path = _spill(raw)
            doc = ingest_pdf(tmp_path, name)
            paths[doc.id] = tmp_path
        else:
            doc = ingest_text(raw.decode("utf-8", errors="replace"), name)
        docs.append(doc)
        _store.save_document(doc)

    try:
        bundle = analyze_contract(
            docs, title=title, counterparty=counterparty,
            our_role=OurRole(our_role), our_party=our_party, today=_today(),
            annual_value=annual_value, doc_paths=paths,
        )
    except ExtractionUnavailable as exc:
        raise HTTPException(status_code=503, detail=f"Extraction unavailable: {exc}")
    except Exception as exc:  # provider error, rate limit, timeout
        raise HTTPException(
            status_code=502,
            detail=f"Extraction failed against {llm_model()}: {exc}",
        )
    finally:
        for tmp_path in paths.values():
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    _state["bundles"] = [b for b in _state["bundles"]
                         if b.contract.id != bundle.contract.id] + [bundle]
    _refresh_portfolio()
    persist(bundle)
    _store.audit(ACTOR, "analyze", bundle.contract.id,
             f"{len(bundle.claims)} clauses, grounding {bundle.grounding_rate:.2%}")
    return bundle.result().model_dump()


# --------------------------------------------------------------------------
# contract views
# --------------------------------------------------------------------------

@app.get("/contracts")
def list_contracts():
    rows = []
    for bundle in _state["bundles"]:
        result = bundle.result()
        rows.append({
            "id": bundle.contract.id,
            "title": bundle.contract.title,
            "counterparty": bundle.contract.counterparty,
            "our_role": bundle.contract.our_role.value,
            "annual_value": bundle.contract.annual_value,
            "effective_date": bundle.contract.effective_date.isoformat()
            if bundle.contract.effective_date else None,
            "risk": result.risk.overall if result.risk else 0,
            "band": band(result.risk.overall if result.risk else 0),
            "findings": len(bundle.findings),
            "clauses": len(bundle.claims),
            "grounding_rate": bundle.grounding_rate,
        })
    return rows


@app.get("/contracts/{contract_id}")
def get_contract(contract_id: str):
    bundle = _bundle(contract_id)
    _store.audit(ACTOR, "view", contract_id)
    result = bundle.result().model_dump()
    result["unresolved"] = bundle.unresolved
    result["documents"] = [
        {"id": d.id, "filename": d.filename, "chars": len(d.text),
         "contract_type": d.contract_type.value}
        for d in bundle.docs
    ]
    return result


@app.get("/contracts/{contract_id}/clauses")
def get_clauses(contract_id: str, effective_only: bool = Query(True)):
    from api.family import lineage_text

    bundle = _bundle(contract_id)
    docs = {d.id: d for d in bundle.docs}
    rows = []
    for claim in bundle.claims:
        if effective_only and not claim.effective:
            continue
        doc = docs.get(claim.span.doc_id)
        rows.append({
            **claim.model_dump(),
            "document": doc.filename if doc else claim.span.doc_id,
            "page": doc.page_for(claim.span.char_start) if doc else None,
            "lineage": lineage_text(claim, bundle.claims, docs),
            "effective": claim.effective,
        })
    return rows


@app.get("/contracts/{contract_id}/obligations")
def get_obligations(contract_id: str):
    bundle = _bundle(contract_id)
    today = _today()
    return {
        "obligations": [
            {**o.model_dump(), "days_remaining": o.days_remaining(today)}
            for o in bundle.obligations
        ],
        "unresolved": bundle.unresolved,
    }


@app.get("/contracts/{contract_id}/risk")
def get_risk(contract_id: str):
    from api.risk import RUBRIC

    bundle = _bundle(contract_id)
    profile = bundle.result().risk
    return {
        "profile": profile.model_dump() if profile else None,
        "overall": profile.overall if profile else 0,
        "band": band(profile.overall if profile else 0),
        "rubric": RUBRIC,   # the algorithm is published, not hidden
    }


@app.get("/contracts/{contract_id}/findings")
def get_findings(contract_id: str):
    bundle = _bundle(contract_id)
    own = [f.model_dump() for f in bundle.findings]
    cross = [f.model_dump() for f in _state["gaps"] if contract_id in f.contract_ids]
    return own + cross


@app.post("/contracts/{contract_id}/termination-cost")
def post_termination_cost(contract_id: str, exit_date: str = Form(...)):
    bundle = _bundle(contract_id)
    try:
        parsed = date.fromisoformat(exit_date)
    except ValueError:
        raise HTTPException(status_code=400, detail="exit_date must be YYYY-MM-DD")
    cost = termination_cost(bundle.contract, bundle.claims, bundle.obligations,
                            exit_date=parsed, today=_today())
    _store.audit(ACTOR, "termination_cost", contract_id, exit_date)
    return cost.model_dump()


# --------------------------------------------------------------------------
# portfolio
# --------------------------------------------------------------------------

@app.get("/portfolio/deadlines")
def portfolio_deadlines(within_days: int = Query(365)):
    return upcoming_deadlines(_state["bundles"], _today(), within_days)


@app.get("/portfolio/gaps")
def portfolio_gaps():
    gaps = _state["gaps"]
    return {"summary": exposure_summary(gaps), "gaps": [g.model_dump() for g in gaps]}


@app.get("/portfolio/stats")
def portfolio_stats():
    from api.extract import GroundingStats
    from api.llm import MODEL

    bundles = _state["bundles"]
    total = sum(len(b.claims) for b in bundles)
    dropped = sum(b.dropped for b in bundles)
    merged = GroundingStats()
    for b in bundles:
        merged = merged.merge(b.grounding)
    return {
        "today": _today().isoformat(),
        "model": MODEL,
        "span_provenance": {
            "exact": merged.exact, "reflowed": merged.whitespace,
            "realigned": merged.fuzzy, "discarded": merged.dropped,
        },
        "contracts": len(bundles),
        "documents": sum(len(b.docs) for b in bundles),
        "grounded_claims": total,
        "discarded_ungrounded": dropped,
        "grounding_rate": 1.0 if total + dropped == 0 else total / (total + dropped),
        "hallucination_rate": 0.0,   # by construction: see api/verify.py
        "findings": sum(len(b.findings) for b in bundles) + len(_state["gaps"]),
        "injections_detected": sum(
            1 for b in bundles for f in b.findings if f.kind == "injection"),
        "committed_annual_value": sum(
            b.contract.annual_value or 0 for b in bundles
            if b.contract.our_role == OurRole.BUYER),
    }


@app.get("/portfolio/deadlines.ics")
def deadlines_ics():
    """Calendar subscription. The product's core loop is push, not pull."""
    rows = upcoming_deadlines(_state["bundles"], _today(), 730)
    lines = ["BEGIN:VCALENDAR", "VERSION:2.0",
             "PRODID:-//Contract Intelligence//EN", "CALSCALE:GREGORIAN"]
    for i, row in enumerate(rows):
        stamp = row["due_date"].replace("-", "")
        lines += [
            "BEGIN:VEVENT",
            f"UID:{row['contract_id']}-{i}@contract-intel",
            f"DTSTART;VALUE=DATE:{stamp}",
            f"DTEND;VALUE=DATE:{stamp}",
            f"SUMMARY:{row['kind'].upper()}: {row['contract']}",
            f"DESCRIPTION:{row['description']} "
            f"(counterparty: {row['counterparty']})".replace("\n", " "),
            "BEGIN:VALARM", "TRIGGER:-P14D", "ACTION:DISPLAY",
            f"DESCRIPTION:{row['kind']} deadline in 14 days", "END:VALARM",
            "END:VEVENT",
        ]
    lines.append("END:VCALENDAR")
    return PlainTextResponse("\r\n".join(lines), media_type="text/calendar")


def _retriever():
    from api.rag import Retriever

    if _state.get("retriever") is None:
        _state["retriever"] = Retriever(_state["bundles"], _state["gaps"],
                                        _today(), vectors=_vectors)
    return _state["retriever"]


@app.post("/vectors/sync")
def vectors_sync():
    """Push the current extracted layer to Pinecone."""
    return _retriever().sync_vectors()


@app.get("/vectors/stats")
def vectors_stats():
    return _vectors.stats()


@app.get("/system")
def system_info():
    """What this process is actually backed by."""
    return {
        "database": "postgres" if is_postgres() else "sqlite",
        "vectors": _vectors.stats(),
        "contracts": len(_state["bundles"]),
        "restored_from_storage": bool(_state.get("restored")),
        "retrieval": _retriever().stats,
    }


@app.get("/rag/search")
def rag_search(q: str = Query(...), k: int = Query(8),
               contract_id: str | None = Query(None)):
    """The retrieval layer on its own: verified records plus verbatim passages."""
    hits = _retriever().search(q, k=k, contract_id=contract_id)
    return {
        "query": q,
        "index": _retriever().stats,
        "hits": [{**h.citation(), "score": round(h.score, 3), "type": h.kind}
                 for h in hits],
    }


@app.post("/agent/ask")
def agent_ask(question: str = Form(...), max_steps: int = Form(5)):
    """Planning, tool-using agent over the verified layer."""
    from api.agent import Agent

    question = (question or "").strip()
    if not question:
        raise HTTPException(status_code=400, detail="question is required")
    agent = Agent(_state["bundles"], _state["gaps"], _retriever(), _today())
    try:
        result = agent.run(question, max_steps=max(2, min(6, max_steps)))
    except ExtractionUnavailable as exc:
        raise HTTPException(status_code=503, detail=f"Agent unavailable: {exc}")
    except Exception as exc:
        raise HTTPException(status_code=502,
                            detail=f"Agent failed against {llm_model()}: {exc}")
    _store.audit(ACTOR, "agent", None, question[:200])
    return result.to_dict()


@app.get("/calendar")
def calendar(start: str | None = Query(None), end: str | None = Query(None),
             contract_id: str | None = Query(None),
             group: str = Query("month")):
    """Document arrival dates, computed deadlines, and dates written in the text."""
    from api.calendar import build_calendar, by_month, in_range, summary

    events = build_calendar(_state["bundles"], _today())
    try:
        lo = date.fromisoformat(start) if start else None
        hi = date.fromisoformat(end) if end else None
    except ValueError:
        raise HTTPException(status_code=400, detail="dates must be YYYY-MM-DD")
    events = in_range(events, lo, hi)
    if contract_id:
        events = [e for e in events if e.contract_id == contract_id]
    payload = {"today": _today().isoformat(),
               "summary": summary(events, _today())}
    if group == "month":
        payload["months"] = by_month(events, _today())
    else:
        payload["events"] = [e.to_dict(_today()) for e in events]
    return payload


@app.post("/ask")
def post_ask(question: str = Form(...), contract_id: str | None = Form(None)):
    """Answer from the extracted layer only. Never from the raw contract."""
    from api.ask import Index, ask, build_records

    question = (question or "").strip()
    if not question:
        raise HTTPException(status_code=400, detail="question is required")

    index = _state.get("ask_index")
    if index is None:
        index = Index(build_records(_state["bundles"], _state["gaps"], _today()))
        _state["ask_index"] = index

    try:
        answer = ask(question, index, contract_id=contract_id or None)
    except ExtractionUnavailable as exc:
        raise HTTPException(status_code=503, detail=f"Ask unavailable: {exc}")
    except Exception as exc:
        raise HTTPException(status_code=502,
                            detail=f"Ask failed against {llm_model()}: {exc}")

    _store.audit(ACTOR, "ask", contract_id, question[:200])
    return answer.to_dict()


@app.get("/ui/model")
def ui_model():
    """Everything the designed front end renders, in the shapes it expects."""
    from api.viewmodel import build_model

    _store.audit(ACTOR, "view", "ui_model")
    return build_model(
        _state["bundles"], _state["gaps"], _today(),
        upload_checks=_state.get("upload_checks") or [],
        last_upload=_state.get("last_upload") or {},
    )


@app.get("/audit")
def audit_log(limit: int = Query(100)):
    return _store.read_audit(limit)


@app.get("/")
def index():
    """The designed front end. /raw serves the minimal test harness."""
    wired = ROOT / "web" / "app" / "index.html"
    return FileResponse(wired if wired.exists() else ROOT / "web" / "index.html")


@app.get("/raw")
def raw_index():
    return FileResponse(ROOT / "web" / "index.html")


app.mount("/vendor", StaticFiles(directory=ROOT / "web" / "app" / "vendor"),
          name="vendor")
app.mount("/_ds", StaticFiles(directory=ROOT / "web" / "app" / "_ds"), name="ds")
app.mount("/static", StaticFiles(directory=ROOT / "web"), name="static")


@app.get("/support.js")
def support_js():
    return FileResponse(ROOT / "web" / "app" / "support.js",
                        media_type="application/javascript")
