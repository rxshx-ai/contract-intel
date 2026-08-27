"""FastAPI surface.

The throwaway test UI is just a client of this. The API is designed properly
because the real interface, when it is designed, will consume exactly these
routes.
"""

from __future__ import annotations

import tempfile
from datetime import date
from pathlib import Path

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Query
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles

from api import db, demo
from api.llm import MODEL as llm_model_name, ExtractionUnavailable
from api.findings.backtoback import exposure_summary
from api.findings.termination import termination_cost
from api.ingest import ingest_pdf, ingest_text
from api.pipeline import analyze_contract, analyze_portfolio, upcoming_deadlines
from api.risk import band
from api.schemas import OurRole

def llm_model() -> str:
    return llm_model_name


ROOT = Path(__file__).resolve().parents[1]
TENANT = "demo"          # single-tenant demo; every query still filters on it
ACTOR = "demo@contoso.example"

_conn = db.connect()
_state: dict = {"bundles": [], "gaps": [], "today": date.today()}


@asynccontextmanager
async def lifespan(_app: FastAPI):
    load_demo(_state["today"])
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


def load_demo(today: date) -> None:
    _state["today"] = today
    _state["bundles"] = demo.load(today)
    _refresh_portfolio()
    for bundle in _state["bundles"]:
        db.save_contract(_conn, TENANT, bundle.contract, bundle.result().model_dump_json())
        db.audit(_conn, TENANT, "system", "analyze", bundle.contract.id,
                 f"{len(bundle.claims)} grounded clauses")


# --------------------------------------------------------------------------
# documents
# --------------------------------------------------------------------------

@app.post("/documents")
async def upload_document(file: UploadFile = File(...)):
    """Ingest one document. Firewall runs here, before any model call."""
    from api.firewall import inspect

    raw = await file.read()
    suffix = Path(file.filename or "upload").suffix.lower()
    if suffix == ".pdf":
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(raw)
            tmp_path = tmp.name
        doc = ingest_pdf(tmp_path, file.filename)
        report = inspect(doc, tmp_path)
    else:
        doc = ingest_text(raw.decode("utf-8", errors="replace"), file.filename or "upload.txt")
        report = inspect(doc)

    db.save_document(_conn, TENANT, doc)
    db.audit(_conn, TENANT, ACTOR, "upload", doc.id, doc.filename)
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
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                tmp.write(raw)
                doc = ingest_pdf(tmp.name, name)
                paths[doc.id] = tmp.name
        else:
            doc = ingest_text(raw.decode("utf-8", errors="replace"), name)
        docs.append(doc)
        db.save_document(_conn, TENANT, doc)

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

    _state["bundles"] = [b for b in _state["bundles"]
                         if b.contract.id != bundle.contract.id] + [bundle]
    _refresh_portfolio()
    db.save_contract(_conn, TENANT, bundle.contract, bundle.result().model_dump_json())
    db.audit(_conn, TENANT, ACTOR, "analyze", bundle.contract.id,
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
    db.audit(_conn, TENANT, ACTOR, "view", contract_id)
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
    db.audit(_conn, TENANT, ACTOR, "termination_cost", contract_id, exit_date)
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


@app.get("/audit")
def audit_log(limit: int = Query(100)):
    return db.read_audit(_conn, TENANT, limit)


@app.get("/")
def index():
    return FileResponse(ROOT / "web" / "index.html")


app.mount("/static", StaticFiles(directory=ROOT / "web"), name="static")
