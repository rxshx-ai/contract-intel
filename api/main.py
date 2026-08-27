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
from fastapi.responses import (FileResponse, PlainTextResponse,
                               StreamingResponse)
from fastapi.staticfiles import StaticFiles

from api import demo
from api.store import Store, is_postgres
from api.vectors import VectorIndex
from api.llm import HEALTH, MODEL as llm_model_name, ExtractionUnavailable
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
                "retriever": None, "skipped": []}


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
    restored, skipped = [], []
    for contract_id in _store.contract_ids():
        payload = _store.get_contract(contract_id)
        if not payload or "documents" not in payload:
            skipped.append((contract_id, "stored before analysis was persisted"))
            continue
        try:
            restored.append(ContractBundle.from_payload(payload))
        except Exception as exc:           # schema drift, not a user error
            # Silence here is dangerous: a dropped contract becomes "that
            # contract does not exist" in an answer, stated as fact. Record it
            # and surface it on /system instead.
            skipped.append((contract_id, f"{type(exc).__name__}: {exc}"))
    _state["skipped"] = skipped
    for contract_id, why in skipped:
        print(f"[boot] could not restore {contract_id}: {why}", flush=True)

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
                 f"({'restored' if restored else 'seeded'}), "
                 f"{len(skipped)} skipped")


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

@app.post("/contracts/stream")
async def create_contract_stream(
    files: list[UploadFile] = File(...),
    our_role: str = Form("buyer"),
    our_party: str = Form(demo.OUR_PARTY),
):
    """Analyse uploads, reporting every stage as it happens.

    One contract per file, so a bulk upload becomes a list the user can click
    through. Each file reports: the text it read, where in that text the model
    is working, every clause found with its offsets, the deadlines derived, and
    finally what changed in the interface as a result.
    """
    import queue
    import threading

    spooled = []
    for upload in files:
        raw = await upload.read()
        spooled.append((upload.filename or "upload.txt", raw))
    if not spooled:
        raise HTTPException(status_code=400, detail="no files")

    events: "queue.Queue[dict | None]" = queue.Queue()
    today = _today()

    def work():
        try:
            for index, (name, raw) in enumerate(spooled):
                _analyse_one(name, raw, index, len(spooled), our_role,
                             our_party, today, events.put)
            _refresh_portfolio()
            events.put({"type": "all_done", "contracts": len(_state["bundles"]),
                        "gaps": len(_state["gaps"])})
        except Exception as exc:                       # noqa: BLE001
            events.put({"type": "error", "message": str(exc)[:300]})
        finally:
            events.put(None)

    threading.Thread(target=work, daemon=True).start()

    def emit():
        while True:
            event = events.get()
            if event is None:
                break
            yield f"data: {json.dumps(event)}\n\n"
        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return StreamingResponse(
        emit(), media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


def _analyse_one(name, raw, index, total, our_role, our_party, today, emit):
    """Ingest and analyse one file, emitting progress and a change summary."""
    from api.calendar import build_calendar

    emit({"type": "file_start", "index": index, "total": total, "filename": name})

    tmp_path = None
    try:
        if name.lower().endswith(".pdf"):
            tmp_path = _spill(raw)
            doc = ingest_pdf(tmp_path, name)
        else:
            doc = ingest_text(raw.decode("utf-8", errors="replace"), name)

        # The text goes to the client so it can show the document being read.
        emit({"type": "ingested", "doc_id": doc.id, "filename": doc.filename,
              "chars": len(doc.text), "pages": len(doc.pages),
              "used_ocr": doc.used_ocr, "contract_type": doc.contract_type.value,
              "text": doc.text})

        before = _interface_counts(today)
        contract_id = f"k_{doc.sha256[:8]}"
        bundle = analyze_contract(
            [doc], title=name, counterparty=_counterparty_from(name),
            our_role=OurRole(our_role), our_party=our_party, today=today,
            contract_id=contract_id,
            doc_paths={doc.id: tmp_path} if tmp_path else None,
            on_event=emit)

        _state["bundles"] = [b for b in _state["bundles"]
                             if b.contract.id != bundle.contract.id] + [bundle]
        _refresh_portfolio()
        persist(bundle)
        _store.audit(ACTOR, "analyze", bundle.contract.id, name)

        after = _interface_counts(today)
        emit({"type": "changes", "filename": name,
              "contract_id": bundle.contract.id,
              "counterparty": bundle.contract.counterparty,
              "risk": (bundle.result().risk.overall if bundle.result().risk else 0),
              "delta": {k: after[k] - before[k] for k in after},
              "totals": after})
        emit({"type": "file_done", "index": index, "filename": name,
              "contract_id": bundle.contract.id,
              "clauses": len(bundle.claims), "findings": len(bundle.findings),
              "deadlines": len(bundle.obligations),
              "grounding": round(bundle.grounding_rate, 4)})
    except ExtractionUnavailable as exc:
        emit({"type": "file_error", "filename": name,
              "message": f"Extraction unavailable: {exc}"})
    except Exception as exc:                           # noqa: BLE001
        emit({"type": "file_error", "filename": name, "message": str(exc)[:240]})
    finally:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


def _interface_counts(today: date) -> dict[str, int]:
    """What the interface currently shows, so a change can be reported as a
    delta rather than as a claim."""
    from api.calendar import build_calendar

    bundles = _state["bundles"]
    events = build_calendar(bundles, today)
    return {
        "contracts": len(bundles),
        "clauses": sum(len(b.claims) for b in bundles),
        "findings": sum(len(b.findings) for b in bundles),
        "deadlines": sum(len(b.obligations) for b in bundles),
        "calendar_events": len(events),
        "actionable_dates": sum(1 for e in events if e.actionable),
        "flow_down_gaps": len(_state["gaps"]),
    }


def _counterparty_from(filename: str) -> str:
    stem = filename.rsplit("/", 1)[-1].rsplit(".", 1)[0]
    words = [w for w in stem.replace("_", " ").replace("-", " ").split()
             if not w.isdigit()]
    return " ".join(w.capitalize() for w in words[:4]) or stem


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
        retriever = Retriever(_state["bundles"], _state["gaps"], _today(),
                              vectors=_vectors)
        # Index automatically when the vector store is empty or out of date, so
        # hybrid retrieval works without anyone remembering to POST /vectors/sync.
        # Idempotent: upserts overwrite by id.
        if retriever.vectors.available:
            expected = len(retriever.records) + len(retriever.passages)
            held = retriever.vectors.stats().get("vectors", 0)
            if held != expected:
                retriever.sync_vectors()
        _state["retriever"] = retriever
    return _state["retriever"]


@app.post("/vectors/sync")
def vectors_sync():
    """Push the current extracted layer to Pinecone."""
    return _retriever().sync_vectors()


@app.get("/vectors/stats")
def vectors_stats():
    return _vectors.stats()


@app.delete("/contracts/{contract_id}")
def delete_contract(contract_id: str):
    """Remove a contract, its documents and its vectors together."""
    _bundle(contract_id)                       # 404 if unknown
    _store.delete_contract(contract_id)
    _vectors.delete_contract(contract_id)      # same database when on pgvector
    _state["bundles"] = [b for b in _state["bundles"]
                         if b.contract.id != contract_id]
    _refresh_portfolio()
    _store.audit(ACTOR, "delete", contract_id)
    return {"deleted": contract_id, "remaining": len(_state["bundles"])}


@app.get("/health/model")
def model_health():
    """Whether the model is answering, learned from real calls."""
    return HEALTH.snapshot()


@app.get("/system")
def system_info():
    """What this process is actually backed by."""
    return {
        "database": "postgres" if is_postgres() else "sqlite",
        "model": HEALTH.snapshot(),
        "vectors": _vectors.stats(),
        "contracts": len(_state["bundles"]),
        "restored_from_storage": bool(_state.get("restored")),
        "skipped_on_boot": [{"contract_id": c, "reason": w}
                            for c, w in _state.get("skipped", [])],
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
    from api.ask import keyword_answer

    retriever = _retriever()
    agent = Agent(_state["bundles"], _state["gaps"], retriever, _today())
    try:
        result = agent.run(question, max_steps=max(2, min(6, max_steps)))
        payload = result.to_dict()
        payload["degraded"] = False
    except Exception as exc:                           # noqa: BLE001
        HEALTH.failed(exc)
        fallback = keyword_answer(question, retriever, reason=str(exc)[:160])
        payload = {**fallback.to_dict(), "plan": [], "steps": [], "tables": [],
                   "degraded": True, "stopped_early": False}

    payload["model"] = HEALTH.snapshot()
    _store.audit(ACTOR, "agent", None, question[:200])
    return payload


@app.get("/agent/stream")
def agent_stream(question: str = Query(...), max_steps: int = Query(5)):
    """The same agent run, reported as it happens (Server-Sent Events).

    The agent runs on a worker thread and pushes events into a queue; this
    generator drains the queue. Running it inline would buffer everything until
    the answer, which defeats the point -- the tool calls and what they
    retrieved are the interesting part, and they are worth watching arrive.
    """
    import queue
    import threading

    from api.agent import Agent
    from api.ask import keyword_answer

    question = (question or "").strip()
    if not question:
        raise HTTPException(status_code=400, detail="question is required")

    events: "queue.Queue[dict | None]" = queue.Queue()
    agent = Agent(_state["bundles"], _state["gaps"], _retriever(), _today())

    def work():
        try:
            agent.run(question, max_steps=max(2, min(6, max_steps)),
                      on_event=events.put)
        except ExtractionUnavailable as exc:
            events.put({"type": "error", "message": f"Agent unavailable: {exc}"})
        except Exception as exc:                       # noqa: BLE001
            HEALTH.failed(exc)
            fallback = keyword_answer(question, _retriever(),
                                      reason=str(exc)[:160])
            events.put({"type": "degraded", "model": HEALTH.snapshot(),
                        **fallback.to_dict()})
        finally:
            events.put(None)

    threading.Thread(target=work, daemon=True).start()
    _store.audit(ACTOR, "agent_stream", None, question[:200])

    def emit():
        yield f"data: {json.dumps({'type': 'started', 'question': question})}\n\n"
        while True:
            event = events.get()
            if event is None:
                break
            yield f"data: {json.dumps(event)}\n\n"
        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return StreamingResponse(
        emit(), media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


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
    from api.ask import ask, keyword_answer

    question = (question or "").strip()
    if not question:
        raise HTTPException(status_code=400, detail="question is required")

    retriever = _retriever()
    try:
        answer = ask(question, retriever, contract_id=contract_id or None)
        payload = answer.to_dict()
        payload["degraded"] = False
    except Exception as exc:                           # noqa: BLE001
        # Retrieval is local and unaffected by a model outage, so fall back to
        # showing the matching passages rather than failing the request. The
        # user loses the summary, not the search.
        HEALTH.failed(exc)
        answer = keyword_answer(question, retriever, contract_id or None,
                                reason=str(exc)[:160])
        payload = answer.to_dict()
        payload["degraded"] = True

    payload["model"] = HEALTH.snapshot()
    _store.audit(ACTOR, "ask", contract_id, question[:200])
    return payload


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
