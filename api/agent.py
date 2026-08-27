"""A planning, tool-using agent over the verified layer.

The agent decides WHAT to look up; it never decides what is true. Every tool
returns data computed or verified by code, and the final answer cites record
ids that must exist. The same guarantee as Ask, extended over several steps:

  * `plan` must be called first, so the reasoning is visible rather than
    implied. The plan is shown to the user alongside the answer.
  * `search` is the RAG layer (records + verbatim passages).
  * `compare`, `deadlines`, `calendar`, `contract_facts`, `exit_cost` are
    deterministic Python. The model chooses arguments, not answers.
  * `finish` ends the run with an answer and citation ids. Ids that do not
    resolve are dropped, so a fabricated citation cannot be displayed.

Step count is capped, and every call goes through the shared TPM budget --
free-tier limits make an unbounded agent loop a way to hang the demo.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date
from typing import Any

from pydantic import BaseModel

from api.llm import (BUDGET, MODEL, ExtractionUnavailable, client,
                     complete_json)
from api.chunking import estimate_tokens

MAX_STEPS = 5
MAX_TOOL_CHARS = 1800


@dataclass
class Step:
    n: int
    tool: str
    args: dict[str, Any]
    summary: str
    ok: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {"n": self.n, "tool": self.tool, "args": self.args,
                "summary": self.summary, "ok": self.ok}


@dataclass
class AgentResult:
    question: str
    plan: list[str] = field(default_factory=list)
    steps: list[Step] = field(default_factory=list)
    answer: str = ""
    citations: list[dict[str, Any]] = field(default_factory=list)
    tables: list[dict[str, Any]] = field(default_factory=list)
    sufficient: bool = True
    stopped_early: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "plan": self.plan,
            "steps": [s.to_dict() for s in self.steps],
            "answer": self.answer,
            "citations": self.citations,
            "tables": self.tables,
            "sufficient": self.sufficient,
            "stopped_early": self.stopped_early,
        }


# --------------------------------------------------------------------------
# tool surface
# --------------------------------------------------------------------------

def _fn(name: str, description: str, properties: dict, required: list[str]) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
                "additionalProperties": False,
            },
        },
    }


def tool_specs(dimensions: list[str]) -> list[dict]:
    s = {"type": "string"}
    ns = {"type": ["string", "null"]}
    ni = {"type": ["integer", "null"]}
    return [
        _fn("search",
            "Search verified records (clauses, deadlines, findings, missing "
            "wording, cross-contract gaps) and verbatim document passages.",
            {"query": s, "contract_id": ns}, ["query", "contract_id"]),
        _fn("compare",
            "Compare one dimension across contracts. Returns a table with the "
            "wording behind every figure. Use this for any 'which/compare' "
            "question rather than searching each contract separately.",
            {"dimension": {"type": "string", "enum": dimensions},
             "contract_ids": {"type": ["array", "null"], "items": s}},
            ["dimension", "contract_ids"]),
        _fn("deadlines",
            "Upcoming computed deadlines, soonest first.",
            {"within_days": ni, "contract_id": ns}, ["within_days", "contract_id"]),
        _fn("calendar",
            "Calendar events between two ISO dates: when documents arrived, "
            "computed deadlines, and dates written in the text.",
            {"start": ns, "end": ns, "contract_id": ns}, ["start", "end", "contract_id"]),
        _fn("contract_facts",
            "List the contracts, or the key facts of one.", {"contract_id": ns},
            ["contract_id"]),
        _fn("exit_cost",
            "Itemised cost of leaving a contract on a date (YYYY-MM-DD).",
            {"contract_id": s, "exit_date": s}, ["contract_id", "exit_date"]),
        _fn("finish",
            "End the run. Give the answer and cite the record ids you used.",
            {"answer": s,
             "cited_ids": {"type": "array", "items": s},
             "sufficient": {"type": "boolean"}},
            ["answer", "cited_ids", "sufficient"]),
    ]


SYSTEM = """You answer questions about a company's contracts.

You do not read contracts. You call tools that return facts already extracted
and verified against the source documents, then you assemble an answer.

HOW TO WORK
1. Your plan is already made and is shown to you. Follow it.
2. For any question comparing contracts or asking which is best/worst, call
   `compare`. It returns a computed table. Do not compare by searching each
   contract separately -- you will miss the ones that say nothing.
3. Use `search` for wording and for anything `compare` has no dimension for.
4. Call `finish` as soon as you can answer. Always end with `finish`.
5. Never repeat a lookup you have already made with the same arguments. If a
   result was empty, that IS the answer to that part -- say so and move on.

RULES
- Use only what the tools returned. Never add knowledge about how contracts
  usually work. If the tools do not answer it, call `finish` with
  sufficient=false and say what is missing.
- Numbers and dates in tool results were computed by code. Repeat them exactly.
- A contract that does not state something is not the same as a good value.
  Say when a contract is silent; `compare` reports this as `not_stated`.
- Cite the record ids (`record_id`, or `id` on search hits) behind every claim.
- Be specific and brief: two to four sentences. Name the contract for each fact.
- Plain sentences. No markdown, no bold, no bullet lists, and never write the
  cited ids or "sufficient" into the answer text -- they belong in the `finish`
  arguments, not in the prose.
- You are decision support, not a lawyer. Say what the documents say; do not
  advise whether to sign and do not opine on enforceability.
"""


# --------------------------------------------------------------------------

class _Plan(BaseModel):
    steps: list[str] | None = None


PLAN_SYSTEM = """Plan how to answer a question about a company's contracts.

Available lookups: compare (one dimension across contracts, computed),
search (verified records and verbatim passages), deadlines, calendar,
contract_facts, exit_cost.

Give 2 to 4 short steps, each naming the lookup you would use. Do not answer
the question. Prefer `compare` for anything asking which contract is
best/worst or comparing a term across contracts.
"""


class Agent:
    def __init__(self, bundles, gaps, retriever, today: date):
        self.bundles = bundles
        self.gaps = gaps
        self.retriever = retriever
        self.today = today
        self.by_id = dict(retriever.by_id)

    # ---- tool implementations -----------------------------------------

    def _tool_search(self, query: str, contract_id: str | None = None):
        hits = self.retriever.search(query or "", k=6, contract_id=contract_id)
        for hit in hits:
            self.by_id.setdefault(hit.id, hit.payload)
        return {"hits": [h.for_model() for h in hits]}

    def _tool_compare(self, dimension: str, contract_ids=None):
        from api.compare import compare

        result = compare(self.bundles, dimension, contract_ids)
        if result.get("ok"):
            for row in result["rows"]:
                item = self.by_id.get(row["record_id"])
                if item is None:
                    self.by_id[row["record_id"]] = _RowCitation(row, dimension)
            result = {
                **result,
                "rows": [{k: v for k, v in row.items()
                          if k in ("record_id", "contract", "we_are", "display",
                                   "value", "quote")}
                         for row in result["rows"]],
            }
        return result

    def _tool_deadlines(self, within_days: int | None = None,
                        contract_id: str | None = None):
        from api.pipeline import upcoming_deadlines

        rows = upcoming_deadlines(self.bundles, self.today, within_days or 365)
        if contract_id:
            rows = [r for r in rows if r["contract_id"] == contract_id]
        return {"deadlines": [
            {"contract": r["contract"], "kind": r["kind"], "due": r["due_date"],
             "days": r["days_remaining"], "what": r["description"][:150],
             "anchor": r["anchor"]}
            for r in rows[:12]]}

    def _tool_calendar(self, start=None, end=None, contract_id=None):
        from api.calendar import build_calendar, in_range

        events = build_calendar(self.bundles, self.today)
        lo = date.fromisoformat(start) if start else None
        hi = date.fromisoformat(end) if end else None
        events = in_range(events, lo, hi)
        if contract_id:
            events = [e for e in events if e.contract_id == contract_id]
        return {"events": [
            {"date": e.date.isoformat(), "kind": e.kind, "source": e.source,
             "contract": e.contract, "title": e.title, "days": e.days_from(self.today)}
            for e in events[:20]]}

    def _tool_contract_facts(self, contract_id: str | None = None):
        out = []
        for bundle in self.bundles:
            c = bundle.contract
            if contract_id and c.id != contract_id:
                continue
            profile = bundle.result().risk
            out.append({
                "contract_id": c.id, "party": c.counterparty or c.title,
                "we_are": c.our_role.value, "type": c.contract_type.value,
                "annual_value": c.annual_value,
                "effective": c.effective_date.isoformat() if c.effective_date else None,
                "attention_score": profile.overall if profile else 0,
                "documents": [d.filename for d in bundle.docs],
                "findings": len(bundle.findings),
            })
        return {"contracts": out}

    def _tool_exit_cost(self, contract_id: str, exit_date: str):
        from api.findings.termination import termination_cost

        bundle = next((b for b in self.bundles if b.contract.id == contract_id), None)
        if bundle is None:
            return {"error": f"unknown contract {contract_id}"}
        try:
            when = date.fromisoformat(exit_date)
        except ValueError:
            return {"error": "exit_date must be YYYY-MM-DD"}
        cost = termination_cost(bundle.contract, bundle.claims, bundle.obligations,
                                exit_date=when, today=self.today)
        return {"total": cost.total, "currency": cost.currency,
                "lines": [{"label": i["label"], "amount": i["amount"]}
                          for i in cost.line_items],
                "notes": cost.notes[:4]}

    # ---- the loop ------------------------------------------------------

    def make_plan(self, question: str) -> list[str]:
        try:
            plan = complete_json(PLAN_SYSTEM, f"Question: {question}", _Plan,
                                 schema_name="agent_plan", max_tokens=400)
            return [str(s).strip() for s in (plan.steps or []) if str(s).strip()][:4]
        except Exception:
            return []

    def run(self, question: str, max_steps: int = MAX_STEPS) -> AgentResult:
        from api.compare import DIMENSIONS

        result = AgentResult(question=question)
        result.plan = self.make_plan(question)
        tools = tool_specs(sorted(DIMENSIONS))
        plan_text = ("\n".join(f"{i}. {s}" for i, s in enumerate(result.plan, 1))
                     or "1. Look up what is needed. 2. Answer.")
        messages = [{"role": "system", "content": SYSTEM},
                    {"role": "user",
                     "content": f"Today is {self.today.isoformat()}.\n"
                                f"Question: {question}\n\nYour plan:\n{plan_text}"}]
        handlers = {
            "search": self._tool_search,
            "compare": self._tool_compare,
            "deadlines": self._tool_deadlines,
            "calendar": self._tool_calendar,
            "contract_facts": self._tool_contract_facts,
            "exit_cost": self._tool_exit_cost,
        }
        groq = client()
        already: dict[str, Any] = {}

        for step_no in range(1, max_steps + 1):
            # Force the plan on the first turn and the finish on the last, so a
            # run always has a visible plan and always ends with resolvable
            # citations rather than ids written into prose.
            if step_no == max_steps:
                messages.append({
                    "role": "user",
                    "content": "You have no lookups left. Call finish now with "
                               "your answer and the record ids you used.",
                })

            prompt_tokens = sum(estimate_tokens(str(m.get("content") or ""))
                                for m in messages) + 500
            BUDGET.reserve(prompt_tokens + 800)
            try:
                response = groq.chat.completions.create(
                    model=MODEL, messages=messages, tools=tools, tool_choice="auto",
                    temperature=0.0, max_tokens=800,
                )
            except Exception as exc:
                salvaged = _salvage_finish(exc)
                if salvaged is None:
                    raise
                result.answer = _strip_meta(str(salvaged.get("answer") or ""))
                result.citations = self._resolve(salvaged.get("cited_ids") or [])
                result.sufficient = bool(salvaged.get("sufficient", True))
                result.steps.append(Step(len(result.steps) + 1, "finish", {},
                                         "recovered from a malformed tool call"))
                return result
            message = response.choices[0].message
            calls = message.tool_calls or []

            if not calls:
                # The model answered in prose instead of calling finish. Keep the
                # answer, but recover citations by matching real ids mentioned in
                # the text -- never by trusting the text itself.
                text = (message.content or "").strip()
                result.answer = _strip_meta(text)
                result.citations = self._resolve(_ids_in(text, self.by_id))
                result.sufficient = bool(result.answer)
                result.steps.append(Step(len(result.steps) + 1, "answer", {},
                                         "answered without calling finish"))
                return result

            messages.append({
                "role": "assistant", "content": message.content or "",
                "tool_calls": [
                    {"id": c.id, "type": "function",
                     "function": {"name": c.function.name,
                                  "arguments": c.function.arguments}}
                    for c in calls],
            })

            for call in calls:
                name = call.function.name
                try:
                    args = json.loads(call.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}
                args = {k: v for k, v in args.items() if v is not None}

                payload: Any = {"ok": True}
                if name == "finish":
                    result.answer = str(args.get("answer") or "").strip()
                    result.sufficient = bool(args.get("sufficient", True))
                    result.citations = self._resolve(args.get("cited_ids") or [])
                    result.steps.append(Step(len(result.steps) + 1, "finish", {},
                                             f"{len(result.citations)} citations"))
                    return result
                elif name in handlers:
                    signature = name + json.dumps(args, sort_keys=True)
                    if signature in already:
                        # Repeats burn a step and, on a rate-limited tier, a
                        # minute of wall clock. Hand back the earlier result.
                        payload = {"repeat_of_earlier_call": True,
                                   "result": already[signature]}
                        result.steps.append(Step(len(result.steps) + 1, name, args,
                                                 "repeat — reused earlier result"))
                        messages.append({
                            "role": "tool", "tool_call_id": call.id, "name": name,
                            "content": json.dumps(payload)[:MAX_TOOL_CHARS]})
                        continue
                    try:
                        payload = handlers[name](**args)
                        already[signature] = payload
                        summary = _summarize(name, payload)
                        ok = True
                    except Exception as exc:               # tool bug, not model
                        payload = {"error": str(exc)[:200]}
                        summary = f"failed: {str(exc)[:80]}"
                        ok = False
                    result.steps.append(
                        Step(len(result.steps) + 1, name, args, summary, ok))
                    if name == "compare" and payload.get("ok"):
                        result.tables.append(payload)
                else:
                    payload = {"error": f"unknown tool {name}"}
                    result.steps.append(Step(len(result.steps) + 1, name, args,
                                             "unknown tool", False))

                messages.append({
                    "role": "tool", "tool_call_id": call.id, "name": name,
                    "content": json.dumps(payload)[:MAX_TOOL_CHARS],
                })

        result.stopped_early = True
        result.answer = result.answer or (
            "I ran out of steps before reaching an answer. The plan and what I "
            "looked at are shown below."
        )
        result.sufficient = False
        return result

    def _resolve(self, ids: list[str]) -> list[dict[str, Any]]:
        """Ids to citations. Unknown ids are dropped, never invented.

        A bare contract id is accepted and resolved to that contract's facts
        record -- the model reaches for it naturally, and refusing would lose a
        real citation over a naming detail rather than over truthfulness.
        """
        out: list[dict[str, Any]] = []
        seen: set[str] = set()
        for rid in ids:
            item = self.by_id.get(rid) or self.by_id.get(f"{rid}:fact")
            if item is None or not hasattr(item, "citation"):
                continue
            citation = item.citation()
            key = citation.get("record_id", "")
            if key in seen:
                continue
            seen.add(key)
            out.append(citation)
        return out


class _RowCitation:
    """Makes a comparison row citable in the same shape as everything else."""

    def __init__(self, row: dict[str, Any], dimension: str):
        self.row = row
        self.dimension = dimension

    def citation(self) -> dict[str, Any]:
        return {
            "record_id": self.row["record_id"],
            "contract_id": self.row.get("contract_id", ""),
            "contract": self.row["contract"],
            "kind": "comparison",
            "title": f"{self.dimension}: {self.row['display']}",
            "quote": self.row.get("quote"),
            "file": self.row.get("file"),
            "start": self.row.get("start"),
            "end": self.row.get("end"),
        }


_META_LINE = None


def _strip_meta(text: str) -> str:
    """Remove the citation/sufficiency boilerplate some runs write into prose."""
    import re

    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    lines = [l for l in text.splitlines()
             if not re.match(r"^\s*(cited ids?|sufficient)\s*[:.]", l, re.I)]
    return "\n".join(lines).strip()


def _ids_in(text: str, known: dict[str, Any]) -> list[str]:
    return [rid for rid in known if rid and rid in text]


def _salvage_finish(exc: Exception) -> dict[str, Any] | None:
    """Recover a finish call the model emitted under the wrong tool name.

    gpt-oss intermittently routes a tool call through an internal channel name
    ('commentary', 'json') that is not in our tool list, and Groq rejects the
    whole request. The arguments are still exactly what we asked for, so
    discarding a completed run would be throwing away a correct answer over a
    naming quirk. Only salvaged when it actually carries an answer.
    """
    body = getattr(exc, "body", None)
    if not isinstance(body, dict):
        try:
            body = exc.response.json()          # type: ignore[attr-defined]
        except Exception:
            return None
    error = body.get("error") if isinstance(body, dict) else None
    if not isinstance(error, dict) or error.get("code") != "tool_use_failed":
        return None
    raw = error.get("failed_generation")
    if not raw:
        return None
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return None
    args = data.get("arguments") if isinstance(data, dict) else None
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            return None
    if isinstance(args, dict) and str(args.get("answer") or "").strip():
        return args
    return None


def _summarize(name: str, payload: dict[str, Any]) -> str:
    if name == "search":
        return f"{len(payload.get('hits', []))} hits"
    if name == "compare":
        if not payload.get("ok"):
            return payload.get("error", "failed")
        return (f"{len(payload.get('rows', []))} contracts compared, "
                f"{len(payload.get('not_stated', []))} silent")
    if name == "deadlines":
        return f"{len(payload.get('deadlines', []))} deadlines"
    if name == "calendar":
        return f"{len(payload.get('events', []))} events"
    if name == "contract_facts":
        return f"{len(payload.get('contracts', []))} contracts"
    if name == "exit_cost":
        return f"total {payload.get('total')}"
    return "ok"
