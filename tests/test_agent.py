"""The agent plans and calls tools. Scripted client: no network."""

import json
from datetime import date

import pytest

from api import agent as agent_mod
from api.agent import Agent
from api.pipeline import analyze_portfolio
from api.rag import Retriever

TODAY = date(2026, 8, 27)


class _Call:
    def __init__(self, name, args, cid="c1"):
        self.id = cid
        self.type = "function"
        self.function = type("F", (), {"name": name,
                                       "arguments": json.dumps(args)})()


class _Msg:
    def __init__(self, tool_calls=None, content=""):
        self.tool_calls = tool_calls
        self.content = content


class _Resp:
    def __init__(self, msg):
        self.choices = [type("C", (), {"message": msg, "finish_reason": "stop"})()]


class _FakeClient:
    """Replays a script of assistant turns."""

    def __init__(self, script):
        self.script = list(script)
        self.seen = []
        self.chat = type("Chat", (), {"completions": self})()

    def create(self, **kwargs):
        # Snapshot: the agent appends to the same list after this returns.
        self.seen.append({**kwargs, "messages": list(kwargs["messages"])})
        return _Resp(self.script.pop(0))


@pytest.fixture(scope="module")
def state():
    from api import demo

    bundles = demo.load(TODAY)
    gaps = analyze_portfolio(bundles)
    return bundles, gaps, Retriever(bundles, gaps, TODAY)


@pytest.fixture(autouse=True)
def no_throttle(monkeypatch):
    """The TPM budget sleeps to respect a rate limit. These tests spend no
    tokens, so waiting on it would just make the suite take minutes."""
    monkeypatch.setattr(agent_mod.BUDGET, "reserve", lambda *a, **k: 0.0)


@pytest.fixture
def make_agent(state, monkeypatch):
    bundles, gaps, retriever = state

    def build(script, plan=("step one", "step two")):
        fake = _FakeClient(script)
        monkeypatch.setattr(agent_mod, "client", lambda: fake)
        monkeypatch.setattr(Agent, "make_plan", lambda self, q: list(plan))
        return Agent(bundles, gaps, retriever, TODAY), fake

    return build


def _cap_id(state):
    bundles, _, _ = state
    nw = next(b for b in bundles if b.contract.id == "k_northwind")
    return next(c.id for c in nw.claims
                if c.clause_type.value == "limitation_of_liability" and c.effective)


# ── planning ─────────────────────────────────────────────────────────────

def test_the_plan_is_made_before_any_tool_runs(make_agent):
    agent, _ = make_agent([_Msg([_Call("finish", {"answer": "x", "cited_ids": [],
                                                  "sufficient": True})])])
    result = agent.run("anything")
    assert result.plan == ["step one", "step two"]


def test_plan_is_shown_to_the_model(make_agent):
    agent, fake = make_agent([_Msg([_Call("finish", {"answer": "x",
                                                     "cited_ids": [], "sufficient": True})])])
    agent.run("anything")
    user_msg = fake.seen[0]["messages"][1]["content"]
    assert "Your plan:" in user_msg and "step one" in user_msg


# ── tools ────────────────────────────────────────────────────────────────

def test_compare_tool_runs_and_is_returned_as_a_table(make_agent):
    agent, _ = make_agent([
        _Msg([_Call("compare", {"dimension": "liability_cap", "contract_ids": None})]),
        _Msg([_Call("finish", {"answer": "Northwind is weakest.",
                               "cited_ids": [], "sufficient": True})]),
    ])
    result = agent.run("which supplier has the weakest cap?")
    assert [s.tool for s in result.steps] == ["compare", "finish"]
    assert result.tables and result.tables[0]["dimension"] == "liability_cap"


def test_citations_resolve_to_real_records(make_agent, state):
    cap = _cap_id(state)
    agent, _ = make_agent([
        _Msg([_Call("finish", {"answer": "The cap is 250,000.",
                               "cited_ids": [cap], "sufficient": True})]),
    ])
    result = agent.run("what is the cap?")
    assert len(result.citations) == 1
    assert result.citations[0]["quote"]


def test_fabricated_citation_ids_are_dropped(make_agent):
    agent, _ = make_agent([
        _Msg([_Call("finish", {"answer": "x", "cited_ids": ["not-a-real-id"],
                               "sufficient": True})]),
    ])
    assert agent.run("q").citations == []


def test_a_bare_contract_id_resolves_to_its_facts(make_agent):
    """The model reaches for the contract id naturally; losing a real citation
    over a naming detail helps nobody."""
    agent, _ = make_agent([
        _Msg([_Call("finish", {"answer": "x", "cited_ids": ["k_northwind"],
                               "sufficient": True})]),
    ])
    citations = agent.run("q").citations
    assert len(citations) == 1
    assert citations[0]["contract"].startswith("Northwind")


def test_repeated_tool_calls_reuse_the_earlier_result(make_agent):
    args = {"dimension": "uptime", "contract_ids": None}
    agent, fake = make_agent([
        _Msg([_Call("compare", args)]),
        _Msg([_Call("compare", args)]),
        _Msg([_Call("finish", {"answer": "x", "cited_ids": [], "sufficient": True})]),
    ])
    result = agent.run("compare uptime")
    assert result.steps[1].summary.startswith("repeat")
    tool_msgs = [m for m in fake.seen[2]["messages"] if m.get("role") == "tool"]
    assert json.loads(tool_msgs[-1]["content"])["repeat_of_earlier_call"] is True


def test_tool_failure_is_reported_not_fatal(make_agent):
    agent, _ = make_agent([
        _Msg([_Call("exit_cost", {"contract_id": "nope", "exit_date": "bad-date"})]),
        _Msg([_Call("finish", {"answer": "could not compute",
                               "cited_ids": [], "sufficient": False})]),
    ])
    result = agent.run("exit cost?")
    assert result.steps[0].tool == "exit_cost"
    assert result.sufficient is False


# ── termination ──────────────────────────────────────────────────────────

def test_running_out_of_steps_is_admitted(make_agent):
    search = _Msg([_Call("search", {"query": "liability", "contract_id": None})])
    agent, _ = make_agent([search] * 6)
    result = agent.run("q", max_steps=3)
    assert result.stopped_early is True
    assert result.sufficient is False
    assert "ran out of steps" in result.answer


def test_prose_answer_without_finish_still_recovers_citations(make_agent, state):
    cap = _cap_id(state)
    agent, _ = make_agent([
        _Msg(None, content=f"The cap is 250,000 ({cap}).\nCited ids: {cap}\n"
                           f"Sufficient: true"),
    ])
    result = agent.run("what is the cap?")
    assert len(result.citations) == 1
    assert "Cited ids" not in result.answer      # boilerplate stripped
    assert "Sufficient" not in result.answer


def test_final_step_is_told_to_finish(make_agent):
    agent, fake = make_agent([
        _Msg([_Call("search", {"query": "x", "contract_id": None})]),
        _Msg([_Call("finish", {"answer": "y", "cited_ids": [], "sufficient": True})]),
    ])
    agent.run("q", max_steps=2)
    nudges = [m for m in fake.seen[1]["messages"]
              if "Call finish now" in str(m.get("content") or "")]
    assert nudges, "the last step must be told to finish"


# ── the malformed-tool-name quirk ────────────────────────────────────────

def test_a_finish_emitted_under_the_wrong_tool_name_is_salvaged(state,
                                                                monkeypatch):
    """gpt-oss sometimes routes a call through an internal channel name and
    Groq rejects the request. The arguments are still ours."""
    bundles, gaps, retriever = state
    cap = _cap_id(state)
    body = {"error": {"code": "tool_use_failed", "failed_generation": json.dumps({
        "name": "commentary",
        "arguments": {"answer": "The cap is 250,000.", "cited_ids": [cap],
                      "sufficient": True}})}}

    class Boom(Exception):
        def __init__(self):
            self.body = body

    class Exploding:
        def __init__(self):
            self.chat = type("C", (), {"completions": self})()

        def create(self, **kw):
            raise Boom()

    monkeypatch.setattr(agent_mod, "client", lambda: Exploding())
    monkeypatch.setattr(Agent, "make_plan", lambda self, q: [])
    result = Agent(bundles, gaps, retriever, TODAY).run("what is the cap?")
    assert result.answer == "The cap is 250,000."
    assert len(result.citations) == 1


def test_an_unrelated_error_is_not_swallowed(state, monkeypatch):
    bundles, gaps, retriever = state

    class Exploding:
        def __init__(self):
            self.chat = type("C", (), {"completions": self})()

        def create(self, **kw):
            raise RuntimeError("network down")

    monkeypatch.setattr(agent_mod, "client", lambda: Exploding())
    monkeypatch.setattr(Agent, "make_plan", lambda self, q: [])
    with pytest.raises(RuntimeError):
        Agent(bundles, gaps, retriever, TODAY).run("q")
