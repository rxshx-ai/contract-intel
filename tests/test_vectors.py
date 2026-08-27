"""Vector layer. Inert without a key, so the suite never needs Pinecone."""

import pytest

from api.vectors import VectorIndex, enabled, reciprocal_rank_fusion


@pytest.fixture(autouse=True)
def no_backend(monkeypatch):
    """Default these tests to no vector backend.

    Without this they pick up the LOCAL backend whenever Ollama happens to be
    running, so the suite would pass or fail depending on what is installed on
    the machine. Tests that want a backend opt in explicitly.
    """
    monkeypatch.delenv("PINECONE_API_KEY", raising=False)
    monkeypatch.setenv("VECTOR_BACKEND", "none")


def test_disabled_when_no_backend_is_configured():
    assert enabled() is False
    index = VectorIndex()
    assert index.available is False
    assert index.stats()["enabled"] is False


def test_every_operation_is_a_no_op_when_disabled():
    """Callers must never have to branch on availability."""
    index = VectorIndex()
    assert index.upsert([{"id": "a", "text": "hello"}]) == 0
    assert index.search("anything") == []
    index.delete_contract("k1")          # must not raise


def test_retrieval_falls_back_to_lexical_when_vectors_are_off():
    from datetime import date

    from api import demo
    from api.pipeline import analyze_portfolio
    from api.rag import Retriever

    bundles = demo.load(date(2026, 8, 27))
    r = Retriever(bundles, analyze_portfolio(bundles), date(2026, 8, 27))
    assert r.vectors.available is False
    hits = r.search("liability cap", k=5)
    assert hits and hits[0].kind == "record"


def test_sync_reports_why_it_did_nothing():
    from datetime import date

    from api import demo
    from api.pipeline import analyze_portfolio
    from api.rag import Retriever

    bundles = demo.load(date(2026, 8, 27))
    r = Retriever(bundles, analyze_portfolio(bundles), date(2026, 8, 27))
    result = r.sync_vectors()
    assert result["enabled"] is False and result["synced"] == 0
    assert result["reason"]


def test_vector_payload_covers_records_and_passages():
    from datetime import date

    from api import demo
    from api.pipeline import analyze_portfolio
    from api.rag import Retriever

    bundles = demo.load(date(2026, 8, 27))
    r = Retriever(bundles, analyze_portfolio(bundles), date(2026, 8, 27))
    items = r.vector_payload()
    assert len(items) == r.stats["records"] + r.stats["passages"]
    for item in items:
        assert item["id"] and item["text"].strip()
        assert item["contract_id"] and item["kind"]


# ── fusion ───────────────────────────────────────────────────────────────

def test_rrf_rewards_agreement_between_the_two_rankings():
    fused = reciprocal_rank_fusion(["a", "b", "c"], ["c", "a", "z"])
    order = [k for k, _ in sorted(fused.items(), key=lambda kv: -kv[1])]
    assert order[0] == "a"          # top-3 in both
    assert order.index("c") < order.index("b")


def test_rrf_needs_no_score_calibration():
    """BM25 scores are unbounded, Pinecone's are bounded. RRF uses position
    only, so blending them requires no shared scale."""
    lexical = ["x", "y"]
    semantic = ["y", "x"]
    fused = reciprocal_rank_fusion(lexical, semantic)
    assert round(fused["x"], 6) == round(fused["y"], 6)


def test_items_in_one_ranking_only_still_score():
    fused = reciprocal_rank_fusion(["a"], ["b"])
    assert fused["a"] > 0 and fused["b"] > 0


# ── the hybrid path, exercised with a stub ───────────────────────────────
# Without a Pinecone key the fusion branch in Retriever.search never runs, so
# these drive it with a fake index. That leaves only the Pinecone HTTP call
# unverified, rather than the whole hybrid design.

class _StubVectors:
    """Stands in for Pinecone. Returns whatever ids the test asks for."""

    def __init__(self, ids, available=True):
        self.available = available
        self.ids = ids
        self.queries = []
        self.last_error = None

    def search(self, query, k=10, contract_id=None):
        from api.vectors import VectorHit

        self.queries.append((query, k, contract_id))
        return [VectorHit(id=i, score=1.0 - n * 0.01, metadata={})
                for n, i in enumerate(self.ids)]

    def stats(self):
        return {"enabled": self.available}


@pytest.fixture(scope="module")
def portfolio():
    from datetime import date

    from api import demo
    from api.pipeline import analyze_portfolio

    bundles = demo.load(date(2026, 8, 27))
    return bundles, analyze_portfolio(bundles), date(2026, 8, 27)


def _retriever(portfolio, stub):
    from api.rag import Retriever

    bundles, gaps, today = portfolio
    return Retriever(bundles, gaps, today, vectors=stub)


def test_fusion_pulls_in_a_hit_bm25_never_returned(portfolio):
    """The whole point of adding vectors: recall BM25 cannot reach."""
    from api.rag import Retriever

    bundles, gaps, today = portfolio
    lexical_only = Retriever(bundles, gaps, today)
    query = "what happens if they go bust"
    baseline = {h.id for h in lexical_only.search(query, k=8)}

    # A semantically related record that shares no words with the question:
    # termination wording, which is what "go bust" is really asking about.
    target = next(r.id for r in lexical_only.records
                  if "termination" in str(r.meta.get("clause_type", ""))
                  and r.id not in baseline)

    hybrid = _retriever(portfolio, _StubVectors([target]))
    fused = hybrid.search(query, k=8)
    assert target in {h.id for h in fused}


def test_fused_results_are_still_grounded(portfolio):
    bundles, _, _ = portfolio
    docs = {d.id: d for b in bundles for d in b.docs}
    target = next(p.id for b in [None] for p in
                  _retriever(portfolio, _StubVectors([])).passages)
    hybrid = _retriever(portfolio, _StubVectors([target]))
    for hit in hybrid.search("liability", k=8):
        if hit.kind == "passage":
            p = hit.payload
            assert docs[p.doc_id].text[p.start:p.end] == p.text


def test_ids_the_vector_store_invents_are_dropped(portfolio):
    """A stale or hallucinated vector id must not become a citation."""
    hybrid = _retriever(portfolio, _StubVectors(["ghost-id", "another-ghost"]))
    hits = hybrid.search("liability cap", k=6)
    assert hits                                  # lexical still answers
    assert all(h.id in hybrid.by_id for h in hits)


def test_contract_scope_is_passed_through_to_the_vector_store(portfolio):
    stub = _StubVectors([])
    hybrid = _retriever(portfolio, stub)
    hybrid.search("liability", k=5, contract_id="k_helios")
    assert stub.queries and stub.queries[-1][2] == "k_helios"


def test_include_filter_survives_fusion(portfolio):
    passage_id = _retriever(portfolio, _StubVectors([])).passages[0].id
    hybrid = _retriever(portfolio, _StubVectors([passage_id]))
    hits = hybrid.search("liability", k=8, include=("record",))
    assert hits and all(h.kind == "record" for h in hits)


def test_empty_vector_response_falls_back_to_lexical(portfolio):
    from api.rag import Retriever

    bundles, gaps, today = portfolio
    lexical = Retriever(bundles, gaps, today).search("liability cap", k=5)
    hybrid = _retriever(portfolio, _StubVectors([])).search("liability cap", k=5)
    assert [h.id for h in hybrid] == [h.id for h in lexical]


def test_agreement_between_both_rankings_wins(portfolio):
    """A hit both systems rank highly should outrank one only BM25 liked."""
    from api.rag import Retriever

    bundles, gaps, today = portfolio
    lexical = Retriever(bundles, gaps, today).search("liability cap", k=8)
    second = lexical[1].id
    hybrid = _retriever(portfolio, _StubVectors([second]))
    fused = hybrid.search("liability cap", k=8)
    assert fused[0].id == second


# ── local backend ────────────────────────────────────────────────────────

def _local(tmp_path, monkeypatch, vectors_by_text=None):
    """A LocalBackend whose embedding call is replaced, so these tests need
    neither Ollama nor a network."""
    from api import vectors as vec

    monkeypatch.setattr(vec, "LOCAL_STORE", tmp_path)
    backend = vec.LocalBackend.__new__(vec.LocalBackend)
    backend.namespace = "t"
    backend.model = "fake"
    backend.url = "http://fake"
    backend.last_error = None
    backend.ids, backend.meta, backend._matrix = [], [], None
    backend.path = tmp_path / "t.json"
    backend.available = True

    table = vectors_by_text or {}

    def embed(texts):
        out = []
        for t in texts:
            out.append(table.get(t, [float(len(t) % 7), 1.0, 0.0]))
        return out

    backend.embed = embed
    return backend


def test_local_backend_upserts_and_searches(tmp_path, monkeypatch):
    table = {"alpha": [1.0, 0.0, 0.0], "beta": [0.0, 1.0, 0.0],
             "find alpha": [0.99, 0.1, 0.0]}
    b = _local(tmp_path, monkeypatch, table)
    assert b.upsert([{"id": "a", "text": "alpha", "contract_id": "k1"},
                     {"id": "b", "text": "beta", "contract_id": "k2"}]) == 2
    hits = b.search("find alpha", k=2)
    assert hits[0].id == "a"
    assert hits[0].score > hits[1].score


def test_local_backend_upsert_is_idempotent(tmp_path, monkeypatch):
    b = _local(tmp_path, monkeypatch)
    item = {"id": "a", "text": "alpha", "contract_id": "k1"}
    b.upsert([item]); b.upsert([item])
    assert len(b.ids) == 1
    assert b.stats()["vectors"] == 1


def test_local_backend_filters_by_contract(tmp_path, monkeypatch):
    b = _local(tmp_path, monkeypatch)
    b.upsert([{"id": "a", "text": "alpha", "contract_id": "k1"},
              {"id": "b", "text": "beta", "contract_id": "k2"}])
    hits = b.search("anything", k=5, contract_id="k2")
    assert [h.id for h in hits] == ["b"]


def test_local_backend_deletes_a_contract(tmp_path, monkeypatch):
    b = _local(tmp_path, monkeypatch)
    b.upsert([{"id": "a", "text": "alpha", "contract_id": "k1"},
              {"id": "b", "text": "beta", "contract_id": "k2"}])
    b.delete_contract("k1")
    assert b.ids == ["b"]
    assert [h.id for h in b.search("anything", k=5)] == ["b"]


def test_local_backend_persists_across_instances(tmp_path, monkeypatch):
    from api import vectors as vec

    b = _local(tmp_path, monkeypatch)
    b.upsert([{"id": "a", "text": "alpha", "contract_id": "k1"}])

    monkeypatch.setattr(vec, "LOCAL_STORE", tmp_path)
    reopened = vec.LocalBackend.__new__(vec.LocalBackend)
    reopened.namespace = "t"; reopened.path = tmp_path / "t.json"
    reopened.ids, reopened.meta, reopened._matrix = [], [], None
    reopened.last_error = None
    reopened._load()
    assert reopened.ids == ["a"]


# ── backend selection ────────────────────────────────────────────────────

def test_explicit_none_disables(monkeypatch):
    from api.vectors import choose_backend

    monkeypatch.setenv("VECTOR_BACKEND", "none")
    assert choose_backend().available is False


def test_pinecone_is_preferred_when_a_key_exists(monkeypatch):
    from api.vectors import choose_backend

    monkeypatch.setenv("VECTOR_BACKEND", "auto")
    monkeypatch.setenv("PINECONE_API_KEY", "pc-test")
    assert choose_backend().name == "pinecone"


def test_falls_back_to_local_then_none(monkeypatch):
    from api import vectors as vec

    monkeypatch.setenv("VECTOR_BACKEND", "auto")
    monkeypatch.delenv("PINECONE_API_KEY", raising=False)

    class Dead(vec.LocalBackend):
        def __init__(self, namespace="default", **kw):
            self.namespace = namespace; self.last_error = "ollama down"
            self.available = False

    monkeypatch.setattr(vec, "LocalBackend", Dead)
    assert vec.choose_backend().name == "none"


# ── pinecone response parsing (no network) ───────────────────────────────

def test_pinecone_hits_are_parsed_from_the_documented_shape():
    """Guards the shape I got wrong first time: hits expose id/score/fields,
    not _id/_score, and live under response.result.hits."""
    from api.vectors import PineconeBackend, _hits, _attr

    class Hit:
        def __init__(self, i, s): self.id, self.score, self.fields = i, s, {"k": 1}

    class Result:
        hits = [Hit("a", 0.9), Hit("b", 0.5)]

    class Response:
        result = Result()

    parsed = _hits(Response())
    assert [_attr(h, "id") for h in parsed] == ["a", "b"]
    assert _attr(parsed[0], "score") == 0.9
