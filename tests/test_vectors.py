"""Vector layer. Inert without a key, so the suite never needs Pinecone."""

import pytest

from api.vectors import VectorIndex, enabled, reciprocal_rank_fusion


@pytest.fixture(autouse=True)
def no_key(monkeypatch):
    monkeypatch.delenv("PINECONE_API_KEY", raising=False)


def test_disabled_without_a_key():
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
    assert "PINECONE_API_KEY" in result["reason"]


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
