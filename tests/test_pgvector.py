"""Vectors stored in Postgres via pgvector.

Runs against a real database when DATABASE_URL points at one with the `vector`
extension available; skips otherwise, so the suite stays offline by default.
"""

import os
from datetime import date

import pytest

from api.vectors import OllamaEmbedder, PgVectorBackend, VectorIndex

DATABASE_URL = os.environ.get("DATABASE_URL", "").strip()


class _FakeEmbedder:
    """Deterministic 4-dim vectors: no Ollama needed, and similarity is
    predictable enough to assert ordering."""

    available = True
    last_error = None
    model = "fake-embed"
    dimension = 4

    TABLE = {
        "alpha": [1.0, 0.0, 0.0, 0.0],
        "beta": [0.0, 1.0, 0.0, 0.0],
        "gamma": [0.0, 0.0, 1.0, 0.0],
        "find alpha": [0.99, 0.10, 0.0, 0.0],
    }

    def embed(self, texts):
        return [self.TABLE.get(t, [0.5, 0.5, 0.5, 0.5]) for t in texts]


TEST_TABLE = "embeddings_test"


def _backend(namespace, embedder=None):
    """A backend on a dedicated table.

    The production table is dimension-locked to the real 768-dim embedder, so
    tests using a 4-dim fake need their own table rather than fighting over it.
    """
    backend = PgVectorBackend(namespace, url=DATABASE_URL,
                              embedder=embedder or _FakeEmbedder(),
                              table=TEST_TABLE)
    if not backend.available:
        pytest.skip(f"pgvector unavailable: {backend.last_error}")
    with backend.conn.cursor() as cur:
        cur.execute(f"DELETE FROM {TEST_TABLE} WHERE namespace = %s", (namespace,))
    return backend


pytestmark = pytest.mark.skipif(
    not DATABASE_URL, reason="set DATABASE_URL to a Postgres with pgvector")


# ── storage ──────────────────────────────────────────────────────────────

def test_extension_and_table_are_created_on_connect():
    b = _backend("t_setup")
    stats = b.stats()
    assert stats["enabled"] and stats["backend"] == "pgvector"
    assert stats["dimension"] == 4


def test_upsert_then_search_ranks_by_cosine():
    b = _backend("t_search")
    assert b.upsert([
        {"id": "a", "text": "alpha", "contract_id": "k1", "kind": "clause"},
        {"id": "b", "text": "beta", "contract_id": "k2", "kind": "clause"},
        {"id": "c", "text": "gamma", "contract_id": "k1", "kind": "passage"},
    ]) == 3
    hits = b.search("find alpha", k=3)
    assert hits[0].id == "a"
    assert hits[0].score > hits[1].score
    assert 0.0 <= hits[0].score <= 1.0        # similarity, not distance


def test_upsert_is_idempotent():
    b = _backend("t_idem")
    item = {"id": "a", "text": "alpha", "contract_id": "k1"}
    b.upsert([item]); b.upsert([item])
    assert b.stats()["vectors"] == 1


def test_namespaces_do_not_leak():
    a = _backend("t_ns_a")
    other = _backend("t_ns_b")
    a.upsert([{"id": "a", "text": "alpha", "contract_id": "k1"}])
    assert a.stats()["vectors"] == 1
    assert other.stats()["vectors"] == 0
    assert other.search("find alpha", k=5) == []


def test_search_can_be_scoped_to_one_contract():
    b = _backend("t_scope")
    b.upsert([{"id": "a", "text": "alpha", "contract_id": "k1"},
              {"id": "b", "text": "beta", "contract_id": "k2"}])
    hits = b.search("find alpha", k=5, contract_id="k2")
    assert [h.id for h in hits] == ["b"]


def test_deleting_a_contract_removes_its_vectors():
    """Vectors live in the same database as the analysis, so they cannot be
    orphaned by a delete that succeeded in one store and not the other."""
    b = _backend("t_delete")
    b.upsert([{"id": "a", "text": "alpha", "contract_id": "k1"},
              {"id": "b", "text": "beta", "contract_id": "k2"}])
    b.delete_contract("k1")
    assert b.stats()["vectors"] == 1
    assert [h.id for h in b.search("find alpha", k=5)] == ["b"]


def test_metadata_is_stored_for_inspection():
    b = _backend("t_meta")
    b.upsert([{"id": "a", "text": "alpha", "contract_id": "k1",
               "kind": "clause", "contract": "Northwind", "file": "msa.txt"}])
    hit = b.search("find alpha", k=1)[0]
    assert hit.metadata["contract"] == "Northwind"
    assert hit.metadata["file"] == "msa.txt"


# ── configuration ────────────────────────────────────────────────────────

def test_missing_database_url_is_reported_not_raised(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    b = PgVectorBackend("t", url=None, embedder=_FakeEmbedder())
    assert b.available is False and b.last_error == "no DATABASE_URL"


def test_changing_embedding_model_is_refused_with_an_explanation():
    """A vector column is fixed at one dimension. Silently failing every
    upsert with a type error is not an acceptable way to learn that."""
    b = _backend("t_dim")
    b.upsert([{"id": "a", "text": "alpha", "contract_id": "k1"}])

    class Bigger(_FakeEmbedder):
        dimension = 8
        model = "some-other-embedder"

        def embed(self, texts):
            return [[0.1] * 8 for _ in texts]

    clash = PgVectorBackend("t_dim", url=DATABASE_URL, embedder=Bigger(),
                            table=TEST_TABLE)
    assert clash.available is False
    assert "dimension 4" in clash.last_error
    assert "some-other-embedder" in clash.last_error


def test_an_empty_table_is_rebuilt_for_a_new_model():
    b = _backend("t_rebuild")
    with b.conn.cursor() as cur:
        cur.execute(f"DELETE FROM {TEST_TABLE}")

    class Bigger(_FakeEmbedder):
        dimension = 8

        def embed(self, texts):
            return [[0.1] * 8 for _ in texts]

    rebuilt = PgVectorBackend("t_rebuild", url=DATABASE_URL, embedder=Bigger(),
                              table=TEST_TABLE)
    assert rebuilt.available is True
    assert rebuilt.stats()["dimension"] == 8
    with rebuilt.conn.cursor() as cur:      # leave the table as the others expect
        cur.execute(f"DROP TABLE {TEST_TABLE}")


def test_a_broken_embedder_fails_loudly():
    class Dead:
        available = False
        last_error = "Ollama unreachable"
        model = "x"

    b = PgVectorBackend("t", url=DATABASE_URL, embedder=Dead())
    assert b.available is False
    assert "embedder is not" in b.last_error


def test_postgres_is_preferred_over_pinecone_when_both_are_configured(monkeypatch):
    """One store beats two: the analysis and its vectors stay together."""
    from api.vectors import choose_backend

    monkeypatch.setenv("VECTOR_BACKEND", "auto")
    monkeypatch.setenv("PINECONE_API_KEY", "pc-test")
    monkeypatch.setenv("DATABASE_URL", DATABASE_URL)
    embedder = OllamaEmbedder()
    if not embedder.available:
        pytest.skip("needs a working embedder")
    assert choose_backend("t_pref", embedder=embedder).name == "pgvector"


# ── grounding, end to end ────────────────────────────────────────────────

def test_retrieval_through_postgres_stays_grounded():
    embedder = OllamaEmbedder()
    if not embedder.available:
        pytest.skip("needs Ollama for real embeddings")

    from api import demo
    from api.pipeline import analyze_portfolio
    from api.rag import Retriever

    bundles = demo.load(date(2026, 8, 27))
    gaps = analyze_portfolio(bundles)
    backend = PgVectorBackend("t_ground", url=DATABASE_URL, embedder=embedder,
                              table="embeddings_test_real")
    if not backend.available:
        pytest.skip(backend.last_error or "pgvector unavailable")
    with backend.conn.cursor() as cur:
        cur.execute("DELETE FROM embeddings_test_real WHERE namespace = 't_ground'")

    r = Retriever(bundles, gaps, date(2026, 8, 27),
                  vectors=VectorIndex("t_ground", backend=backend))
    assert r.sync_vectors()["synced"] > 0

    docs = {d.filename: d for b in bundles for d in b.docs}
    checked = 0
    for hit in r.search("how quickly must they report a breach", k=6):
        citation = hit.citation()
        if citation.get("quote") and citation.get("file") in docs:
            doc = docs[citation["file"]]
            assert doc.text[citation["start"]:citation["end"]] == citation["quote"]
            checked += 1
    assert checked, "expected at least one grounded citation"
