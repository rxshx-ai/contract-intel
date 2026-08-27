"""Persistence. Runs on SQLite; the Postgres path is the same SQL.

Set DATABASE_URL to run this suite against a real Postgres instead.
"""

import json
import os
from datetime import date

import pytest

from api.pipeline import ContractBundle
from api.store import Store


@pytest.fixture
def store(tmp_path):
    s = Store(url=os.environ.get("DATABASE_URL"),
              sqlite_path=tmp_path / "t.db", tenant="t1")
    yield s
    s.close()


@pytest.fixture(scope="module")
def bundle():
    from api import demo

    return demo.load(date(2026, 8, 27))[0]


# ── the gap this closes ──────────────────────────────────────────────────

def test_a_bundle_round_trips_through_storage(bundle, store):
    """An analysed contract must survive a restart without re-extracting.
    Previously it lived only in memory and an upload died with the process."""
    store.save_contract(bundle.contract, json.dumps(bundle.to_payload()))
    restored = ContractBundle.from_payload(store.get_contract(bundle.contract.id))

    assert len(restored.claims) == len(bundle.claims)
    assert len(restored.obligations) == len(bundle.obligations)
    assert len(restored.findings) == len(bundle.findings)
    assert restored.grounding_rate == bundle.grounding_rate
    assert restored.result().risk.overall == bundle.result().risk.overall


def test_restored_quotes_are_still_grounded(bundle, store):
    """Serialisation must not disturb offsets -- invariant 1 across a restart."""
    store.save_contract(bundle.contract, json.dumps(bundle.to_payload()))
    restored = ContractBundle.from_payload(store.get_contract(bundle.contract.id))
    docs = {d.id: d for d in restored.docs}
    for claim in restored.claims:
        assert claim.span.is_grounded_in(docs[claim.span.doc_id].text)


def test_saving_twice_replaces_rather_than_duplicates(bundle, store):
    payload = json.dumps(bundle.to_payload())
    store.save_contract(bundle.contract, payload)
    store.save_contract(bundle.contract, payload)
    assert len(store.list_contracts()) == 1


# ── tenancy ──────────────────────────────────────────────────────────────

def test_tenants_cannot_see_each_other(bundle, tmp_path):
    payload = json.dumps(bundle.to_payload())
    a = Store(url=None, sqlite_path=tmp_path / "shared.db", tenant="a")
    b = Store(url=None, sqlite_path=tmp_path / "shared.db", tenant="b")
    a.save_contract(bundle.contract, payload)

    assert len(a.list_contracts()) == 1
    assert b.list_contracts() == []
    assert b.get_contract(bundle.contract.id) is None
    a.close(); b.close()


# ── documents and audit ──────────────────────────────────────────────────

def test_documents_are_linked_to_their_contract(bundle, store):
    for doc in bundle.docs:
        store.save_document(doc, contract_id=bundle.contract.id)
    assert len(store.documents_for(bundle.contract.id)) == len(bundle.docs)
    assert store.documents_for("k_nonexistent") == []


def test_audit_is_append_only_and_newest_first(store):
    store.audit("me", "upload", "d1", "first")
    store.audit("me", "view", "d1", "second")
    rows = store.read_audit()
    assert [r["detail"] for r in rows] == ["second", "first"]


def test_delete_removes_contract_and_its_documents(bundle, store):
    store.save_contract(bundle.contract, json.dumps(bundle.to_payload()))
    for doc in bundle.docs:
        store.save_document(doc, contract_id=bundle.contract.id)
    store.delete_contract(bundle.contract.id)
    assert store.list_contracts() == []
    assert store.documents_for(bundle.contract.id) == []


# ── migration ────────────────────────────────────────────────────────────

def test_an_older_database_is_migrated_not_broken(tmp_path):
    """CREATE TABLE IF NOT EXISTS will not add a column to an existing table,
    and an index on the missing column fails the whole migration."""
    import sqlite3

    path = tmp_path / "old.db"
    old = sqlite3.connect(path)
    old.execute("CREATE TABLE documents (id TEXT PRIMARY KEY, tenant_id TEXT, "
                "filename TEXT, sha256 TEXT, payload TEXT, created_at TEXT)")
    old.commit(); old.close()

    s = Store(url=None, sqlite_path=path, tenant="t1")
    assert "contract_id" in s._columns("documents")
    s.close()
