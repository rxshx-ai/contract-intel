"""Persistence. Postgres when DATABASE_URL is set, SQLite otherwise.

Two reasons this exists rather than a straight swap to Postgres:

  * Tests and the offline demo must keep running with no server. A hard
    Postgres dependency would mean 237 tests that need a container.
  * The interesting change is not the engine. It is that the ANALYSIS now
    survives a restart. Previously `_state["bundles"]` lived in memory, so an
    uploaded contract vanished when the process recycled -- fine locally,
    embarrassing once it is a URL you send people, and the reason the service
    could only ever run as one instance.

Contracts are stored as their analysed form (the AnalysisResult payload plus
the raw documents) so a fresh process can rehydrate without re-extracting and
without spending a single token.
"""

from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

SQLITE_PATH = Path("contract_intel.db")


def database_url() -> str | None:
    url = os.environ.get("DATABASE_URL", "").strip()
    return url or None


def is_postgres() -> bool:
    return database_url() is not None


# --------------------------------------------------------------------------
# schema -- written once in portable SQL, with the few dialect differences
# isolated here rather than sprinkled through the queries.
# --------------------------------------------------------------------------

_TABLES = """
CREATE TABLE IF NOT EXISTS documents (
    id           TEXT PRIMARY KEY,
    tenant_id    TEXT NOT NULL,
    contract_id  TEXT,
    filename     TEXT NOT NULL,
    sha256       TEXT NOT NULL,
    payload      TEXT NOT NULL,
    created_at   TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS contracts (
    id           TEXT PRIMARY KEY,
    tenant_id    TEXT NOT NULL,
    title        TEXT NOT NULL,
    counterparty TEXT,
    our_role     TEXT,
    payload      TEXT NOT NULL,
    created_at   TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS audit_log (
    id           {serial},
    tenant_id    TEXT NOT NULL,
    actor        TEXT NOT NULL,
    action       TEXT NOT NULL,
    subject_id   TEXT,
    detail       TEXT,
    at           TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_documents_tenant ON documents(tenant_id);
CREATE INDEX IF NOT EXISTS idx_documents_contract ON documents(contract_id);
CREATE INDEX IF NOT EXISTS idx_contracts_tenant ON contracts(tenant_id);
CREATE INDEX IF NOT EXISTS idx_audit_tenant ON audit_log(tenant_id, at);
"""


class Store:
    """One interface, two engines. `?` placeholders are rewritten for Postgres."""

    def __init__(self, url: str | None = None, sqlite_path: Path | str = SQLITE_PATH,
                 tenant: str = "demo"):
        # Tenant is bound once rather than threaded through every call: every
        # query filters on it, and an argument you must remember to pass is an
        # isolation bug waiting to happen.
        self.tenant = tenant
        self.url = url if url is not None else database_url()
        self.postgres = self.url is not None
        if self.postgres:
            import psycopg

            self.conn = psycopg.connect(self.url, autocommit=True)
        else:
            self.conn = sqlite3.connect(str(sqlite_path), check_same_thread=False)
            self.conn.row_factory = sqlite3.Row
        self._migrate()

    # ---- dialect ------------------------------------------------------

    def _sql(self, sql: str) -> str:
        return sql.replace("?", "%s") if self.postgres else sql

    # Columns added after the first release. CREATE TABLE IF NOT EXISTS will not
    # add them to a database that already exists, and an index on a missing
    # column fails the whole migration -- so they are applied explicitly before
    # any index is created.
    _ADDED_COLUMNS = [("documents", "contract_id", "TEXT")]

    def _migrate(self) -> None:
        serial = ("BIGSERIAL PRIMARY KEY" if self.postgres
                  else "INTEGER PRIMARY KEY AUTOINCREMENT")
        script = _TABLES.format(serial=serial)
        statements = [st.strip() for st in script.split(";") if st.strip()]
        tables = [st for st in statements if st.upper().startswith("CREATE TABLE")]
        indexes = [st for st in statements if st.upper().startswith("CREATE INDEX")]

        for statement in tables:
            self._raw(statement)
        for table, column, coltype in self._ADDED_COLUMNS:
            if column not in self._columns(table):
                self._raw(f"ALTER TABLE {table} ADD COLUMN {column} {coltype}")
        for statement in indexes:
            self._raw(statement)

    def _raw(self, statement: str) -> None:
        if self.postgres:
            with self.conn.cursor() as cur:
                cur.execute(statement)
        else:
            self.conn.execute(statement)
            self.conn.commit()

    def _columns(self, table: str) -> set[str]:
        if self.postgres:
            rows = self.query(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_name = ?", (table,))
            return {r["column_name"] for r in rows}
        rows = self.conn.execute(f"PRAGMA table_info({table})").fetchall()
        return {r[1] for r in rows}

    def execute(self, sql: str, args: Iterable[Any] = ()) -> None:
        sql = self._sql(sql)
        if self.postgres:
            with self.conn.cursor() as cur:
                cur.execute(sql, tuple(args))
        else:
            self.conn.execute(sql, tuple(args))
            self.conn.commit()

    def query(self, sql: str, args: Iterable[Any] = ()) -> list[dict[str, Any]]:
        sql = self._sql(sql)
        if self.postgres:
            with self.conn.cursor() as cur:
                cur.execute(sql, tuple(args))
                cols = [c.name for c in cur.description or []]
                return [dict(zip(cols, row)) for row in cur.fetchall()]
        rows = self.conn.execute(sql, tuple(args)).fetchall()
        return [dict(r) for r in rows]

    def close(self) -> None:
        self.conn.close()

    # ---- documents -----------------------------------------------------

    def save_document(self, doc, contract_id: str | None = None) -> None:
        self.execute(
            "DELETE FROM documents WHERE id = ?", (doc.id,))
        self.execute(
            "INSERT INTO documents (id, tenant_id, contract_id, filename, sha256, "
            "payload, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (doc.id, self.tenant, contract_id, doc.filename, doc.sha256,
             doc.model_dump_json(), _now()),
        )

    def documents_for(self, contract_id: str) -> list[dict[str, Any]]:
        rows = self.query(
            "SELECT payload FROM documents WHERE tenant_id = ? AND contract_id = ? "
            "ORDER BY created_at", (self.tenant, contract_id))
        return [json.loads(r["payload"]) for r in rows]

    # ---- contracts -----------------------------------------------------

    def save_contract(self, contract, payload: str) -> None:
        self.execute("DELETE FROM contracts WHERE id = ?", (contract.id,))
        self.execute(
            "INSERT INTO contracts (id, tenant_id, title, counterparty, our_role, "
            "payload, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (contract.id, self.tenant, contract.title, contract.counterparty,
             contract.our_role.value, payload, _now()),
        )

    def contract_ids(self) -> list[str]:
        return [r["id"] for r in self.query(
            "SELECT id FROM contracts WHERE tenant_id = ? ORDER BY created_at",
            (self.tenant,))]

    def get_contract(self, contract_id: str) -> dict[str, Any] | None:
        rows = self.query(
            "SELECT payload FROM contracts WHERE tenant_id = ? AND id = ?",
            (self.tenant, contract_id))
        return json.loads(rows[0]["payload"]) if rows else None

    def list_contracts(self) -> list[dict[str, Any]]:
        return self.query(
            "SELECT id, title, counterparty, our_role FROM contracts "
            "WHERE tenant_id = ? ORDER BY created_at DESC", (self.tenant,))

    def delete_contract(self, contract_id: str) -> None:
        self.execute("DELETE FROM documents WHERE tenant_id = ? AND contract_id = ?",
                     (self.tenant, contract_id))
        self.execute("DELETE FROM contracts WHERE tenant_id = ? AND id = ?",
                     (self.tenant, contract_id))

    # ---- audit ---------------------------------------------------------

    def audit(self, actor: str, action: str,
              subject_id: str | None = None, detail: str | None = None) -> None:
        self.execute(
            "INSERT INTO audit_log (tenant_id, actor, action, subject_id, detail, at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (self.tenant, actor, action, subject_id, detail, _now()),
        )

    def read_audit(self, limit: int = 200) -> list[dict[str, Any]]:
        return self.query(
            "SELECT actor, action, subject_id, detail, at FROM audit_log "
            "WHERE tenant_id = ? ORDER BY id DESC LIMIT ?", (self.tenant, limit))


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
