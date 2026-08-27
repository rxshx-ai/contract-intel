"""SQLite persistence. One file, no ORM, schema inline.

Row-level tenant isolation is present from the first migration rather than
retrofitted: every row carries a tenant_id and every query filters on it, so
the single-tenant demo and a multi-tenant deployment share one code path.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DB_PATH = Path("contract_intel.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    id           TEXT PRIMARY KEY,
    tenant_id    TEXT NOT NULL,
    filename     TEXT NOT NULL,
    sha256       TEXT NOT NULL,
    payload      TEXT NOT NULL,
    created_at   TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_documents_tenant ON documents(tenant_id);

CREATE TABLE IF NOT EXISTS contracts (
    id           TEXT PRIMARY KEY,
    tenant_id    TEXT NOT NULL,
    title        TEXT NOT NULL,
    counterparty TEXT,
    our_role     TEXT,
    payload      TEXT NOT NULL,
    created_at   TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_contracts_tenant ON contracts(tenant_id);

-- Append-only. No UPDATE or DELETE is ever issued against this table.
CREATE TABLE IF NOT EXISTS audit_log (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    tenant_id    TEXT NOT NULL,
    actor        TEXT NOT NULL,
    action       TEXT NOT NULL,
    subject_id   TEXT,
    detail       TEXT,
    at           TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_audit_tenant ON audit_log(tenant_id, at);
"""


def connect(path: Path | str = DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    return conn


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def save_document(conn, tenant_id: str, doc) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO documents (id, tenant_id, filename, sha256, payload, "
        "created_at) VALUES (?, ?, ?, ?, ?, ?)",
        (doc.id, tenant_id, doc.filename, doc.sha256, doc.model_dump_json(), _now()),
    )
    conn.commit()


def get_documents(conn, tenant_id: str) -> list[dict[str, Any]]:
    rows = conn.execute(
        "SELECT payload FROM documents WHERE tenant_id = ? ORDER BY created_at",
        (tenant_id,),
    ).fetchall()
    return [json.loads(r["payload"]) for r in rows]


def save_contract(conn, tenant_id: str, contract, result_json: str) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO contracts (id, tenant_id, title, counterparty, "
        "our_role, payload, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (contract.id, tenant_id, contract.title, contract.counterparty,
         contract.our_role.value, result_json, _now()),
    )
    conn.commit()


def get_contract(conn, tenant_id: str, contract_id: str) -> dict[str, Any] | None:
    row = conn.execute(
        "SELECT payload FROM contracts WHERE tenant_id = ? AND id = ?",
        (tenant_id, contract_id),
    ).fetchone()
    return json.loads(row["payload"]) if row else None


def list_contracts(conn, tenant_id: str) -> list[dict[str, Any]]:
    rows = conn.execute(
        "SELECT id, title, counterparty, our_role FROM contracts WHERE tenant_id = ? "
        "ORDER BY created_at DESC",
        (tenant_id,),
    ).fetchall()
    return [dict(r) for r in rows]


def audit(conn, tenant_id: str, actor: str, action: str,
          subject_id: str | None = None, detail: str | None = None) -> None:
    """Append-only. Every extraction and every view lands here."""
    conn.execute(
        "INSERT INTO audit_log (tenant_id, actor, action, subject_id, detail, at) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (tenant_id, actor, action, subject_id, detail, _now()),
    )
    conn.commit()


def read_audit(conn, tenant_id: str, limit: int = 200) -> list[dict[str, Any]]:
    rows = conn.execute(
        "SELECT actor, action, subject_id, detail, at FROM audit_log "
        "WHERE tenant_id = ? ORDER BY id DESC LIMIT ?",
        (tenant_id, limit),
    ).fetchall()
    return [dict(r) for r in rows]
