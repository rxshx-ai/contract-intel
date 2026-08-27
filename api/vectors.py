"""Pinecone vector index for semantic retrieval.

WHY THIS SITS ALONGSIDE BM25 RATHER THAN REPLACING IT
-----------------------------------------------------
Each is better at something the other is bad at, and contract questions need
both:

  BM25    exact terms, numbers, clause-type names. "99.9%", "liability cap",
          "forty-five (45) days". Lexical match is not a weaker form of
          semantic match here -- a contract question often hinges on a literal
          figure, and an embedding will happily rank 99.99% next to 99.9%.
  Vectors paraphrase and vocabulary mismatch. "what if they go bust" finds
          insolvency and termination wording that shares no words with the
          question. BM25 returns nothing for that.

They are fused with Reciprocal Rank Fusion, which needs no score calibration
between two systems whose scores are not comparable.

EMBEDDINGS
----------
Pinecone's integrated inference embeds server-side: we upsert text and it
handles the model. That keeps the credential count at one, which matters
because Groq serves no embedding model and the alternative was adding OpenAI
or Cohere purely to make vectors work.

Set PINECONE_API_KEY to enable. Without it this module is inert and retrieval
falls back to BM25 alone -- the tests and the offline demo depend on that.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Iterable

INDEX_NAME = os.environ.get("PINECONE_INDEX", "contract-intel")
EMBED_MODEL = os.environ.get("PINECONE_EMBED_MODEL", "llama-text-embed-v2")
CLOUD = os.environ.get("PINECONE_CLOUD", "aws")
REGION = os.environ.get("PINECONE_REGION", "us-east-1")
BATCH = 90                      # Pinecone caps upsert_records batches
MAX_TEXT_CHARS = 1800


def enabled() -> bool:
    return bool(os.environ.get("PINECONE_API_KEY", "").strip())


@dataclass
class VectorHit:
    id: str
    score: float
    metadata: dict[str, Any]


class VectorIndex:
    """Thin wrapper. Every method is a no-op when Pinecone is not configured,
    so callers never branch on availability."""

    def __init__(self, namespace: str = "default", index_name: str = INDEX_NAME):
        self.namespace = namespace
        self.index_name = index_name
        self._index = None
        self.available = enabled()
        self.last_error: str | None = None

    # ---- lifecycle -----------------------------------------------------

    def connect(self) -> bool:
        if not self.available:
            return False
        if self._index is not None:
            return True
        try:
            from pinecone import Pinecone

            pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
            existing = {i["name"] for i in pc.list_indexes()}
            if self.index_name not in existing:
                # An index with integrated inference embeds on upsert, so the
                # text field is named here and never embedded client-side.
                pc.create_index_for_model(
                    name=self.index_name,
                    cloud=CLOUD,
                    region=REGION,
                    embed={"model": EMBED_MODEL,
                           "field_map": {"text": "text"}},
                )
            self._index = pc.Index(self.index_name)
            return True
        except Exception as exc:                      # noqa: BLE001
            self.last_error = f"{type(exc).__name__}: {exc}"
            self.available = False
            return False

    # ---- writes --------------------------------------------------------

    def upsert(self, items: Iterable[dict[str, Any]]) -> int:
        """Items: {id, text, kind, contract_id, ...}. Text is embedded by
        Pinecone. Returns the number written."""
        if not self.connect():
            return 0
        batch: list[dict[str, Any]] = []
        written = 0
        for item in items:
            text = (item.get("text") or "").strip()
            if not text:
                continue
            record = {k: v for k, v in item.items() if v is not None}
            record["_id"] = record.pop("id")
            record["text"] = text[:MAX_TEXT_CHARS]
            batch.append(record)
            if len(batch) >= BATCH:
                written += self._flush(batch)
                batch = []
        if batch:
            written += self._flush(batch)
        return written

    def _flush(self, batch: list[dict[str, Any]]) -> int:
        try:
            self._index.upsert_records(self.namespace, batch)
            return len(batch)
        except Exception as exc:                      # noqa: BLE001
            self.last_error = f"{type(exc).__name__}: {exc}"
            return 0

    def delete_contract(self, contract_id: str) -> None:
        if not self.connect():
            return
        try:
            self._index.delete(filter={"contract_id": {"$eq": contract_id}},
                               namespace=self.namespace)
        except Exception as exc:                      # noqa: BLE001
            self.last_error = f"{type(exc).__name__}: {exc}"

    # ---- reads ---------------------------------------------------------

    def search(self, query: str, k: int = 10,
               contract_id: str | None = None) -> list[VectorHit]:
        if not self.connect() or not query.strip():
            return []
        body: dict[str, Any] = {"inputs": {"text": query}, "top_k": k}
        if contract_id:
            body["filter"] = {"contract_id": {"$eq": contract_id}}
        try:
            response = self._index.search(namespace=self.namespace, query=body)
        except Exception as exc:                      # noqa: BLE001
            self.last_error = f"{type(exc).__name__}: {exc}"
            return []
        hits = []
        for row in (response.get("result", {}) or {}).get("hits", []) or []:
            hits.append(VectorHit(
                id=row.get("_id") or row.get("id", ""),
                score=float(row.get("_score", 0.0)),
                metadata=row.get("fields", {}) or {},
            ))
        return hits

    def stats(self) -> dict[str, Any]:
        if not self.connect():
            return {"enabled": False, "reason": self.last_error or "no PINECONE_API_KEY"}
        try:
            described = self._index.describe_index_stats()
            namespaces = described.get("namespaces", {}) or {}
            return {
                "enabled": True,
                "index": self.index_name,
                "model": EMBED_MODEL,
                "namespace": self.namespace,
                "vectors": namespaces.get(self.namespace, {}).get("vector_count", 0),
                "total": described.get("total_vector_count", 0),
            }
        except Exception as exc:                      # noqa: BLE001
            return {"enabled": False, "reason": f"{type(exc).__name__}: {exc}"}


# --------------------------------------------------------------------------
# fusion
# --------------------------------------------------------------------------

RRF_K = 60


def reciprocal_rank_fusion(*rankings: list[str], k: int = RRF_K) -> dict[str, float]:
    """Combine rankings whose scores are not on the same scale.

    BM25 returns unbounded relevance; Pinecone returns a bounded similarity.
    Blending those numerically requires calibration that drifts the moment
    either side changes. RRF only uses POSITION, so it needs none.
    """
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, item_id in enumerate(ranking):
            scores[item_id] = scores.get(item_id, 0.0) + 1.0 / (k + rank + 1)
    return scores
