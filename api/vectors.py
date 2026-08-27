"""Semantic retrieval, with a pluggable backend.

WHY THIS SITS ALONGSIDE BM25 RATHER THAN REPLACING IT
-----------------------------------------------------
  BM25    exact terms, numbers, clause-type names. "99.9%", "forty-five (45)
          days". An embedding happily ranks 99.99% beside 99.9%, which is
          exactly the distinction a contract question turns on.
  Vectors paraphrase. "what if they go bust" finds insolvency and termination
          wording sharing no words with the question. BM25 returns nothing.

Fused with Reciprocal Rank Fusion (see `reciprocal_rank_fusion`), which uses
POSITION only -- BM25 scores are unbounded and Pinecone's are bounded, so a
numeric blend would need a calibration that drifts whenever either side moves.

BACKENDS
--------
  pinecone  production. Integrated inference: we upsert text and Pinecone
            embeds server-side, which keeps the credential count at one --
            Groq serves no embedding model, so the alternative was adding
            OpenAI or Cohere purely to make vectors work.
  local     Ollama embeddings + cosine over an in-process matrix. No account,
            no network beyond localhost, and it makes the hybrid path
            genuinely runnable rather than merely written.
  none      BM25 only. Every method is a no-op so callers never branch.

Chosen by VECTOR_BACKEND, or auto: Pinecone if PINECONE_API_KEY is set,
otherwise local if Ollama answers, otherwise none.
"""

from __future__ import annotations

import json
import os
import pathlib
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Iterable, Protocol

INDEX_NAME = os.environ.get("PINECONE_INDEX", "contract-intel")
EMBED_MODEL = os.environ.get("PINECONE_EMBED_MODEL", "llama-text-embed-v2")
CLOUD = os.environ.get("PINECONE_CLOUD", "aws")
REGION = os.environ.get("PINECONE_REGION", "us-east-1")
BATCH = 90
MAX_TEXT_CHARS = 1800

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")
OLLAMA_EMBED_MODEL = os.environ.get("OLLAMA_EMBED_MODEL", "nomic-embed-text")
LOCAL_STORE = pathlib.Path(os.environ.get("VECTOR_STORE", ".vectors"))


class OllamaEmbedder:
    """nomic-embed-text via Ollama. The SAME model feeds the local index and
    Pinecone, so a query embedded here matches vectors written from here.

    Using Pinecone's integrated inference instead would mean their hosted model
    in production and this one in development -- two different vector spaces,
    and retrieval quality you cannot reproduce locally.
    """

    def __init__(self, model: str = OLLAMA_EMBED_MODEL, url: str = OLLAMA_URL):
        self.model = model
        self.url = url.rstrip("/")
        self.last_error: str | None = None
        self._dimension: int | None = None
        self.available = self._probe()

    def _probe(self) -> bool:
        try:
            with urllib.request.urlopen(f"{self.url}/api/tags", timeout=3) as r:
                tags = json.loads(r.read())
            names = {m.get("name", "").split(":")[0] for m in tags.get("models", [])}
            if self.model.split(":")[0] not in names:
                self.last_error = (f"Ollama is running but '{self.model}' is not "
                                   f"pulled. Run: ollama pull {self.model}")
                return False
            return True
        except Exception as exc:                       # noqa: BLE001
            self.last_error = f"Ollama unreachable at {self.url} ({type(exc).__name__})"
            return False

    def embed(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for text in texts:
            body = json.dumps({"model": self.model, "prompt": text}).encode()
            req = urllib.request.Request(
                f"{self.url}/api/embeddings", data=body,
                headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=60) as r:
                out.append(json.loads(r.read())["embedding"])
        return out

    @property
    def dimension(self) -> int:
        """Probed once. Pinecone needs it at index creation and it must match
        the vectors written later, or every upsert is rejected."""
        if self._dimension is None:
            self._dimension = len(self.embed(["dimension probe"])[0])
        return self._dimension


@dataclass
class VectorHit:
    id: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)


class Backend(Protocol):
    available: bool
    last_error: str | None

    def upsert(self, items: list[dict[str, Any]]) -> int: ...
    def search(self, query: str, k: int, contract_id: str | None) -> list[VectorHit]: ...
    def delete_contract(self, contract_id: str) -> None: ...
    def stats(self) -> dict[str, Any]: ...


# --------------------------------------------------------------------------
# disabled
# --------------------------------------------------------------------------

class NullBackend:
    name = "none"

    def __init__(self, reason: str = "no vector backend configured"):
        self.available = False
        self.last_error = reason

    def upsert(self, items): return 0
    def search(self, query, k, contract_id=None): return []
    def delete_contract(self, contract_id): return None
    def stats(self): return {"enabled": False, "backend": "none",
                             "reason": self.last_error}


# --------------------------------------------------------------------------
# local: Ollama embeddings + cosine
# --------------------------------------------------------------------------

class LocalBackend:
    """Embeddings from Ollama, similarity in-process.

    Honest about scale: this holds every vector in memory and scores them all.
    At ~15,000 records (300 contracts) that is a 15,000 x 768 matrix -- about
    45 MB and a few milliseconds per query with numpy. Well past that, use
    Pinecone.
    """

    name = "local"

    def __init__(self, namespace: str = "default",
                 embedder: "OllamaEmbedder | None" = None):
        self.namespace = namespace
        self.embedder = embedder or OllamaEmbedder()
        self.model = self.embedder.model
        self.last_error = self.embedder.last_error
        self.ids: list[str] = []
        self.meta: list[dict[str, Any]] = []
        self._matrix = None
        self.path = LOCAL_STORE / f"{namespace}.json"
        self.available = self.embedder.available
        if self.available:
            self._load()

    def embed(self, texts: list[str]) -> list[list[float]]:
        return self.embedder.embed(texts)

    # ---- persistence ---------------------------------------------------

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            import numpy as np

            blob = json.loads(self.path.read_text())
            self.ids = blob["ids"]
            self.meta = blob["meta"]
            self._matrix = np.array(blob["vectors"], dtype="float32")
        except Exception as exc:                       # noqa: BLE001
            self.last_error = f"could not load {self.path}: {exc}"

    def _save(self) -> None:
        import numpy as np

        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps({
            "ids": self.ids, "meta": self.meta,
            "vectors": np.asarray(self._matrix).tolist(),
        }))

    # ---- api -----------------------------------------------------------

    def upsert(self, items: list[dict[str, Any]]) -> int:
        if not self.available or not items:
            return 0
        import numpy as np

        try:
            vectors = self.embed([i["text"][:MAX_TEXT_CHARS] for i in items])
        except Exception as exc:                       # noqa: BLE001
            self.last_error = f"embedding failed: {type(exc).__name__}: {exc}"
            return 0

        index = {rid: n for n, rid in enumerate(self.ids)}
        rows = list(np.asarray(self._matrix)) if self._matrix is not None else []
        for item, vector in zip(items, vectors):
            arr = np.asarray(vector, dtype="float32")
            arr = arr / (np.linalg.norm(arr) or 1.0)   # normalise once, not per query
            meta = {k: v for k, v in item.items() if k != "text"}
            meta["text"] = item["text"][:400]
            if item["id"] in index:
                rows[index[item["id"]]] = arr
                self.meta[index[item["id"]]] = meta
            else:
                index[item["id"]] = len(self.ids)
                self.ids.append(item["id"])
                self.meta.append(meta)
                rows.append(arr)
        self._matrix = np.vstack(rows) if rows else None
        self._save()
        return len(items)

    def search(self, query: str, k: int = 10,
               contract_id: str | None = None) -> list[VectorHit]:
        if not self.available or self._matrix is None or not query.strip():
            return []
        import numpy as np

        try:
            q = np.asarray(self.embed([query])[0], dtype="float32")
        except Exception as exc:                       # noqa: BLE001
            self.last_error = f"embedding failed: {type(exc).__name__}: {exc}"
            return []
        q = q / (np.linalg.norm(q) or 1.0)
        scores = np.asarray(self._matrix) @ q

        order = np.argsort(-scores)
        hits: list[VectorHit] = []
        for i in order:
            meta = self.meta[int(i)]
            if contract_id and meta.get("contract_id") != contract_id:
                continue
            hits.append(VectorHit(self.ids[int(i)], float(scores[int(i)]), meta))
            if len(hits) >= k:
                break
        return hits

    def delete_contract(self, contract_id: str) -> None:
        if not self.available or self._matrix is None:
            return
        import numpy as np

        keep = [n for n, m in enumerate(self.meta)
                if m.get("contract_id") != contract_id]
        self.ids = [self.ids[n] for n in keep]
        self.meta = [self.meta[n] for n in keep]
        matrix = np.asarray(self._matrix)
        self._matrix = matrix[keep] if keep else None
        self._save()

    def stats(self) -> dict[str, Any]:
        if not self.available:
            return {"enabled": False, "backend": "local", "reason": self.last_error}
        return {"enabled": True, "backend": "local", "model": self.model,
                "namespace": self.namespace, "vectors": len(self.ids),
                "store": str(self.path)}


# --------------------------------------------------------------------------
# pinecone
# --------------------------------------------------------------------------

class PineconeBackend:
    """Pinecone as pure vector storage. Embeddings come from OllamaEmbedder.

    Deliberately NOT Pinecone integrated inference. Integrated inference embeds
    server-side with their hosted model, which would mean one vector space in
    production and a different one locally -- retrieval you cannot reproduce or
    debug on your machine. Generating vectors ourselves keeps both identical.

    GROUNDING: Pinecone stores an id, a vector, and a little metadata. The text
    it holds is for debugging only and is NEVER what the user sees. Every hit
    is resolved back through the local verified layer (Retriever.by_id), so a
    displayed quote is always one that was checked character-for-character
    against the source document. An id Pinecone returns that we do not
    recognise is dropped, not rendered.
    """

    name = "pinecone"

    def __init__(self, namespace: str = "default", index_name: str = INDEX_NAME,
                 embedder: "OllamaEmbedder | None" = None):
        self.namespace = namespace
        self.index_name = index_name
        self.embedder = embedder or OllamaEmbedder()
        self._index = None
        self.last_error: str | None = None
        if not os.environ.get("PINECONE_API_KEY", "").strip():
            self.available = False
            self.last_error = "no PINECONE_API_KEY"
        elif not self.embedder.available:
            self.available = False
            self.last_error = (f"Pinecone is configured but the embedder is not: "
                               f"{self.embedder.last_error}")
        else:
            self.available = True

    def connect(self) -> bool:
        if not self.available:
            return False
        if self._index is not None:
            return True
        try:
            from pinecone import Pinecone, ServerlessSpec

            pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
            names = {i["name"] for i in pc.list_indexes()}
            if self.index_name not in names:
                # Dimension must match the embedder exactly or every upsert is
                # rejected; it is probed rather than hardcoded.
                pc.create_index(
                    name=self.index_name,
                    dimension=self.embedder.dimension,
                    metric="cosine",
                    spec=ServerlessSpec(cloud=CLOUD, region=REGION),
                )
            self._index = pc.Index(self.index_name)
            return True
        except Exception as exc:                       # noqa: BLE001
            self.last_error = f"{type(exc).__name__}: {exc}"
            self.available = False
            return False

    # ---- writes --------------------------------------------------------

    def upsert(self, items: list[dict[str, Any]]) -> int:
        if not self.connect() or not items:
            return 0
        written = 0
        for start in range(0, len(items), BATCH):
            chunk = [i for i in items[start:start + BATCH]
                     if (i.get("text") or "").strip()]
            if not chunk:
                continue
            try:
                vectors = self.embedder.embed(
                    [i["text"][:MAX_TEXT_CHARS] for i in chunk])
            except Exception as exc:                   # noqa: BLE001
                self.last_error = f"embedding failed: {type(exc).__name__}: {exc}"
                return written
            payload = []
            for item, vector in zip(chunk, vectors):
                metadata = {k: v for k, v in item.items()
                            if k not in ("id", "text") and v is not None}
                # Kept for debugging in the Pinecone console. Never displayed --
                # the user always sees the verified local copy.
                metadata["text"] = item["text"][:400]
                payload.append({"id": item["id"], "values": vector,
                                "metadata": metadata})
            try:
                self._index.upsert(vectors=payload, namespace=self.namespace)
                written += len(payload)
            except Exception as exc:                   # noqa: BLE001
                self.last_error = f"{type(exc).__name__}: {exc}"
                return written
        return written

    def delete_contract(self, contract_id: str) -> None:
        if not self.connect():
            return
        try:
            self._index.delete(filter={"contract_id": {"$eq": contract_id}},
                               namespace=self.namespace)
        except Exception as exc:                       # noqa: BLE001
            self.last_error = f"{type(exc).__name__}: {exc}"

    # ---- reads ---------------------------------------------------------

    def search(self, query: str, k: int = 10,
               contract_id: str | None = None) -> list[VectorHit]:
        if not self.connect() or not query.strip():
            return []
        try:
            vector = self.embedder.embed([query])[0]
        except Exception as exc:                       # noqa: BLE001
            self.last_error = f"embedding failed: {type(exc).__name__}: {exc}"
            return []
        kwargs: dict[str, Any] = {
            "vector": vector, "top_k": k, "namespace": self.namespace,
            "include_metadata": True,
        }
        if contract_id:
            kwargs["filter"] = {"contract_id": {"$eq": contract_id}}
        try:
            response = self._index.query(**kwargs)
        except Exception as exc:                       # noqa: BLE001
            self.last_error = f"{type(exc).__name__}: {exc}"
            return []
        return [VectorHit(id=_attr(m, "id"), score=float(_attr(m, "score") or 0.0),
                          metadata=_attr(m, "metadata") or {})
                for m in (_attr(response, "matches") or [])]

    def stats(self) -> dict[str, Any]:
        if not self.connect():
            return {"enabled": False, "backend": "pinecone",
                    "reason": self.last_error}
        try:
            described = self._index.describe_index_stats()
            namespaces = _attr(described, "namespaces") or {}
            ns = namespaces.get(self.namespace)
            return {"enabled": True, "backend": "pinecone", "index": self.index_name,
                    "model": self.embedder.model,
                    "dimension": self.embedder.dimension,
                    "namespace": self.namespace,
                    "vectors": (_attr(ns, "vector_count") if ns else 0) or 0}
        except Exception as exc:                       # noqa: BLE001
            return {"enabled": False, "backend": "pinecone",
                    "reason": f"{type(exc).__name__}: {exc}"}


def _attr(obj: Any, name: str) -> Any:
    """Pinecone returns msgspec Structs with dict-like access; be tolerant."""
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj.get(name)
    value = getattr(obj, name, None)
    if value is None and hasattr(obj, "__getitem__"):
        try:
            return obj[name]
        except Exception:                              # noqa: BLE001
            return None
    return value


def _hits(response: Any) -> list[Any]:
    result = _attr(response, "result")
    return list(_attr(result, "hits") or []) if result is not None else []


# --------------------------------------------------------------------------
# facade
# --------------------------------------------------------------------------

def choose_backend(namespace: str = "default",
                   embedder: "OllamaEmbedder | None" = None) -> Backend:
    """Both backends share one embedder: the vectors must come from the same
    model, or a query embedded locally would not match what is stored."""
    requested = os.environ.get("VECTOR_BACKEND", "auto").strip().lower()

    if requested in ("none", "off", "disabled"):
        return NullBackend("VECTOR_BACKEND=none")

    embedder = embedder or OllamaEmbedder()

    if requested == "pinecone":
        return PineconeBackend(namespace, embedder=embedder)
    if requested == "local":
        return LocalBackend(namespace, embedder=embedder)

    if os.environ.get("PINECONE_API_KEY", "").strip():
        pinecone = PineconeBackend(namespace, embedder=embedder)
        if pinecone.available:
            return pinecone
        # A Pinecone key with no working embedder is a misconfiguration worth
        # surfacing rather than silently downgrading.
        return NullBackend(pinecone.last_error or "pinecone unavailable")
    local = LocalBackend(namespace, embedder=embedder)
    return local if local.available else NullBackend(local.last_error or "no backend")


class VectorIndex:
    """Stable surface over whichever backend is configured."""

    def __init__(self, namespace: str = "default", backend: Backend | None = None):
        self.namespace = namespace
        self.backend = backend or choose_backend(namespace)

    @property
    def available(self) -> bool:
        return self.backend.available

    @property
    def last_error(self) -> str | None:
        return self.backend.last_error

    def upsert(self, items: Iterable[dict[str, Any]]) -> int:
        return self.backend.upsert(list(items))

    def search(self, query: str, k: int = 10,
               contract_id: str | None = None) -> list[VectorHit]:
        return self.backend.search(query, k, contract_id)

    def delete_contract(self, contract_id: str) -> None:
        self.backend.delete_contract(contract_id)

    def stats(self) -> dict[str, Any]:
        return self.backend.stats()


def enabled() -> bool:
    return bool(os.environ.get("PINECONE_API_KEY", "").strip())


RRF_K = 60


def reciprocal_rank_fusion(*rankings: list[str], k: int = RRF_K) -> dict[str, float]:
    """Combine rankings whose scores are not on the same scale.

    RRF uses POSITION only, so blending an unbounded BM25 score with a bounded
    cosine similarity needs no calibration.
    """
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, item_id in enumerate(ranking):
            scores[item_id] = scores.get(item_id, 0.0) + 1.0 / (k + rank + 1)
    return scores
