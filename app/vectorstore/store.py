import logging
import threading
from typing import Optional

import chromadb
from chromadb import ClientAPI, Collection
from chromadb.config import Settings as ChromaSettings

from config import settings

logger = logging.getLogger(__name__)

# ── Singleton client + collection ──────────────────────────────────────────────
# FIX 1: The deadlock happened because uvicorn --reload re-imports modules while
# the old process thread may still hold _lock. The fix is two-part:
#   a) Use a try/except heartbeat to detect a dead/stale client and rebuild it.
#   b) Document: never use --reload with PersistentClient (use --reload only in
#      dev without ChromaDB writes, or use uvicorn programmatically with
#      reload=False in production).

_client: Optional[ClientAPI] = None
_collection: Optional[Collection] = None
_lock = threading.Lock()


def get_client() -> ClientAPI:
    global _client
    if _client is None:
        with _lock:
            if _client is None:
                _client = _make_client()
    return _client


def _make_client() -> ClientAPI:
    logger.info("Initialising ChromaDB PersistentClient...")
    return chromadb.PersistentClient(
        path=str(settings.chroma_dir),
        settings=ChromaSettings(anonymized_telemetry=False),
    )


def get_collection() -> Collection:
    """
    Return (or create) the documents collection.

    If the underlying client has become stale (e.g. after a hot-reload),
    the heartbeat call will raise and we reinitialise transparently.
    """
    global _client, _collection
    # Fast path — both already initialised
    if _collection is not None:
        try:
            # Cheap heartbeat: just check the count. Raises if client is dead.
            _collection.count()
            return _collection
        except Exception:
            # Client is stale — fall through to reinitialise
            logger.warning("ChromaDB collection heartbeat failed — reinitialising.")
            _collection = None
            _client = None

    with _lock:
        # Double-checked locking
        if _collection is None:
            if _client is None:
                _client = _make_client()
            _collection = _client.get_or_create_collection(
                name=settings.chroma_collection,
                metadata={"hnsw:space": "cosine"},
            )
    return _collection


# ── Query helper ───────────────────────────────────────────────────────────────

def query_collection(
    query_embedding: list[float],
    top_k: Optional[int] = None,
    doc_id: Optional[str] = None,
) -> list[dict]:
    """
    Run a nearest-neighbour search and return the top-k results as dicts.
    """
    n = top_k or settings.top_k
    collection = get_collection()

    total = collection.count()
    if total == 0:
        return []

    # If filtering by doc_id, count only matching chunks so n doesn't exceed them
    where = {"doc_id": doc_id} if doc_id else None
    if where:
        scoped = collection.get(where=where, include=[])
        scoped_total = len(scoped["ids"])
        if scoped_total == 0:
            return []
        n = min(n, scoped_total)
    else:
        n = min(n, total)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n,
        include=["documents", "metadatas", "distances"],
        where=where,
    )

    hits = []
    for i in range(len(results["ids"][0])):
        meta = results["metadatas"][0][i]
        hits.append({
            "id": results["ids"][0][i],
            "text": results["documents"][0][i],
            "score": round(1 - results["distances"][0][i], 4),
            "source": meta.get("source", ""),
            "page": meta.get("page", 0),
            "chunk_index": meta.get("chunk_index", 0),
            "chunk_total": meta.get("chunk_total"),
            "char_start": meta.get("char_start"),
            "char_end": meta.get("char_end"),
            "content_type": meta.get("content_type", "text"),
            "section_heading": meta.get("section_heading", ""),
        })

    if settings.similarity_threshold > 0:
        hits = [h for h in hits if h["score"] >= settings.similarity_threshold]

    return hits


def list_documents() -> list[dict]:
    """Return a deduplicated list of ingested documents with chunk counts."""
    collection = get_collection()
    if collection.count() == 0:
        return []

    all_meta = collection.get(include=["metadatas"])["metadatas"]

    docs: dict[str, dict] = {}
    for meta in all_meta:
        doc_id = meta.get("doc_id", "unknown")
        if doc_id not in docs:
            docs[doc_id] = {
                "doc_id": doc_id,
                "filename": meta.get("source", ""),
                "chunk_count": 0,
            }
        docs[doc_id]["chunk_count"] += 1

    return list(docs.values())


def delete_document(doc_id: str) -> int:
    """Delete all chunks belonging to a document. Returns number of chunks removed."""
    collection = get_collection()
    results = collection.get(where={"doc_id": doc_id}, include=[])
    ids_to_delete = results["ids"]
    if ids_to_delete:
        collection.delete(ids=ids_to_delete)
    return len(ids_to_delete)