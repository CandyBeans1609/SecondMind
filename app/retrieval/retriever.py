from typing import Optional

from app.ingestion.embedder import embed_single
from app.vectorstore.store import query_collection
from config import settings


def retrieve(
    query: str,
    top_k: Optional[int] = None,
    doc_id: Optional[str] = None,
) -> list[dict]:
    """
    Embed the query and return the top-k most relevant chunks.

    Returns:
        [
            {
                "text": str,
                "source": str,
                "page": int,
                "chunk_index": int,
                "score": float,   # cosine similarity, 0–1
            },
            ...
        ]
    """
    k = top_k or settings.top_k

    # 1. Embed the raw query string with the same model used during ingestion
    query_embedding = embed_single(query)

    # 2. Search ChromaDB
    hits = query_collection(query_embedding, top_k=k, doc_id=doc_id)

    return hits