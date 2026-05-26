import hashlib
import logging
from pathlib import Path

from app.ingestion.loader import load_file
from app.ingestion.chunker import chunk_pages
from app.ingestion.embedder import embed_texts
from app.vectorstore.store import get_collection

logger = logging.getLogger(__name__)


def ingest_file(file_path: Path) -> dict:
    """
    Full ingestion pipeline for a single file:
        load → chunk → embed → upsert into ChromaDB

    Returns:
        {
            "doc_id": str,        # stable hash of the filename
            "filename": str,
            "chunk_count": int,
        }
    """
    logger.info(f"[1/4] Loading file: {file_path.name}")
    pages = load_file(file_path)
    if not pages:
        raise ValueError(f"No text could be extracted from '{file_path.name}'.")
    logger.info(f"      → {len(pages)} page(s) extracted")

    logger.info("[2/4] Chunking...")
    chunks = chunk_pages(pages)
    if not chunks:
        raise ValueError(f"Chunking produced no output for '{file_path.name}'.")
    logger.info(f"      → {len(chunks)} chunk(s) produced")

    logger.info(f"[3/4] Embedding {len(chunks)} chunks via Ollama (this may take a while)...")
    texts = [c["text"] for c in chunks]
    embeddings = embed_texts(texts)

    # ── FIX 2: Guard against length mismatch before touching ChromaDB ──────────
    if len(embeddings) != len(texts):
        raise ValueError(
            f"Embedding count mismatch: got {len(embeddings)} embeddings "
            f"for {len(texts)} chunks. Ollama may have dropped a request."
        )
    logger.info("      → Embeddings done")

    logger.info("[4/4] Upserting into ChromaDB...")
    doc_id = _make_doc_id(file_path.name)

    ids = [f"{doc_id}_{c['chunk_index']}" for c in chunks]

    # ── FIX 3: Explicitly cast metadata values to ChromaDB-safe types ──────────
    # ChromaDB only accepts str | int | float | bool. PyMuPDF can return
    # numpy.int64 for page numbers; cast everything explicitly.
    metadatas = [
        {
            "source": str(c["source"]),
            "page": int(c["page"]),
            "chunk_index": int(c["chunk_index"]),
            "doc_id": str(doc_id),
            "chunk_total": int(c["chunk_total"]),
            "char_start": int(c["char_start"]),
            "char_end": int(c["char_end"]),
            "content_type": str(c["content_type"]),
            "section_heading": str(c["section_heading"]),
        }
        for c in chunks
    ]

    collection = get_collection()

    # Delete any existing chunks for this doc before upserting (clean re-upload)
    existing = collection.get(where={"doc_id": doc_id}, include=[])
    if existing["ids"]:
        logger.info(f"      → Removing {len(existing['ids'])} stale chunk(s) for re-upload")
        collection.delete(ids=existing["ids"])

    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=texts,
        metadatas=metadatas,
    )
    logger.info(f"      → Done. doc_id={doc_id}")

    return {
        "doc_id": doc_id,
        "filename": file_path.name,
        "chunk_count": len(chunks),
    }


def _make_doc_id(filename: str) -> str:
    """Stable 12-char ID derived from the filename."""
    return hashlib.md5(filename.encode()).hexdigest()[:12]