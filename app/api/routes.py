import asyncio
import shutil
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

from app.api.schemas import (
    DeleteResponse,
    DocumentInfo,
    DocumentListResponse,
    HealthResponse,
    QueryRequest,
    QueryResponse,
    SourceChunk,
    UploadResponse,
)
from app.generation.llm import generate, generate_stream
from app.ingestion.pipeline import ingest_file
from app.retrieval.retriever import retrieve
from app.vectorstore.store import delete_document, get_collection, list_documents
from config import settings

router = APIRouter()


# ── POST /upload ───────────────────────────────────────────────────────────────

@router.post("/upload", response_model=UploadResponse, tags=["Ingestion"])
async def upload_document(file: UploadFile = File(...)):
    """
    Upload a PDF, DOCX, or TXT file. The file is chunked, embedded,
    and stored in ChromaDB. Re-uploading the same filename overwrites
    the previous version.
    """
    suffix = Path(file.filename).suffix.lower()
    if suffix not in settings.allowed_extensions:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type '{suffix}'. Allowed: {list(settings.allowed_extensions)}",
        )

    # Save upload to disk (sync I/O — run in thread pool)
    dest = settings.upload_dir / file.filename

    def _save():
        with dest.open("wb") as f:
            shutil.copyfileobj(file.file, f)

    await asyncio.to_thread(_save)

    # Run ingestion pipeline in thread pool (CPU + sync I/O bound)
    try:
        result = await asyncio.to_thread(ingest_file, dest)
    except Exception as exc:
        dest.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=str(exc))

    return UploadResponse(**result)


# ── POST /query ────────────────────────────────────────────────────────────────

@router.post("/query", response_model=QueryResponse, tags=["Query"])
async def query_documents(body: QueryRequest):
    """
    Ask a question. Retrieves the top-k relevant chunks from ChromaDB,
    then generates a grounded answer via the local LLM.
    """
    # retrieve() calls embed_single (sync httpx) + ChromaDB — run in thread pool
    hits = await asyncio.to_thread(retrieve, body.question, body.top_k, body.doc_id)

    if not hits:
        return QueryResponse(
            question=body.question,
            answer="No relevant documents found. Please upload some documents first.",
            sources=[],
        )

    # generate() is now async — await directly
    try:
        answer = await generate(body.question, hits)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"LLM error: {exc}")

    sources = [
        SourceChunk(
            source=h["source"],
            page=h["page"],
            chunk_index=h["chunk_index"],
            text=h["text"],
            score=h["score"],
            chunk_total=h.get("chunk_total"),
            char_start=h.get("char_start"),
            char_end=h.get("char_end"),
            content_type=h.get("content_type", "text"),
            section_heading=h.get("section_heading", ""),
        )
        for h in hits
    ]

    return QueryResponse(question=body.question, answer=answer, sources=sources)


# ── POST /query/stream ─────────────────────────────────────────────────────────

@router.post("/query/stream", tags=["Query"])
async def query_stream(body: QueryRequest):
    """
    Streaming version of /query. Tokens arrive as plain text as the LLM generates.
    """
    hits = await asyncio.to_thread(retrieve, body.question, body.top_k, body.doc_id)
    if not hits:
        async def _empty_gen():
            yield "No relevant documents found. Please upload some documents first."
        return StreamingResponse(_empty_gen(), media_type="text/plain")

    return StreamingResponse(
        generate_stream(body.question, hits),
        media_type="text/plain",
    )


# ── GET /documents ─────────────────────────────────────────────────────────────

@router.get("/documents", response_model=DocumentListResponse, tags=["Documents"])
async def list_all_documents():
    """List all ingested documents with their chunk counts."""
    docs = await asyncio.to_thread(list_documents)
    return DocumentListResponse(
        documents=[DocumentInfo(**d) for d in docs],
        total=len(docs),
    )


# ── DELETE /documents/{doc_id} ─────────────────────────────────────────────────

@router.delete("/documents/{doc_id}", response_model=DeleteResponse, tags=["Documents"])
async def delete_document_by_id(doc_id: str):
    """Remove all chunks for a document from ChromaDB."""
    deleted = await asyncio.to_thread(delete_document, doc_id)
    if deleted == 0:
        raise HTTPException(status_code=404, detail=f"No document found with id '{doc_id}'.")
    return DeleteResponse(
        doc_id=doc_id,
        chunks_deleted=deleted,
        message=f"Deleted {deleted} chunks.",
    )


# ── GET /health ────────────────────────────────────────────────────────────────

@router.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Quick health check — confirms settings and total chunk count."""
    collection = await asyncio.to_thread(get_collection)
    total = await asyncio.to_thread(collection.count)
    return HealthResponse(
        status="ok",
        ollama_url=settings.ollama_base_url,
        llm_model=settings.llm_model,
        embed_model=settings.embed_model,
        total_chunks=total,
    )