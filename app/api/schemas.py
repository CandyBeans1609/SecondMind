from typing import Optional
from pydantic import BaseModel, Field


# ── Upload ─────────────────────────────────────────────────────────────────────

class UploadResponse(BaseModel):
    doc_id: str
    filename: str
    chunk_count: int
    message: str = "Document ingested successfully."


# ── Query ──────────────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, description="The question to answer.")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of chunks to retrieve.")
    doc_id: Optional[str] = Field(default=None, description="Restrict search to a specific document.")


class SourceChunk(BaseModel):
    source: str       # filename
    page: int
    chunk_index: int
    text: str         # the retrieved excerpt
    score: float      # cosine similarity (0–1)
    chunk_total: Optional[int] = None
    char_start: Optional[int] = None
    char_end: Optional[int] = None
    content_type: Optional[str] = "text"
    section_heading: Optional[str] = ""


class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: list[SourceChunk]


# ── Documents ──────────────────────────────────────────────────────────────────

class DocumentInfo(BaseModel):
    doc_id: str
    filename: str
    chunk_count: int


class DocumentListResponse(BaseModel):
    documents: list[DocumentInfo]
    total: int


# ── Delete ─────────────────────────────────────────────────────────────────────

class DeleteResponse(BaseModel):
    doc_id: str
    chunks_deleted: int
    message: str


# ── Health ─────────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str
    ollama_url: str
    llm_model: str
    embed_model: str
    total_chunks: int