from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # ── Ollama ────────────────────────────────────────────────────────────────
    ollama_base_url: str = "http://localhost:11434"
    llm_model: str = "llama3.1:8b"
    embed_model: str = "nomic-embed-text"

    # ── Embedding ────────────────────────────────────────────────────────────
    # embed_use_batch=True  → use /api/embed (Ollama ≥ 0.1.34, one round-trip)
    # embed_use_batch=False → fall back to parallel /api/embeddings calls
    embed_use_batch: bool = True
    embed_batch_size: int = 32    # max chunks per /api/embed request
    embed_concurrency: int = 8    # parallel workers for fallback strategy

    # ── Chunking ──────────────────────────────────────────────────────────────
    chunk_size: int = 512        # characters per chunk
    chunk_overlap: int = 64      # overlap between consecutive chunks

    # ── Retrieval ─────────────────────────────────────────────────────────────
    top_k: int = 5               # number of chunks to retrieve per query
    similarity_threshold: float = 0.0   # min score to include a chunk (0 = off)

    # ── Generation ────────────────────────────────────────────────────────────
    max_tokens: int = 1024
    temperature: float = 0.2     # low = more factual, high = more creative
    system_prompt: str = (
        "You are a helpful assistant that answers questions strictly based on "
        "the provided context. If the context does not contain enough information "
        "to answer the question, say so clearly. Do not make up information."
    )

    # ── Paths ─────────────────────────────────────────────────────────────────
    base_dir: Path = Path(__file__).parent
    upload_dir: Path = base_dir / "data" / "uploads"
    chroma_dir: Path = base_dir / "data" / "chroma_db"

    # ── ChromaDB ──────────────────────────────────────────────────────────────
    chroma_collection: str = "documents"

    # ── Supported file types ──────────────────────────────────────────────────
    allowed_extensions: tuple = (".pdf", ".docx", ".txt")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()

# Ensure directories exist on import
settings.upload_dir.mkdir(parents=True, exist_ok=True)
settings.chroma_dir.mkdir(parents=True, exist_ok=True)