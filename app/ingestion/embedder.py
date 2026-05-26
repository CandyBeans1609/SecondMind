import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

import httpx

from config import settings

logger = logging.getLogger(__name__)


# ── Public API ─────────────────────────────────────────────────────────────────

def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Embed a list of strings via Ollama.

    Strategy (auto-detected on first call):
      1. Batch  — POST /api/embed with {"input": [...]}  (Ollama ≥ 0.1.34)
                  One HTTP round-trip for the whole list.
      2. Parallel — POST /api/embeddings one request per chunk, up to
                    EMBED_CONCURRENCY concurrent requests.
                  Falls back to this when the batch endpoint is unavailable.
    """
    if not texts:
        return []

    if settings.embed_use_batch:
        return _embed_batch(texts)
    else:
        return _embed_parallel(texts)


def embed_single(text: str) -> list[float]:
    """Convenience wrapper for embedding a single query string."""
    return embed_texts([text])[0]


# ── Strategy 1: batch endpoint (/api/embed) ────────────────────────────────────

def _embed_batch(texts: list[str]) -> list[list[float]]:
    """
    Single HTTP call using Ollama's /api/embed endpoint.
    Ollama processes the whole list server-side and returns one vector per item.

    Falls back to parallel if the endpoint returns 404 (older Ollama build),
    and disables the batch strategy for the rest of the process lifetime.
    """
    total = len(texts)
    logger.info(f"      Embedding {total} chunk(s) via batch /api/embed...")

    # Process in batches to avoid oversized payloads on very large documents
    batch_size = settings.embed_batch_size
    all_embeddings: list[list[float]] = []

    with httpx.Client(timeout=120.0) as client:
        for start in range(0, total, batch_size):
            batch = texts[start : start + batch_size]
            end = min(start + batch_size, total)
            logger.info(f"      → chunks {start + 1}–{end} / {total}")

            try:
                response = client.post(
                    f"{settings.ollama_base_url}/api/embed",
                    json={"model": settings.embed_model, "input": batch},
                )
                # 404 means this Ollama build doesn't have /api/embed yet
                if response.status_code == 404:
                    logger.warning(
                        "Ollama /api/embed not found (older version). "
                        "Falling back to parallel /api/embeddings. "
                        "Set EMBED_USE_BATCH=false in .env to suppress this warning."
                    )
                    settings.embed_use_batch = False  # disable for subsequent calls
                    already_done = all_embeddings
                    remaining = texts[start:]
                    return already_done + _embed_parallel(remaining)

                response.raise_for_status()
                data = response.json()
                all_embeddings.extend(data["embeddings"])

            except httpx.HTTPStatusError as exc:
                raise RuntimeError(
                    f"Ollama batch embedding failed (HTTP {exc.response.status_code}): "
                    f"{exc.response.text}"
                ) from exc

    return all_embeddings


# ── Strategy 2: parallel /api/embeddings ──────────────────────────────────────

def _embed_parallel(texts: list[str]) -> list[list[float]]:
    """
    Embed texts using concurrent requests to /api/embeddings.
    Uses a ThreadPoolExecutor so we can fire EMBED_CONCURRENCY requests at once
    without switching to async (this function is called from a thread pool itself).

    Results are re-ordered to match the original text order.
    """
    total = len(texts)
    concurrency = settings.embed_concurrency
    logger.info(
        f"      Embedding {total} chunk(s) in parallel "
        f"(concurrency={concurrency}) via /api/embeddings..."
    )

    # Pre-allocate result list so we can fill by index regardless of completion order
    embeddings: list[list[float] | None] = [None] * total

    def _single(index: int, text: str) -> tuple[int, list[float]]:
        with httpx.Client(timeout=60.0) as client:
            response = client.post(
                f"{settings.ollama_base_url}/api/embeddings",
                json={"model": settings.embed_model, "prompt": text},
            )
            response.raise_for_status()
            return index, response.json()["embedding"]

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {pool.submit(_single, i, t): i for i, t in enumerate(texts)}
        completed = 0
        for future in as_completed(futures):
            idx, vec = future.result()   # re-raises on exception
            embeddings[idx] = vec
            completed += 1
            if completed % 10 == 0 or completed == total:
                logger.info(f"      Embedded {completed}/{total}...")

    return embeddings  # type: ignore[return-value]  # all slots filled