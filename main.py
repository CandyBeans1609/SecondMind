import json
import logging
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router
from config import settings

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)

logger = logging.getLogger(__name__)


# ── Ollama helpers ─────────────────────────────────────────────────────────────

async def _get_local_models(client: httpx.AsyncClient) -> set[str]:
    """Return the set of model names already present in Ollama."""
    try:
        resp = await client.get(f"{settings.ollama_base_url}/api/tags", timeout=10.0)
        resp.raise_for_status()
        return {m["name"] for m in resp.json().get("models", [])}
    except httpx.HTTPError as exc:
        logger.warning(f"Could not reach Ollama at {settings.ollama_base_url}: {exc}")
        return set()


async def _pull_model(client: httpx.AsyncClient, model: str) -> None:
    """
    Stream-pull a model from Ollama, logging progress lines as they arrive.
    Uses /api/pull with stream=True so we get incremental status updates
    instead of silently blocking until the download is complete.
    """
    logger.info(f"Pulling '{model}' — this may take several minutes on first run…")
    last_status = ""

    async with client.stream(
        "POST",
        f"{settings.ollama_base_url}/api/pull",
        json={"name": model, "stream": True},
        timeout=None,  # large models can take a long time
    ) as response:
        response.raise_for_status()
        async for raw_line in response.aiter_lines():
            if not raw_line:
                continue
            try:
                data = json.loads(raw_line)
            except json.JSONDecodeError:
                continue

            status = data.get("status", "")

            # Log layer-level download progress without flooding the console
            if "pulling" in status or "downloading" in status:
                completed = data.get("completed", 0)
                total = data.get("total", 0)
                if total:
                    pct = completed / total * 100
                    # Only log every ~5 % to keep output readable
                    bucket = int(pct // 5) * 5
                    bucket_key = f"{model}:{bucket}"
                    if bucket_key != getattr(_pull_model, "_last_bucket", ""):
                        _pull_model._last_bucket = bucket_key  # type: ignore[attr-defined]
                        logger.info(f"  ↓ {model}: {pct:.0f}%  ({completed:,} / {total:,} bytes)")
            elif status and status != last_status:
                logger.info(f"  · {model}: {status}")
                last_status = status

            if data.get("error"):
                raise RuntimeError(f"Ollama pull error for '{model}': {data['error']}")

    logger.info(f"✓ '{model}' is ready")


async def _ensure_models(models: list[str]) -> None:
    """Pull any models that are not yet available locally."""
    async with httpx.AsyncClient() as client:
        available = await _get_local_models(client)

        for model in models:
            # Ollama tags can be bare names ("llama3.1:8b") or include a digest —
            # check both the exact name and the name without a sha256 suffix.
            base_name = model.split(":")[0]
            already_have = any(
                m == model or m.startswith(f"{model}:") or m.startswith(f"{base_name}:")
                for m in available
            )

            if already_have:
                logger.info(f"✓ '{model}' already available locally, skipping pull")
            else:
                await _pull_model(client, model)


# ── Startup / Shutdown ─────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    await _ensure_models([settings.llm_model, settings.embed_model])
    yield  # server runs here


# ── App ────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Local RAG API",
    description=(
        "A fully local Retrieval-Augmented Generation system. "
        "Upload PDF, DOCX, or TXT documents and query them using a local LLM via Ollama."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# ── CORS ───────────────────────────────────────────────────────────────────────

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routes ─────────────────────────────────────────────────────────────────────

app.include_router(router, prefix="/api/v1")


@app.get("/", tags=["System"])
async def root():
    return {
        "message": "Local RAG API is running.",
        "docs": "/docs",
        "models": {
            "llm": settings.llm_model,
            "embeddings": settings.embed_model,
        },
    }