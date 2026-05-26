import json
from collections.abc import AsyncGenerator

import httpx

from app.generation.prompt import build_messages
from config import settings


async def generate(question: str, chunks: list[dict]) -> str:
    """
    Async non-streaming generation. Sends the full prompt to Ollama and
    returns the complete answer string.
    """
    messages = build_messages(question, chunks)

    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.post(
            f"{settings.ollama_base_url}/api/chat",
            json={
                "model": settings.llm_model,
                "messages": messages,
                "stream": False,
                "options": {
                    "temperature": settings.temperature,
                    "num_predict": settings.max_tokens,
                },
            },
        )
        response.raise_for_status()

    data = response.json()
    return data["message"]["content"].strip()


async def generate_stream(question: str, chunks: list[dict]) -> AsyncGenerator[str, None]:
    """
    Async streaming generation — yields text tokens as they arrive from Ollama.
    Use with FastAPI's StreamingResponse.
    """
    messages = build_messages(question, chunks)

    async with httpx.AsyncClient(timeout=120.0) as client:
        async with client.stream(
            "POST",
            f"{settings.ollama_base_url}/api/chat",
            json={
                "model": settings.llm_model,
                "messages": messages,
                "stream": True,
                "options": {
                    "temperature": settings.temperature,
                    "num_predict": settings.max_tokens,
                },
            },
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line:
                    continue
                try:
                    chunk = json.loads(line)
                    token = chunk.get("message", {}).get("content", "")
                    if token:
                        yield token
                    if chunk.get("done"):
                        break
                except json.JSONDecodeError:
                    continue