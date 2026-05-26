# SecondMind 

A fully **local** Retrieval-Augmented Generation (RAG) API. Upload your documents and ask questions — every step runs on your own machine. No cloud, no API keys.

```
Upload PDF / DOCX / TXT  →  Chunk  →  Embed  →  Store in ChromaDB
                                                        ↓
                              Answer  ←  LLM  ←  Retrieve top-k chunks
```

---

## Tech stack

| Layer | Tool |
|---|---|
| API | FastAPI |
| Vector store | ChromaDB (persisted to disk) |
| Embeddings | `nomic-embed-text` via Ollama |
| LLM | `llama3.1:8b` via Ollama |
| PDF parsing | PyMuPDF |
| DOCX parsing | python-docx |
| Package manager | [uv](https://docs.astral.sh/uv/) |

---

## Prerequisites

| Requirement | Notes |
|---|---|
| Python 3.12+ | Checked via `.python-version` |
| [Ollama](https://ollama.com) | Must be running before you start the server |
| [uv](https://docs.astral.sh/uv/getting-started/installation/) | Fast Python package manager |

> **Models are pulled automatically.** When the server starts for the first time it checks whether `llama3.1:8b` and `nomic-embed-text` are already present locally and pulls them if not. You do not need to run `ollama pull` manually.

---

## Quickstart

### 1. Clone the repo

```bash
git clone https://github.com/your-username/SecondMind.git
cd SecondMind
```

### 2. Install dependencies

```bash
uv sync
```

### 3. Configure environment

```bash
cp .env.example .env   # then open .env and adjust if needed
```

The defaults work out of the box — you only need to edit `.env` if you want to use a different model or change chunking behaviour.

### 4. Start Ollama

Make sure the Ollama app is running. On macOS/Windows it runs as a background service after installation. On Linux:

```bash
ollama serve
```

### 5. Start the API

```bash
uv run uvicorn main:app --reload --port 8000
```

On first run you will see the models being pulled:

```
12:00:01  INFO  __main__ — Pulling 'llama3.1:8b' — this may take several minutes on first run…
12:00:03  INFO  __main__   · llama3.1:8b: pulling manifest
12:00:05  INFO  __main__   ↓ llama3.1:8b: 5%  (234,123,456 / 4,661,211,136 bytes)
...
12:04:10  INFO  __main__ ✓ 'llama3.1:8b' is ready
```

Subsequent starts skip the pull entirely.

### 6. Open the docs

Visit **http://localhost:8000/docs** for the interactive Swagger UI.

---

## API reference

### Upload a document

```bash
curl -X POST http://localhost:8000/api/v1/upload \
  -F "file=@your_document.pdf"
```

```json
{
  "doc_id": "a3f2c1d4e5b6",
  "filename": "your_document.pdf",
  "chunk_count": 42,
  "message": "Document ingested successfully."
}
```

Supported formats: `.pdf`, `.docx`, `.txt`

---

### Ask a question

```bash
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the main argument of the document?", "top_k": 5}'
```

```json
{
  "question": "What is the main argument of the document?",
  "answer": "According to the document...",
  "sources": [
    {
      "source": "your_document.pdf",
      "page": 3,
      "chunk_index": 12,
      "text": "...relevant excerpt...",
      "score": 0.87
    }
  ]
}
```

---

### Streaming query

Tokens stream back as they are generated — useful for long answers.

```bash
curl -X POST http://localhost:8000/api/v1/query/stream \
  -H "Content-Type: application/json" \
  -d '{"question": "Summarise the key findings."}' \
  --no-buffer
```

---

### Other endpoints

```bash
# List all ingested documents
curl http://localhost:8000/api/v1/documents

# Delete a document (and all its chunks)
curl -X DELETE http://localhost:8000/api/v1/documents/{doc_id}

# Health check
curl http://localhost:8000/api/v1/health
```

---

## Configuration

All settings are in `.env`. The table below shows every available variable with its default value.

| Variable | Default | Description |
|---|---|---|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `LLM_MODEL` | `llama3.1:8b` | Generation model — any Ollama-compatible tag |
| `EMBED_MODEL` | `nomic-embed-text` | Embedding model |
| `EMBED_USE_BATCH` | `true` | Use `/api/embed` batch endpoint (Ollama ≥ 0.1.34) |
| `EMBED_BATCH_SIZE` | `32` | Chunks per batch request |
| `EMBED_CONCURRENCY` | `8` | Parallel workers when batch is disabled |
| `CHUNK_SIZE` | `512` | Characters per chunk |
| `CHUNK_OVERLAP` | `64` | Overlap between consecutive chunks |
| `TOP_K` | `5` | Chunks retrieved per query |
| `SIMILARITY_THRESHOLD` | `0.0` | Min similarity score to include a chunk (0 = off) |
| `TEMPERATURE` | `0.2` | LLM sampling temperature (0 = factual, 1 = creative) |
| `MAX_TOKENS` | `1024` | Max tokens the LLM may generate |
| `SYSTEM_PROMPT` | *(see config.py)* | System prompt injected before every conversation |
| `CHROMA_COLLECTION` | `documents` | ChromaDB collection name |

### Swapping models

Change `LLM_MODEL` or `EMBED_MODEL` in `.env` to any model available on [Ollama's model library](https://ollama.com/library). The new model will be pulled automatically on next startup.

Some alternatives:

| Use case | Model |
|---|---|
| Lower VRAM (LLM) | `phi3:mini`, `mistral:7b` |
| Higher quality (LLM) | `llama3.1:70b` |
| Embeddings | `mxbai-embed-large`, `all-minilm` |

---

## Project structure

```
SecondMind/
├── main.py                  # FastAPI app + auto-pull lifespan
├── config.py                # All settings (pydantic-settings)
├── .env                     # Your local config (not committed)
├── .env.example             # Template — safe to commit
├── pyproject.toml           # Dependencies (managed by uv)
├── app/
│   ├── api/
│   │   ├── routes.py        # All API endpoints
│   │   └── schemas.py       # Pydantic request/response models
│   ├── ingestion/
│   │   ├── loader.py        # PDF / DOCX / TXT → plain text
│   │   ├── chunker.py       # Sliding-window text splitter
│   │   ├── embedder.py      # Ollama embedding (batch + parallel)
│   │   └── pipeline.py      # Orchestrates the full ingestion flow
│   ├── retrieval/
│   │   └── retriever.py     # Embeds query + searches ChromaDB
│   ├── generation/
│   │   ├── prompt.py        # Builds context-stuffed prompt
│   │   └── llm.py           # Ollama chat (blocking + streaming)
│   └── vectorstore/
│       └── store.py         # ChromaDB client + CRUD helpers
├── data/
│   ├── uploads/             # Uploaded files saved here
│   └── chroma_db/           # Persisted vector index
└── tests/
    ├── test_api.py
    ├── test_embedder.py
    ├── test_ingestion.py
    └── test_retrieval.py
```

---

## Running tests

```bash
uv run pytest tests/ -v
```

---

## Troubleshooting

**`Connection refused` on startup**
Ollama is not running. Start it with `ollama serve` (Linux) or open the Ollama app (macOS/Windows).

**Model pull hangs or times out**
Large models (e.g. `llama3.1:8b` is ~4.7 GB) take time on first pull. Watch the logs — progress is printed every ~5%. If it fails mid-way, just restart the server; Ollama resumes partial downloads.

**`415 Unsupported Media Type` on upload**
Only `.pdf`, `.docx`, and `.txt` are accepted. Check your file extension.

**Answers say "no relevant documents found"**
Upload at least one document before querying. Use `GET /api/v1/documents` to confirm ingestion succeeded.

---

## .gitignore reminder

Make sure your `.gitignore` includes:

```
.env
data/uploads/
data/chroma_db/
```

Never commit `.env` — it may contain sensitive values. Commit `.env.example` instead.