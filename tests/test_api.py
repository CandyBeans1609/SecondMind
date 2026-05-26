import io
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from main import app

client = TestClient(app)


# ── Health ─────────────────────────────────────────────────────────────────────

def test_health_check():
    with patch("app.api.routes.get_collection") as mock_col:
        mock_col.return_value.count.return_value = 42
        response = client.get("/api/v1/health")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["total_chunks"] == 42


# ── Upload ─────────────────────────────────────────────────────────────────────

def test_upload_unsupported_type():
    """CSV file should be rejected with 415."""
    response = client.post(
        "/api/v1/upload",
        files={"file": ("data.csv", io.BytesIO(b"a,b,c"), "text/csv")},
    )
    assert response.status_code == 415


def test_upload_txt_success():
    """Valid TXT upload runs the ingestion pipeline and returns chunk info."""
    fake_result = {"doc_id": "abc123", "filename": "test.txt", "chunk_count": 7}

    with patch("app.api.routes.ingest_file", return_value=fake_result):
        response = client.post(
            "/api/v1/upload",
            files={"file": ("test.txt", io.BytesIO(b"Hello world " * 100), "text/plain")},
        )

    assert response.status_code == 200
    data = response.json()
    assert data["doc_id"] == "abc123"
    assert data["chunk_count"] == 7


# ── Query ──────────────────────────────────────────────────────────────────────

def test_query_no_documents():
    """Query with empty retrieval returns a helpful message."""
    with patch("app.api.routes.retrieve", return_value=[]):
        response = client.post(
            "/api/v1/query",
            json={"question": "What is this about?"},
        )

    assert response.status_code == 200
    data = response.json()
    assert "upload" in data["answer"].lower()
    assert data["sources"] == []


def test_query_with_results():
    """Query returns answer and sources when retrieval succeeds."""
    fake_hits = [
        {
            "text": "The sky is blue.",
            "source": "science.pdf",
            "page": 2,
            "chunk_index": 5,
            "score": 0.91,
        }
    ]

    with patch("app.api.routes.retrieve", return_value=fake_hits), \
         patch("app.api.routes.generate", return_value="The sky is blue because of Rayleigh scattering."):

        response = client.post(
            "/api/v1/query",
            json={"question": "Why is the sky blue?"},
        )

    assert response.status_code == 200
    data = response.json()
    assert "Rayleigh" in data["answer"]
    assert len(data["sources"]) == 1
    assert data["sources"][0]["source"] == "science.pdf"


# ── Documents ──────────────────────────────────────────────────────────────────

def test_list_documents():
    fake_docs = [
        {"doc_id": "abc", "filename": "a.pdf", "chunk_count": 10},
        {"doc_id": "def", "filename": "b.txt", "chunk_count": 5},
    ]
    with patch("app.api.routes.list_documents", return_value=fake_docs):
        response = client.get("/api/v1/documents")

    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 2


def test_delete_nonexistent_document():
    with patch("app.api.routes.delete_document", return_value=0):
        response = client.delete("/api/v1/documents/doesnotexist")
    assert response.status_code == 404