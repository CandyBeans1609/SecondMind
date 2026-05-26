"""
Tests for the batch and parallel embedding strategies in embedder.py.
All Ollama HTTP calls are mocked — no live server required.
"""
from unittest.mock import MagicMock, patch, call

import pytest

from app.ingestion.embedder import embed_texts, embed_single, _embed_batch, _embed_parallel


FAKE_VEC = [0.1, 0.2, 0.3]


# ── embed_texts: routing ───────────────────────────────────────────────────────

def test_embed_texts_empty_returns_empty():
    assert embed_texts([]) == []


def test_embed_texts_routes_to_batch_when_enabled(monkeypatch):
    monkeypatch.setattr("app.ingestion.embedder.settings.embed_use_batch", True)
    with patch("app.ingestion.embedder._embed_batch", return_value=[[0.1]]) as mock_b:
        result = embed_texts(["hello"])
    mock_b.assert_called_once_with(["hello"])
    assert result == [[0.1]]


def test_embed_texts_routes_to_parallel_when_disabled(monkeypatch):
    monkeypatch.setattr("app.ingestion.embedder.settings.embed_use_batch", False)
    with patch("app.ingestion.embedder._embed_parallel", return_value=[[0.2]]) as mock_p:
        result = embed_texts(["hello"])
    mock_p.assert_called_once_with(["hello"])
    assert result == [[0.2]]


# ── _embed_batch ──────────────────────────────────────────────────────────────

def _make_batch_response(vecs):
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {"embeddings": vecs}
    resp.raise_for_status = MagicMock()
    return resp


def test_embed_batch_single_request():
    """All texts fit in one batch → one POST."""
    texts = ["a", "b", "c"]
    vecs = [FAKE_VEC, FAKE_VEC, FAKE_VEC]
    with patch("httpx.Client") as MockClient:
        mock_post = MockClient.return_value.__enter__.return_value.post
        mock_post.return_value = _make_batch_response(vecs)
        result = _embed_batch(texts)
    assert result == vecs
    assert mock_post.call_count == 1


def test_embed_batch_splits_into_pages(monkeypatch):
    """When embed_batch_size=2, three texts → two requests."""
    monkeypatch.setattr("app.ingestion.embedder.settings.embed_batch_size", 2)
    texts = ["a", "b", "c"]
    with patch("httpx.Client") as MockClient:
        mock_post = MockClient.return_value.__enter__.return_value.post
        mock_post.side_effect = [
            _make_batch_response([FAKE_VEC, FAKE_VEC]),
            _make_batch_response([FAKE_VEC]),
        ]
        result = _embed_batch(texts)
    assert len(result) == 3
    assert mock_post.call_count == 2


def test_embed_batch_falls_back_on_404(monkeypatch):
    """404 from /api/embed should trigger parallel fallback."""
    monkeypatch.setattr("app.ingestion.embedder.settings.embed_use_batch", True)
    not_found = MagicMock()
    not_found.status_code = 404

    with patch("httpx.Client") as MockClient, \
         patch("app.ingestion.embedder._embed_parallel", return_value=[[0.9]]) as mock_p:
        MockClient.return_value.__enter__.return_value.post.return_value = not_found
        result = _embed_batch(["hello"])

    mock_p.assert_called_once_with(["hello"])
    assert result == [[0.9]]


# ── _embed_parallel ───────────────────────────────────────────────────────────

def _make_single_response(vec):
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {"embedding": vec}
    resp.raise_for_status = MagicMock()
    return resp


def test_embed_parallel_preserves_order():
    """Results must match input order even when futures complete out of order."""
    vecs = [[float(i)] for i in range(5)]
    texts = [str(i) for i in range(5)]

    with patch("httpx.Client") as MockClient:
        mock_post = MockClient.return_value.__enter__.return_value.post
        mock_post.side_effect = [_make_single_response(v) for v in vecs]
        result = _embed_parallel(texts)

    assert len(result) == 5
    # Each vector was assigned to the right slot
    for i, vec in enumerate(result):
        assert vec == [float(i)]


# ── embed_single ──────────────────────────────────────────────────────────────

def test_embed_single_returns_first_element():
    with patch("app.ingestion.embedder.embed_texts", return_value=[FAKE_VEC]):
        result = embed_single("hello")
    assert result == FAKE_VEC