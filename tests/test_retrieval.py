from unittest.mock import MagicMock, patch

from app.retrieval.retriever import retrieve


def test_retrieve_calls_embed_and_query():
    """retrieve() embeds the query then calls query_collection."""
    fake_embedding = [0.1] * 768
    fake_hits = [
        {
            "id": "abc_0",
            "text": "Relevant passage.",
            "score": 0.92,
            "source": "doc.pdf",
            "page": 1,
            "chunk_index": 0,
        }
    ]

    with patch("app.retrieval.retriever.embed_single", return_value=fake_embedding) as mock_embed, \
         patch("app.retrieval.retriever.query_collection", return_value=fake_hits) as mock_query:

        results = retrieve("What is the main topic?", top_k=3)

        mock_embed.assert_called_once_with("What is the main topic?")
        mock_query.assert_called_once_with(fake_embedding, top_k=3, doc_id=None)
        assert len(results) == 1
        assert results[0]["score"] == 0.92


def test_retrieve_with_doc_id():
    """retrieve() passes doc_id down to query_collection."""
    fake_embedding = [0.1] * 768
    fake_hits = [{"id": "abc_0", "text": "Passage.", "score": 0.92}]

    with patch("app.retrieval.retriever.embed_single", return_value=fake_embedding), \
         patch("app.retrieval.retriever.query_collection", return_value=fake_hits) as mock_query:

        results = retrieve("Question?", top_k=3, doc_id="custom_doc_123")

        mock_query.assert_called_once_with(fake_embedding, top_k=3, doc_id="custom_doc_123")
        assert len(results) == 1


def test_retrieve_empty_collection():
    """retrieve() returns [] when the collection has no documents."""
    with patch("app.retrieval.retriever.embed_single", return_value=[0.0] * 768), \
         patch("app.retrieval.retriever.query_collection", return_value=[]):

        results = retrieve("Any question?")
        assert results == []