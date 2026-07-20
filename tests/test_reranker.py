

from unittest.mock import patch

from retrievers.reranker import rerank


# ==========================================================
# Empty Input
# ==========================================================

def test_empty_chunks():

    results = rerank(
        "What is FAISS?",
        [],
    )

    assert results == []


# ==========================================================
# Successful Reranking
# ==========================================================

@patch("retrievers.reranker.model.predict")
def test_rerank(mock_predict):

    mock_predict.return_value = [
        0.9,
        0.4,
        0.7,
    ]

    chunks = [
        {
            "id": 1,
            "text": "Chunk A",
        },
        {
            "id": 2,
            "text": "Chunk B",
        },
        {
            "id": 3,
            "text": "Chunk C",
        },
    ]

    results = rerank(
        "FAISS",
        chunks,
        top_k=2,
    )

    assert len(results) == 2

    # Highest score first
    assert results[0]["id"] == 1
    assert results[1]["id"] == 3

    assert results[0]["rerank_score"] == 0.9
    assert results[1]["rerank_score"] == 0.7


# ==========================================================
# Confidence
# ==========================================================

@patch("retrievers.reranker.model.predict")
def test_confidence(mock_predict):

    mock_predict.return_value = [
        0.95,
        0.60,
    ]

    chunks = [
        {
            "id": 1,
            "text": "Chunk A",
        },
        {
            "id": 2,
            "text": "Chunk B",
        },
    ]

    results = rerank(
        "Query",
        chunks,
    )

    for chunk in results:

        assert chunk["confidence"] == 0.95


# ==========================================================
# Top-K
# ==========================================================

@patch("retrievers.reranker.model.predict")
def test_top_k(mock_predict):

    mock_predict.return_value = [
        0.9,
        0.8,
        0.7,
        0.6,
    ]

    chunks = [
        {"id": 1, "text": "A"},
        {"id": 2, "text": "B"},
        {"id": 3, "text": "C"},
        {"id": 4, "text": "D"},
    ]

    results = rerank(
        "Query",
        chunks,
        top_k=3,
    )

    assert len(results) == 3