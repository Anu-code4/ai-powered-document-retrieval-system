

from unittest.mock import patch

from retrievers.hybrid_retriever import hybrid_retriever


# ==========================================================
# Successful Hybrid Retrieval
# ==========================================================

@patch("retrievers.hybrid_retriever.rerank")
@patch("retrievers.hybrid_retriever.retrieve_bm25")
@patch("retrievers.hybrid_retriever.retrieve_faiss")
@patch("retrievers.hybrid_retriever.extract_metadata_filter")
@patch("retrievers.hybrid_retriever.generate_multi_queries")
def test_hybrid_pipeline(
    mock_multi,
    mock_filter,
    mock_faiss,
    mock_bm25,
    mock_rerank,
):

    mock_multi.return_value = [
        "What is FAISS?"
    ]

    mock_filter.return_value = {}

    mock_faiss.return_value = [
        {
            "id": 1,
            "rank": 1,
            "retriever": "faiss",
            "distance": 0.2,
            "text": "Chunk",
            "source": "doc.pdf",
            "file_type": "pdf",
            "index": 0,
        }
    ]

    mock_bm25.return_value = [
        {
            "id": 2,
            "rank": 1,
            "retriever": "bm25",
            "score": 12.4,
            "text": "Chunk2",
            "source": "doc.pdf",
            "file_type": "pdf",
            "index": 1,
        }
    ]

    mock_rerank.return_value = [
        {
            "id": 1,
            "rank": 1,
            "retriever": "faiss",
            "rrf_score": 0.8,
        }
    ]

    results = hybrid_retriever("FAISS")

    assert isinstance(results, list)
    assert len(results) == 1

    mock_multi.assert_called_once()
    mock_filter.assert_called_once()
    mock_faiss.assert_called()
    mock_bm25.assert_called()
    mock_rerank.assert_called_once()


# ==========================================================
# Duplicate Fusion
# ==========================================================

@patch("retrievers.hybrid_retriever.rerank")
@patch("retrievers.hybrid_retriever.retrieve_bm25")
@patch("retrievers.hybrid_retriever.retrieve_faiss")
@patch("retrievers.hybrid_retriever.extract_metadata_filter")
@patch("retrievers.hybrid_retriever.generate_multi_queries")
def test_rrf_duplicate_merge(
    mock_multi,
    mock_filter,
    mock_faiss,
    mock_bm25,
    mock_rerank,
):

    mock_multi.return_value = ["Query"]
    mock_filter.return_value = {}

    duplicate = {
        "id": 1,
        "rank": 1,
        "text": "Chunk",
        "source": "doc.pdf",
        "file_type": "pdf",
        "index": 0,
    }

    mock_faiss.return_value = [
        {
            **duplicate,
            "retriever": "faiss",
            "distance": 0.2,
        }
    ]

    mock_bm25.return_value = [
        {
            **duplicate,
            "retriever": "bm25",
            "score": 9.5,
        }
    ]

    mock_rerank.return_value = [
        {
            "id": 1,
            "rank": 1,
            "rrf_score": 0.9,
        }
    ]

    results = hybrid_retriever("Query")

    assert len(results) == 1