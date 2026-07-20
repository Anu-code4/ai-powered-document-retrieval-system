

from retrievers.bm25_retriever import retrieve_bm25


# ==========================================================
# Basic Retrieval
# ==========================================================

def test_returns_list():

    results = retrieve_bm25("Chunking")

    assert isinstance(results, list)


def test_result_structure():

    results = retrieve_bm25("Chunking")

    if results:

        chunk = results[0]

        assert "id" in chunk
        assert "index" in chunk
        assert "rank" in chunk
        assert "retriever" in chunk
        assert "score" in chunk
        assert "source" in chunk
        assert "file_type" in chunk
        assert "text" in chunk


# ==========================================================
# Metadata Filters
# ==========================================================

def test_invalid_source_filter():

    results = retrieve_bm25(
        "Chunking",
        filters={
            "source": "invalid_document.pdf"
        },
    )

    assert results == []


def test_invalid_file_type():

    results = retrieve_bm25(
        "Chunking",
        filters={
            "file_type": "xyz"
        },
    )

    assert results == []


# ==========================================================
# Empty Query
# ==========================================================

def test_empty_query():

    results = retrieve_bm25("")

    assert isinstance(results, list)


# ==========================================================
# Ranking
# ==========================================================

def test_ranking():

    results = retrieve_bm25("Chunking")

    if len(results) > 1:

        assert results[0]["rank"] == 1

        for i in range(1, len(results)):
            assert results[i]["rank"] == i + 1