

from retrievers.retriever import retrieve_faiss


# ==========================================================
# Basic Retrieval
# ==========================================================

def test_returns_list():

    results = retrieve_faiss("Chunking")

    assert isinstance(results, list)


def test_result_structure():

    results = retrieve_faiss("Chunking")

    if results:

        chunk = results[0]

        assert "id" in chunk
        assert "index" in chunk
        assert "rank" in chunk
        assert "retriever" in chunk
        assert "distance" in chunk
        assert "source" in chunk
        assert "file_type" in chunk
        assert "text" in chunk


# ==========================================================
# Threshold
# ==========================================================

def test_strict_threshold():

    results = retrieve_faiss(
        "Chunking",
        threshold=0.0,
    )

    assert results == []


# ==========================================================
# Metadata Filters
# ==========================================================

def test_invalid_source_filter():

    results = retrieve_faiss(
        "Chunking",
        filters={
            "source": "this_file_does_not_exist.pdf"
        },
    )

    assert results == []


def test_invalid_file_type():

    results = retrieve_faiss(
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

    results = retrieve_faiss("")

    assert isinstance(results, list)