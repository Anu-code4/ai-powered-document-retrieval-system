

from query_router import route_query, QueryType


def test_document_query():
    assert route_query("What is FAISS?") == QueryType.DOCUMENT


def test_chat_query():
    assert route_query("Hello") == QueryType.CHAT


def test_memory_query():
    assert route_query("What did I ask previously?") == QueryType.MEMORY