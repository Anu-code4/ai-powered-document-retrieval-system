

from unittest.mock import patch

from llm.query_rewriter import (
    needs_rewriting,
    rewrite_query,
)


# ---------------------------------------------------------
# needs_rewriting()
# ---------------------------------------------------------

def test_no_history():
    assert needs_rewriting(
        "What is FAISS?",
        None,
    ) is False


def test_standalone_question():
    history = [
        {
            "role": "user",
            "content": "Explain FAISS."
        }
    ]

    assert needs_rewriting(
        "What is BM25?",
        history,
    ) is False


def test_follow_up_question():
    history = [
        {
            "role": "user",
            "content": "Explain FAISS."
        }
    ]

    assert needs_rewriting(
        "What are its advantages?",
        history,
    ) is True


# ---------------------------------------------------------
# rewrite_query()
# ---------------------------------------------------------

def test_no_rewrite_needed():

    question = "What is BM25?"

    assert rewrite_query(
        question,
        [],
    ) == question


@patch("llm.query_rewriter.chat")
def test_successful_rewrite(mock_chat):

    mock_chat.return_value = {
        "message": {
            "content": "What are the advantages of FAISS?"
        }
    }

    history = [
        {
            "role": "user",
            "content": "Explain FAISS."
        }
    ]

    rewritten = rewrite_query(
        "What are its advantages?",
        history,
    )

    assert rewritten == "What are the advantages of FAISS?"


@patch("llm.query_rewriter.chat")
def test_chat_exception(mock_chat):

    mock_chat.side_effect = Exception("LLM Error")

    history = [
        {
            "role": "user",
            "content": "Explain FAISS."
        }
    ]

    question = "What are its advantages?"

    assert rewrite_query(
        question,
        history,
    ) == question