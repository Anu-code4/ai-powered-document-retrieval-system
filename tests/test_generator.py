

from unittest.mock import patch

from llm.generator import (
    build_prompt,
    generate_answer,
    stream_answer,
)

from query_router import QueryType


# ==========================================================
# build_prompt()
# ==========================================================

def test_build_prompt():

    prompt = build_prompt(
        question="What is FAISS?",
        retrieved_chunks=[
            {
                "text": "FAISS is a vector database."
            }
        ],
        conversation_history=[
            {
                "role": "user",
                "content": "Hello"
            }
        ],
        query_type=QueryType.DOCUMENT,
    )

    assert "FAISS is a vector database." in prompt
    assert "Hello" in prompt
    assert "What is FAISS?" in prompt
    assert "DOCUMENT" in prompt


# ==========================================================
# generate_answer()
# ==========================================================

@patch("llm.generator.chat")
def test_generate_answer_success(mock_chat):

    mock_chat.return_value = {
        "message": {
            "content": "FAISS is a similarity search library."
        }
    }

    chunks = [
        {
            "text": "FAISS text",
            "confidence": 9,
            "source": "doc.pdf",
            "id": 1,
        }
    ]

    response = generate_answer(
        "What is FAISS?",
        chunks,
    )

    assert response["answer"] == "FAISS is a similarity search library."
    assert response["confidence"] == "High"
    assert len(response["sources"]) == 1


def test_generate_answer_no_chunks():

    response = generate_answer(
        "Question",
        [],
    )

    assert response["answer"] == "I don't know based on the provided documents."
    assert response["sources"] == []


@patch("llm.generator.chat")
def test_generate_answer_exception(mock_chat):

    mock_chat.side_effect = Exception("LLM Error")

    response = generate_answer(
        "Question",
        [
            {
                "text": "sample",
                "confidence": 9,
            }
        ],
    )

    assert "Sorry" in response["answer"]


# ==========================================================
# Confidence
# ==========================================================

@patch("llm.generator.chat")
def test_confidence_levels(mock_chat):

    mock_chat.return_value = {
        "message": {
            "content": "Answer"
        }
    }

    high = generate_answer(
        "Q",
        [
            {
                "text": "x",
                "confidence": 9,
            }
        ],
    )

    medium = generate_answer(
        "Q",
        [
            {
                "text": "x",
                "confidence": 6,
            }
        ],
    )

    low = generate_answer(
        "Q",
        [
            {
                "text": "x",
                "confidence": 2,
            }
        ],
    )

    assert high["confidence"] == "High"
    assert medium["confidence"] == "Medium"
    assert low["confidence"] == "Low"


# ==========================================================
# Stream
# ==========================================================

@patch("llm.generator.chat")
def test_stream_answer(mock_chat):

    mock_chat.return_value = iter(
        [
            {
                "message": {
                    "content": "Hello "
                }
            },
            {
                "message": {
                    "content": "World"
                }
            },
        ]
    )

    output = "".join(
        stream_answer(
            "Question",
            [
                {
                    "text": "Chunk"
                }
            ],
        )
    )

    assert output == "Hello World"


def test_stream_no_chunks():

    output = "".join(
        stream_answer(
            "Question",
            [],
        )
    )

    assert "I don't know" in output


@patch("llm.generator.chat")
def test_stream_exception(mock_chat):

    mock_chat.side_effect = Exception("Failure")

    output = "".join(
        stream_answer(
            "Question",
            [
                {
                    "text": "Chunk"
                }
            ],
        )
    )

    assert "Sorry" in output