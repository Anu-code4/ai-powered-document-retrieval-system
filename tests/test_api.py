

from unittest.mock import patch

from fastapi.testclient import TestClient

from api import app

client = TestClient(app)


# ==========================================================
# Home Route
# ==========================================================

def test_home():

    response = client.get("/")

    assert response.status_code == 200
    assert response.json() == {
        "message": "AI Revision Companion API is running."
    }


# ==========================================================
# Chat Endpoint
# ==========================================================

@patch("api.get_answer")
def test_chat(mock_get_answer):

    mock_get_answer.return_value = {
        "answer": "FAISS is a vector database.",
        "confidence": "High",
        "sources": [
            {
                "document": "notes.pdf",
                "chunks": [1, 2]
            }
        ],
    }

    response = client.post(
        "/chat",
        json={
            "question": "What is FAISS?"
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["answer"] == "FAISS is a vector database."
    assert data["confidence"] == "High"
    assert len(data["sources"]) == 1


# ==========================================================
# Validation Error
# ==========================================================

def test_chat_validation():

    response = client.post(
        "/chat",
        json={},
    )

    assert response.status_code == 422


# ==========================================================
# Streaming Endpoint
# ==========================================================

@patch("api.stream_response")
def test_stream(mock_stream):

    mock_stream.return_value = iter([
        "Hello ",
        "World",
    ])

    response = client.post(
        "/stream",
        json={
            "question": "Hi"
        },
    )

    assert response.status_code == 200
    assert response.text == "Hello World"


# ==========================================================
# Stream Validation
# ==========================================================

def test_stream_validation():

    response = client.post(
        "/stream",
        json={},
    )

    assert response.status_code == 422