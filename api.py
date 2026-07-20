

"""
FastAPI entry point for AI Revision Companion.
"""

from typing import List

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app import get_answer, stream_response


app = FastAPI(
    title="AI Revision Companion",
    version="1.0.0",
    description="Production RAG API",
)


# ==========================================================
# Request Model
# ==========================================================

class ChatRequest(BaseModel):
    question: str


# ==========================================================
# Response Models
# ==========================================================

class Source(BaseModel):
    document: str
    chunks: List[int]


class ChatResponse(BaseModel):
    answer: str
    confidence: str | None = None
    sources: List[Source] = []


# ==========================================================
# Routes
# ==========================================================

@app.get("/")
def home():

    return {
        "message": "AI Revision Companion API is running."
    }


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):

    result = get_answer(request.question)

    return ChatResponse(
        answer=result["answer"],
        confidence=result["confidence"],
        sources=result["sources"],
    )


# ==========================================================
# Streaming Route
# ==========================================================

@app.post("/stream")
def stream(request: ChatRequest):

    return StreamingResponse(
        stream_response(request.question),
        media_type="text/plain",
    )