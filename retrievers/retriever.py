

"""
Semantic Retriever using FAISS.

Converts a query into an embedding, searches the FAISS
index and returns the most relevant chunks.
"""

import json
import logging

import faiss
from sentence_transformers import SentenceTransformer

from config import FAISS_DISTANCE_THRESHOLD
from utils.logger import setup_logger

# ==========================================================
# Logger
# ==========================================================

setup_logger()
logger = logging.getLogger(__name__)

# ==========================================================
# Paths
# ==========================================================

EMBEDDED_CHUNKS_PATH = "embedded_chunks.json"
FAISS_INDEX_PATH = "faiss_index.bin"

# ==========================================================
# Load Model
# ==========================================================

logger.info("Loading SentenceTransformer model...")

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

# ==========================================================
# Load Embedded Chunks
# ==========================================================

logger.info("Loading embedded chunks...")

with open(
    EMBEDDED_CHUNKS_PATH,
    "r",
    encoding="utf-8",
) as file:
    embedded_chunks = json.load(file)

# ==========================================================
# Load FAISS Index
# ==========================================================

logger.info("Loading FAISS index...")

index = faiss.read_index(FAISS_INDEX_PATH)

logger.info("Retriever initialized successfully.")

# ==========================================================
# FAISS Retriever
# ==========================================================

def retrieve_faiss(
    query: str,
    top_k: int = 3,
    threshold: float = FAISS_DISTANCE_THRESHOLD,
    filters: dict | None = None,
) -> list:

    logger.info(f"Searching FAISS for query: {query}")

    # ------------------------------------------------------
    # Encode Query
    # ------------------------------------------------------

    query_embedding = model.encode([query])

    # ------------------------------------------------------
    # Search
    # ------------------------------------------------------

    distances, indices = index.search(
        query_embedding,
        top_k,
    )

    results = []

    # ------------------------------------------------------
    # Process Results
    # ------------------------------------------------------

    for rank, (distance, idx) in enumerate(
        zip(distances[0], indices[0]),
        start=1,
    ):

        if idx == -1:
            continue

        if (
            threshold is not None
            and distance > threshold
        ):
            continue

        chunk = embedded_chunks[idx]

        # --------------------------------------------------
        # Metadata Filtering
        # --------------------------------------------------

        if filters:

            if (
                "file_type" in filters
                and chunk["file_type"] != filters["file_type"]
            ):
                continue

            if (
                "source" in filters
                and chunk["source"].lower() != filters["source"].lower()
            ):
                continue

        results.append(
            {
                "id": chunk["id"],
                "index": idx,
                "rank": rank,
                "retriever": "faiss",
                "distance": float(distance),
                "source": chunk["source"],
                "file_type": chunk["file_type"],
                "text": chunk["text"],
            }
        )

    logger.info(f"FAISS returned {len(results)} chunks.")

    return results


# ==========================================================
# Standalone Test
# ==========================================================

if __name__ == "__main__":

    query = input("Enter your query: ")

    results = retrieve_faiss(query)

    if not results:

        print("\nNo relevant chunks found.")

    else:

        print("\nRetrieved Chunks\n")

        for chunk in results:

            print("-" * 60)
            print(f"Rank      : {chunk['rank']}")
            print(f"Index     : {chunk['index']}")
            print(f"Retriever : {chunk['retriever']}")
            print(f"Document  : {chunk['source']}")
            print(f"Type      : {chunk['file_type']}")
            print(f"Distance  : {chunk['distance']:.4f}")
            print()
            print(chunk["text"])
            print("-" * 60)