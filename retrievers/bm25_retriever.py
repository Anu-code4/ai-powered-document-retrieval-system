

"""
BM25 Retriever

Retrieves the most relevant chunks using BM25.
The BM25 index is built once when the module is imported.
"""

import json
import logging

import numpy as np
from rank_bm25 import BM25Okapi

from config import TOP_K
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
# Build BM25 Index
# ==========================================================

texts = [
    chunk["text"]
    for chunk in embedded_chunks
]

logger.info("Tokenizing corpus...")

tokenized_corpus = [
    text.lower().split()
    for text in texts
]

logger.info("Building BM25 index...")

bm25 = BM25Okapi(tokenized_corpus)

logger.info("BM25 Retriever initialized successfully.")

# ==========================================================
# BM25 Retrieval
# ==========================================================

def retrieve_bm25(
    query: str,
    top_k: int = TOP_K,
    filters: dict | None = None,
) -> list:

    logger.info(f"Searching BM25 for query: {query}")

    tokenized_query = query.lower().split()

    scores = bm25.get_scores(tokenized_query)

    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []

    # ------------------------------------------------------
    # Process Results
    # ------------------------------------------------------

    for rank, idx in enumerate(
        top_indices,
        start=1,
    ):

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
                "index": int(idx),
                "rank": rank,
                "retriever": "bm25",
                "score": float(scores[idx]),
                "source": chunk["source"],
                "file_type": chunk["file_type"],
                "text": chunk["text"],
            }
        )

    logger.info(f"BM25 returned {len(results)} chunks.")

    return results


# ==========================================================
# Standalone Test
# ==========================================================

if __name__ == "__main__":

    query = input("Enter your query: ")

    results = retrieve_bm25(query)

    if not results:

        print("\nNo relevant chunks found.")

    else:

        print("\nRetrieved Chunks\n")

        for chunk in results:

            print("-" * 60)
            print(f"Rank      : {chunk['rank']}")
            print(f"Index     : {chunk['index']}")
            print(f"Source    : {chunk['source']}")
            print(f"Score     : {chunk['score']:.4f}")
            print()
            print(chunk["text"])
            print("-" * 60)