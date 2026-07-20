

"""
Hybrid Retriever

Combines FAISS and BM25 using
Reciprocal Rank Fusion (RRF).
Supports Multi-Query Retrieval.
"""

import logging

import utils.logger

from config import TOP_K, RRF_K, MAX_RERANK_CANDIDATES

from .retriever import retrieve_faiss
from .bm25_retriever import retrieve_bm25
from .reranker import rerank
from .metadata_filter import extract_metadata_filter

from llm.multi_query import generate_multi_queries


# ==========================================================
# Logger
# ==========================================================

utils.logger.setup_logger()
logger = logging.getLogger(__name__)

logger.info("Hybrid Retriever initialized successfully.")


# ==========================================================
# Hybrid Retriever
# ==========================================================

def hybrid_retriever(
    query: str,
    top_k: int = TOP_K,
) -> list:

    logger.info(f"Running Hybrid Retriever for query: {query}")

    # ------------------------------------------------------
    # Generate Multiple Queries
    # ------------------------------------------------------

    queries = generate_multi_queries(query)

    logger.info(f"Generated {len(queries)} search queries.")

    # ------------------------------------------------------
    # Extract Metadata Filters
    # ------------------------------------------------------

    filters = extract_metadata_filter(query)

    logger.info(f"Metadata Filters: {filters}")

    # ------------------------------------------------------
    # Retrieve Documents
    # ------------------------------------------------------

    fused_results = {}

    for q in queries:

        logger.info(f"Retrieving for query: {q}")

        faiss_results = retrieve_faiss(
            query=q,
            top_k=20,
            filters=filters,
        )

        bm25_results = retrieve_bm25(
            query=q,
            top_k=20,
            filters=filters,
        )

        logger.info(
            f"FAISS={len(faiss_results)} | BM25={len(bm25_results)}"
        )

        # --------------------------------------------------
        # Reciprocal Rank Fusion
        # --------------------------------------------------

        for result in faiss_results + bm25_results:

            chunk_id = result["id"]

            rrf_score = 1 / (RRF_K + result["rank"])

            if chunk_id not in fused_results:

                fused_results[chunk_id] = {
                    **result,
                    "rrf_score": rrf_score,
                }

            else:

                fused_results[chunk_id]["rrf_score"] += rrf_score

                if result["retriever"] == "faiss":
                    fused_results[chunk_id]["distance"] = result.get("distance")

                elif result["retriever"] == "bm25":
                    fused_results[chunk_id]["score"] = result.get("score")

    logger.info(f"Unique chunks after fusion: {len(fused_results)}")

    # ------------------------------------------------------
    # Sort by RRF Score
    # ------------------------------------------------------

    results = sorted(
        fused_results.values(),
        key=lambda x: x["rrf_score"],
        reverse=True,
    )

    logger.info(f"Top fused chunks before reranking: {len(results)}")

    # ------------------------------------------------------
    # Limit Candidates Before Reranking
    # ------------------------------------------------------

    results = results[:MAX_RERANK_CANDIDATES]

    logger.info(
        f"Passing {len(results)} chunks to CrossEncoder."
    )

    # ------------------------------------------------------
    # CrossEncoder Reranking
    # ------------------------------------------------------

    results = rerank(
        query=query,
        retrieved_chunks=results,
        top_k=top_k,
    )

    logger.info(f"Returning {len(results)} chunks.")

    return results


# ==========================================================
# Standalone Test
# ==========================================================

if __name__ == "__main__":

    query = input("Enter your query: ")

    results = hybrid_retriever(query)

    print("\nHybrid Retrieval Results\n")

    for chunk in results:

        print("-" * 60)
        print(f"Rank       : {chunk['rank']}")
        print(f"Index      : {chunk['index']}")
        print(f"RRF Score  : {chunk['rrf_score']:.6f}")
        print(f"Document   : {chunk['source']}")
        print(f"Type       : {chunk['file_type']}")
        print(f"Retriever  : {chunk['retriever']}")

        if "distance" in chunk:
            print(f"Distance   : {chunk['distance']:.4f}")

        if "score" in chunk:
            print(f"BM25 Score : {chunk['score']:.4f}")

        print()
        print(chunk["text"])
        print("-" * 60)