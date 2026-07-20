

"""
RAGAS Evaluator

Evaluates generated answers using RAGAS metrics.
"""

import json
from utils.logger import setup_logger
import time
import logging

from datasets import Dataset
from ragas import evaluate

from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)

from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings

from config import MODEL_NAME

# ==========================================================
# Logger
# ==========================================================

logger = logging.getLogger(__name__)


# ==========================================================
# Model Loaders
# ==========================================================

def get_llm():
    """Create the evaluation LLM."""
    return ChatOllama(
        model=MODEL_NAME,
        temperature=0
    )


def get_embeddings():
    """Create the embedding model."""
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


# ==========================================================
# Validate Evaluation Results
# ==========================================================

def validate_results(results):

    required_fields = [
        "question",
        "generated_answer",
        "contexts",
        "expected_answer",
    ]

    for row in results:
        for field in required_fields:
            if field not in row:
                raise ValueError(
                    f"Missing required field: {field}"
                )


# ==========================================================
# Save Summary Metrics
# ==========================================================

def save_summary(df):

    summary = (
        df.mean(numeric_only=True)
        .to_dict()
    )

    with open(
        "ragas_summary.json",
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            summary,
            file,
            indent=4
        )

    logger.info(
        "RAGAS summary saved to ragas_summary.json"
    )


# ==========================================================
# RAGAS Evaluation
# ==========================================================

def evaluate_ragas(results):

    validate_results(results)

    logger.info(
        "Preparing dataset for RAGAS evaluation..."
    )

    dataset = Dataset.from_dict(
        {
            "question": [
                item["question"]
                for item in results
            ],

            "answer": [
                item["generated_answer"]
                for item in results
            ],

            "contexts": [
                item["contexts"]
                for item in results
            ],

            "ground_truth": [
                item["expected_answer"]
                for item in results
            ],
        }
    )

    logger.info(
        f"Dataset size: {len(dataset)}"
    )

    logger.info(
        "Running RAGAS evaluation..."
    )

    start_time = time.time()

    try:

        scores = evaluate(
            dataset=dataset,
            metrics=[
                faithfulness,
                answer_relevancy,
                context_precision,
                context_recall,
            ],
            llm=get_llm(),
            embeddings=get_embeddings(),
        )

    except Exception:

        logger.exception(
            "RAGAS evaluation failed."
        )
        raise

    elapsed = time.time() - start_time

    logger.info(
        f"Evaluation completed in {elapsed:.2f} seconds."
    )

    df = scores.to_pandas()

    save_summary(df)

    logger.info(
        "Average RAGAS Scores:\n%s",
        df.mean(numeric_only=True)
    )

    return df