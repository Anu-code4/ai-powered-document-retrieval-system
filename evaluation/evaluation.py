

"""
Evaluation pipeline for AI Revision Companion.

Runs evaluation questions through:
Question -> Hybrid Retriever -> Generator -> CSV Results
"""

import csv
import json
import logging
import os
from utils.logger import setup_logger
from retrievers.hybrid_retriever import hybrid_retriever
from llm.generator import generate_answer
from .evaluation_dataset import evaluation_data

# ==========================================================
# Setup Logger
# ==========================================================

setup_logger()
logger = logging.getLogger(__name__)


# ==========================================================
# Evaluation
# ==========================================================

def evaluate():

    results = []

    logger.info("Starting evaluation process...")

    for item in evaluation_data:

        question = item["question"]
        expected = item["expected_answer"]

        logger.info(f"Evaluating: {question}")

        try:

            # --------------------------------------------------
            # Retrieve Relevant Documents
            # --------------------------------------------------

            retrieved_docs = hybrid_retriever(
                question,
                top_k=5
            )

            logger.info(
                f"Retrieved {len(retrieved_docs)} documents."
            )

            for i, doc in enumerate(retrieved_docs, start=1):
                logger.debug(
                    f"Chunk {i}: {doc.get('text', 'No text')[:150]}"
                )

            # --------------------------------------------------
            # Extract Contexts (Required by RAGAS)
            # --------------------------------------------------

            contexts = [
                doc["text"]
                for doc in retrieved_docs
            ]

            # --------------------------------------------------
            # Generate Answer
            # --------------------------------------------------

            answer = generate_answer(
                question,
                retrieved_docs
            )

            logger.info(f"Generated Answer: {answer}")

            # --------------------------------------------------
            # Store Results
            # --------------------------------------------------

            results.append(
                {
                    "question": question,
                    "expected_answer": expected,
                    "generated_answer": answer,
                    "contexts": contexts
                }
            )

            logger.info("Evaluation completed successfully.")

        except Exception as e:

            logger.exception(
                f"Failed evaluation for question: {question}"
            )

            results.append(
                {
                    "question": question,
                    "expected_answer": expected,
                    "generated_answer": f"ERROR: {e}",
                    "contexts": []
                }
            )

    return results


# ==========================================================
# Save CSV Results
# ==========================================================

def save_results(results):

    file_name = "evaluation_results.csv"
    file_path = os.path.abspath(file_name)

    csv_results = []

    for row in results:

        csv_results.append(
            {
                "question": row["question"],
                "expected_answer": row["expected_answer"],
                "generated_answer": row["generated_answer"],
                "contexts": "\n\n".join(row["contexts"])
            }
        )

    with open(
        file_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "question",
                "expected_answer",
                "generated_answer",
                "contexts"
            ]
        )

        writer.writeheader()
        writer.writerows(csv_results)

    logger.info(f"CSV results saved to {file_path}")

    return file_path


# ==========================================================
# Save JSON Results (Recommended for RAGAS)
# ==========================================================

def save_json(results):

    file_name = "evaluation_results.json"

    with open(
        file_name,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            results,
            file,
            indent=4,
            ensure_ascii=False
        )

    logger.info(f"JSON results saved to {file_name}")


# ==========================================================
# Main
# ==========================================================

if __name__ == "__main__":

    results = evaluate()

    # Save Evaluation Results
    saved_path = save_results(results)
    save_json(results)

    # Run RAGAS Evaluation
    from ragas_evaluator import evaluate_ragas

    ragas_results = evaluate_ragas(results)
try:

    ragas_results = evaluate_ragas(results)

    ragas_results.to_csv(
        "ragas_results.csv",
        index=False
    )

    logger.info(
        "RAGAS results saved successfully."
    )

except Exception:

    logger.exception(
        "Failed to run RAGAS evaluation."
    )

    print("\nRAGAS evaluation completed.")
    print("Results saved to ragas_results.csv")

    print("\n==============================")
    print("Evaluation completed.")
    print(f"Total Questions Evaluated: {len(results)}")
    print(f"CSV Results : {saved_path}")
    print("JSON Results: evaluation_results.json")
    print("RAGAS Results: ragas_results.csv")
    print("==============================")


import os

print(os.getcwd())