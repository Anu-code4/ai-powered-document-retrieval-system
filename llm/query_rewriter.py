

"""
Query Rewriter

Converts follow-up questions into standalone questions.
"""

import logging
import re

from ollama import chat

from config import MODEL_NAME
from utils.logger import setup_logger

# ==========================================================
# Logger
# ==========================================================

setup_logger()
logger = logging.getLogger(__name__)

# ==========================================================
# Follow-up Indicators
# ==========================================================

FOLLOW_UP_WORDS = {
    "it", "its",
    "they", "them", "their",
    "this", "that", "these", "those",
    "he", "his",
    "she", "her",
    "former", "latter",
    "previous",
}


# ==========================================================
# Detect Follow-up Query
# ==========================================================

def needs_rewriting(
    question: str,
    conversation_history=None,
) -> bool:
    """
    Returns True if the question appears to be a follow-up.
    """

    if not conversation_history:
        return False

    words = re.findall(
        r"\b\w+\b",
        question.lower(),
    )

    return any(
        word in FOLLOW_UP_WORDS
        for word in words
    )


# ==========================================================
# Rewrite Query
# ==========================================================

def rewrite_query(
    question: str,
    conversation_history=None,
) -> str:

    logger.info("Entered rewrite_query().")

    if not needs_rewriting(
        question,
        conversation_history,
    ):

        logger.info(
            "Standalone query. No rewrite needed."
        )

        return question

    # ------------------------------------------------------
    # Collect previous user questions
    # ------------------------------------------------------

    user_history = [
        message["content"]
        for message in conversation_history
        if message["role"] == "user"
    ]

    history = "\n".join(user_history)

    # ------------------------------------------------------
    # Prompt
    # ------------------------------------------------------

    prompt = f"""
You are a query rewriting assistant.

Rewrite ONLY the final user question.

Rules:

1. Use previous USER questions only.
2. Replace pronouns such as:
   - it
   - this
   - that
   - they
3. Do NOT answer the question.
4. Return ONLY the rewritten question.
5. Do not add explanations.

Previous User Questions:

{history}

Current Question:

{question}

Rewritten Question:
""".strip()

    # ------------------------------------------------------
    # Rewrite
    # ------------------------------------------------------

    try:

        response = chat(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            options={
                "temperature": 0,
            },
        )

        rewritten = (
            response
            .get("message", {})
            .get("content", "")
            .strip()
        )

        if not rewritten:
            return question

        logger.info(
            f"Rewritten Query: {rewritten}"
        )

        return rewritten

    except Exception:

        logger.exception(
            "Query rewriting failed."
        )

        return question