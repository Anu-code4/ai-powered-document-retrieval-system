

"""
Generate multiple search queries from a user's question.
"""

import logging

from ollama import chat

from config import MODEL_NAME, TEMPERATURE
from utils.logger import setup_logger

setup_logger()
logger = logging.getLogger(__name__)


def generate_multi_queries(question: str) -> list[str]:

    logger.info("Generating multiple search queries.")

    prompt = f"""
Generate 4 different search queries for retrieving documents.

Requirements:
- Preserve the original meaning.
- Use different wording.
- One query per line.
- Do NOT number the queries.
- Do NOT explain anything.

Question:
{question}
""".strip()

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
                "temperature": 0.2,
            },
        )

    except Exception as e:

        logger.exception("Failed to generate multi queries.")

        return [question]

    content = response["message"]["content"]

    queries = []

    for line in content.splitlines():

        line = line.strip()

        if line:
            queries.append(line)

    queries.append(question)

    queries = list(dict.fromkeys(queries))

    logger.info(f"Generated {len(queries)} queries.")

    return queries