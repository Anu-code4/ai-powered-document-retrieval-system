

"""
Metadata Filter

Extracts metadata filters from user queries.
"""

import logging
import re

from utils.logger import setup_logger

setup_logger()
logger = logging.getLogger(__name__)


def extract_metadata_filter(query: str) -> dict:

    logger.info("Extracting metadata filters.")

    filters = {}

    # ------------------------------------------------------
    # PDF / DOCX
    # ------------------------------------------------------

    query_lower = query.lower()

    if "pdf" in query_lower:

        filters["file_type"] = ".pdf"

    elif "docx" in query_lower:

        filters["file_type"] = ".docx"

    # ------------------------------------------------------
    # Filename Detection
    # ------------------------------------------------------

    match = re.search(r"([\w\-]+\.pdf)", query, re.IGNORECASE)

    if not match:

        match = re.search(r"([\w\-]+\.docx)", query, re.IGNORECASE)

    if match:

        filters["source"] = match.group(1)

    logger.info(f"Metadata Filters: {filters}")

    return filters