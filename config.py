"""
Application configuration.
"""

from pathlib import Path

# ==========================================================
# LLM Configuration
# ==========================================================

MODEL_NAME = "llama3.2"

TEMPERATURE = 0.0

# ==========================================================
# Retrieval Configuration
# ==========================================================

FAISS_DISTANCE_THRESHOLD = 1.5
TOP_K = 5
RRF_K = 60

# ==========================================
# Evaluation Configuration
# ==========================================

EVALUATION_RESULTS_CSV = "evaluation_results.csv"
EVALUATION_RESULTS_EXCEL = "evaluation_results.xlsx"

MAX_RERANK_CANDIDATES = 50

DOCUMENTS_PATH = Path("Documents")

OLLAMA_HOST = "http://host.docker.internal:11434"
