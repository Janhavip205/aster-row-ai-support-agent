"""Configuration management for the Aster & Row AI Support Agent."""

import os
from pathlib import Path

# Attempt to load dotenv if available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
ORDERS_PATH = DATA_DIR / "orders.json"
ORDERS_DICT_PATH = DATA_DIR / "orders-data-dictionary.md"
KB_DIR = BASE_DIR / "knowledge-base"
EVAL_DIR = BASE_DIR / "evaluation"
VISIBLE_CASES_PATH = EVAL_DIR / "visible-cases.json"
CUSTOM_CASES_PATH = EVAL_DIR / "custom-cases.json"

# Operational Snapshot Time (authoritative for mock data)
SNAPSHOT_AT = "2026-08-15T12:00:00Z"

# Gemini LLM Settings
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

# Local Embeddings Settings
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

# Observability / Debug Settings
DEBUG = os.getenv("DEBUG", "false").lower() in ("true", "1", "yes")


def check_dataset_paths() -> dict[str, bool]:
    """Verify that required data directories and files exist."""
    return {
        "data_dir": DATA_DIR.is_dir(),
        "orders_file": ORDERS_PATH.is_file(),
        "kb_dir": KB_DIR.is_dir(),
        "visible_cases": VISIBLE_CASES_PATH.is_file(),
    }
