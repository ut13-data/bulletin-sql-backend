"""
Paths, model settings and secrets for the Ask bUlleTin agent.

Framework-free on purpose: works inside Streamlit (reads st.secrets) and
inside a plain Python/FastAPI process (reads environment variables).
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# The database can live in data/bulletin.db (Streamlit repo) or at the repo
# root (FastAPI backend repo). BULLETIN_DB_PATH overrides both.
_DB_CANDIDATES = [ROOT / "data" / "bulletin.db", ROOT / "bulletin.db"]


def _find_db() -> Path:
    override = os.getenv("BULLETIN_DB_PATH")
    if override:
        return Path(override)
    for candidate in _DB_CANDIDATES:
        if candidate.exists():
            return candidate
    return _DB_CANDIDATES[0]


DB_PATH = _find_db()
DOCS_DIR = ROOT / "docs"

MODEL = "openai/gpt-oss-120b"

# FAISS returns a DISTANCE (lower = more similar). Chunks above this are ignored.
SIMILARITY_DISTANCE_THRESHOLD = 1.3

# Conversation limits
MAX_TURNS = 10            # questions per chat
HISTORY_TURNS_FOR_LLM = 4  # how many earlier turns the parser sees


def get_secret(name: str) -> str:
    """Read a key from Streamlit secrets if available, else from environment variables."""
    try:
        import streamlit as st  # imported lazily so the backend doesn't need Streamlit
        if name in st.secrets:
            return st.secrets[name]
    except Exception:
        pass
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing secret '{name}'. Add it to .streamlit/secrets.toml or the environment.")
    return value
