"""Central configuration. Reads .env, falls back to sane defaults."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent

load_dotenv(PROJECT_ROOT / ".env")


def _path(name: str, default: str) -> Path:
    raw = os.getenv(name, default)
    p = Path(raw)
    return p if p.is_absolute() else (PROJECT_ROOT / p).resolve()


def _int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except ValueError:
        return default


def _float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    ollama_host: str
    llm_model: str
    embed_model: str
    documents_dir: Path
    vector_store_dir: Path
    chunk_size: int
    chunk_overlap: int
    top_k: int
    min_score: float
    max_answer_tokens: int
    flask_port: int

    @property
    def embeddings_path(self) -> Path:
        return self.vector_store_dir / "embeddings.npy"

    @property
    def chunks_path(self) -> Path:
        return self.vector_store_dir / "chunks.json"

    @property
    def manifest_path(self) -> Path:
        return self.vector_store_dir / "manifest.json"


settings = Settings(
    # 127.0.0.1 rather than localhost: see the note in .env.example.
    ollama_host=os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/"),
    llm_model=os.getenv("LLM_MODEL", "aya-expanse:8b"),
    embed_model=os.getenv("EMBED_MODEL", "nomic-embed-text"),
    documents_dir=_path("DOCUMENTS_DIR", "./data/documents"),
    vector_store_dir=_path("VECTOR_STORE_DIR", "./data/vector_store"),
    chunk_size=_int("CHUNK_SIZE", 800),
    chunk_overlap=_int("CHUNK_OVERLAP", 50),
    top_k=_int("TOP_K", 4),
    min_score=_float("MIN_SCORE", 0.58),
    max_answer_tokens=_int("MAX_ANSWER_TOKENS", 400),
    flask_port=_int("FLASK_PORT", 5000),
)
