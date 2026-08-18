"""Embedding generation via the local Ollama server.

Vectors are L2-normalized here, once, at both ingest and query time. That
makes cosine similarity a plain dot product in the vector store and removes
a whole class of scale bugs.
"""

from __future__ import annotations

import numpy as np
import requests

from .config import settings

BATCH_SIZE = 32
TIMEOUT = 120

# nomic-embed-text is trained with asymmetric task prefixes: stored passages
# and incoming questions are embedded differently. Applied only for that model
# family so swapping EMBED_MODEL stays a one-line change.
_DOC_PREFIX = "search_document: "
_QUERY_PREFIX = "search_query: "


def _uses_task_prefixes() -> bool:
    return settings.embed_model.startswith("nomic-embed-text")


class EmbeddingError(RuntimeError):
    pass


def _normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (matrix / norms).astype(np.float32)


def _embed_batch(texts: list[str]) -> list[list[float]]:
    url = f"{settings.ollama_host}/api/embed"
    try:
        response = requests.post(
            url,
            json={"model": settings.embed_model, "input": texts},
            timeout=TIMEOUT,
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        raise EmbeddingError(
            f"Embedding request to {url} failed ({exc}). "
            f"Is Ollama running and '{settings.embed_model}' pulled?"
        ) from exc

    payload = response.json()
    vectors = payload.get("embeddings")
    if not vectors:
        raise EmbeddingError(f"Ollama returned no embeddings: {payload}")
    return vectors


def embed_texts(texts: list[str]) -> np.ndarray:
    """Embed a list of raw texts -> float32 array of shape [len(texts), dim]."""
    if not texts:
        return np.zeros((0, 0), dtype=np.float32)

    vectors: list[list[float]] = []
    for start in range(0, len(texts), BATCH_SIZE):
        vectors.extend(_embed_batch(texts[start : start + BATCH_SIZE]))

    return _normalize(np.array(vectors, dtype=np.float32))


def embed_documents(texts: list[str]) -> np.ndarray:
    """Embed passages for storage (ingest side)."""
    if _uses_task_prefixes():
        texts = [_DOC_PREFIX + t for t in texts]
    return embed_texts(texts)


def embed_query(text: str) -> np.ndarray:
    """Embed a single question for search -> normalized 1-D vector [dim]."""
    if _uses_task_prefixes():
        text = _QUERY_PREFIX + text
    return embed_texts([text])[0]


def embedding_dimension() -> int:
    """Dimension of the configured embedding model (one live probe call)."""
    return int(embed_query("dimension probe").shape[0])
