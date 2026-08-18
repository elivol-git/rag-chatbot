"""Embedding generation via the local Ollama server.

Vectors are L2-normalized here, once, at both ingest and query time. That
makes cosine similarity a plain dot product in the vector store and removes
a whole class of scale bugs.
"""

from __future__ import annotations

import time

import numpy as np
import requests

from .config import settings

BATCH_SIZE = 32
# Generous, because a request can arrive while Ollama is swapping this model
# back into VRAM after the LLM evicted it.
TIMEOUT = 300
TIMEOUT_RETRIES = 2
ZERO_VECTOR_RETRIES = 3

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


def _request_batch(texts: list[str]) -> list[list[float]]:
    url = f"{settings.ollama_host}/api/embed"
    payload_json = {
        "model": settings.embed_model,
        "input": texts,
        # Keep the embed model resident; a cold load costs seconds per query.
        "keep_alive": "30m",
    }

    for attempt in range(TIMEOUT_RETRIES + 1):
        try:
            response = requests.post(url, json=payload_json, timeout=TIMEOUT)
            response.raise_for_status()
            break
        except requests.Timeout:
            # Usually VRAM pressure: the LLM evicted this model and it is being
            # loaded again. Worth one more try before giving up.
            if attempt == TIMEOUT_RETRIES:
                raise EmbeddingError(
                    f"Embedding request to {url} timed out after "
                    f"{TIMEOUT_RETRIES + 1} attempts of {TIMEOUT}s. Ollama may be "
                    "reloading models under memory pressure - check `ollama ps`."
                ) from None
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


def _has_zero_vector(vectors: list[list[float]]) -> bool:
    return any(not any(value for value in vector) for vector in vectors)


def _embed_batch(texts: list[str]) -> list[list[float]]:
    """Embed one batch, rejecting the all-zero vectors Ollama sometimes returns.

    The first embed request a process makes can come back as a zero vector while
    the model is still warming up. A zero vector is silently catastrophic: every
    cosine score becomes 0, so retrieval degrades to arbitrary chunks (or, with a
    similarity floor, refuses everything). Retry, then fail loudly.
    """
    for attempt in range(ZERO_VECTOR_RETRIES):
        vectors = _request_batch(texts)
        if not _has_zero_vector(vectors):
            return vectors
        time.sleep(0.5 * (attempt + 1))

    raise EmbeddingError(
        f"Ollama returned a zero embedding vector for '{settings.embed_model}' "
        f"after {ZERO_VECTOR_RETRIES} attempts. The model may have failed to load; "
        "check `ollama ps` and the Ollama server log."
    )


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
