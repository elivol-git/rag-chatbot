"""The single retrieval implementation, shared by the Flask API and MCP server.

Nothing else in the project may re-implement search: both entry points call
retrieve() so the knowledge base behaves identically wherever it is used.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

from .config import settings
from .embeddings import embed_query
from .vector_store import VectorStore

_store: VectorStore | None = None


@dataclass
class RetrievedChunk:
    text: str
    source: str
    title: str
    score: float
    chunk_index: int
    source_url: str = ""

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["score"] = round(self.score, 4)
        return data


def get_store() -> VectorStore:
    """Lazily load the store once per process."""
    global _store
    if _store is None:
        _store = VectorStore.load()
    return _store


def reload_store() -> VectorStore:
    """Drop the cached store (called after ingestion writes a new index)."""
    global _store
    _store = None
    return get_store()


def retrieve(
    query: str,
    top_k: int | None = None,
    source_filter: str | None = None,
    min_score: float | None = None,
) -> list[RetrievedChunk]:
    """Embed the question, rank chunks by cosine similarity, drop weak hits."""
    query = (query or "").strip()
    if not query:
        return []

    top_k = settings.top_k if top_k is None else max(1, int(top_k))
    min_score = settings.min_score if min_score is None else float(min_score)

    store = get_store()
    if not len(store):
        return []

    hits = store.search(embed_query(query), top_k=top_k, source_filter=source_filter)

    results: list[RetrievedChunk] = []
    for index, score in hits:
        if score < min_score:
            continue
        record = store.record(index)
        results.append(
            RetrievedChunk(
                text=record["text"],
                source=record["source"],
                title=record.get("title", record["source"]),
                score=score,
                chunk_index=record.get("chunk_index", 0),
                source_url=record.get("source_url", ""),
            )
        )
    return results


def store_stats() -> dict[str, Any]:
    store = get_store()
    return {
        "chunks": len(store),
        "dimension": store.dimension,
        "documents": len(store.manifest),
        "sources": store.sources,
    }
