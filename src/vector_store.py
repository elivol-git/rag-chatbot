"""Hand-rolled vector store: NumPy matrix + JSON metadata, no index library.

Layout on disk (VECTOR_STORE_DIR):
  embeddings.npy  float32 [N, dim], every row L2-normalized
  chunks.json     list of N records, aligned by position with the matrix
  manifest.json   {relpath: {mtime, sha256, chunks}} for incremental ingest

Search is a brute-force dot product. With a few thousand chunks that is well
under a millisecond and keeps the whole similarity step readable.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .config import settings


class DimensionMismatch(RuntimeError):
    pass


@dataclass
class VectorStore:
    directory: Path
    embeddings: np.ndarray = field(
        default_factory=lambda: np.zeros((0, 0), dtype=np.float32)
    )
    records: list[dict[str, Any]] = field(default_factory=list)
    manifest: dict[str, Any] = field(default_factory=dict)

    # ---------- persistence ----------

    @classmethod
    def load(cls, directory: Path | None = None) -> "VectorStore":
        directory = Path(directory or settings.vector_store_dir)
        store = cls(directory=directory)
        emb_path = directory / "embeddings.npy"
        chunks_path = directory / "chunks.json"
        manifest_path = directory / "manifest.json"

        if emb_path.exists() and chunks_path.exists():
            store.embeddings = np.load(emb_path).astype(np.float32)
            store.records = json.loads(chunks_path.read_text(encoding="utf-8"))
            if len(store.records) != store.embeddings.shape[0]:
                raise RuntimeError(
                    f"Vector store corrupt: {store.embeddings.shape[0]} vectors "
                    f"but {len(store.records)} chunk records in {directory}. "
                    "Delete the directory and re-ingest."
                )
        if manifest_path.exists():
            store.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        return store

    def save(self) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        np.save(self.directory / "embeddings.npy", self.embeddings)
        (self.directory / "chunks.json").write_text(
            json.dumps(self.records, ensure_ascii=False), encoding="utf-8"
        )
        (self.directory / "manifest.json").write_text(
            json.dumps(self.manifest, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    # ---------- shape ----------

    def __len__(self) -> int:
        return len(self.records)

    @property
    def dimension(self) -> int:
        return int(self.embeddings.shape[1]) if self.embeddings.size else 0

    @property
    def sources(self) -> list[str]:
        return sorted({r["source"] for r in self.records})

    # ---------- mutation ----------

    def add(self, vectors: np.ndarray, records: list[dict[str, Any]]) -> None:
        if vectors.shape[0] != len(records):
            raise ValueError("vectors and records must be the same length")
        if vectors.shape[0] == 0:
            return
        if self.embeddings.size and vectors.shape[1] != self.dimension:
            raise DimensionMismatch(
                f"Stored vectors are {self.dimension}-dim but new vectors are "
                f"{vectors.shape[1]}-dim. The embedding model changed — delete "
                f"{self.directory} and re-ingest."
            )
        self.embeddings = (
            vectors.astype(np.float32)
            if not self.embeddings.size
            else np.vstack([self.embeddings, vectors.astype(np.float32)])
        )
        self.records.extend(records)

    def delete_by_source(self, source: str) -> int:
        """Drop every chunk that came from one source file. Returns count."""
        keep = [i for i, r in enumerate(self.records) if r["source"] != source]
        removed = len(self.records) - len(keep)
        if removed:
            self.embeddings = (
                self.embeddings[keep] if keep else np.zeros((0, 0), dtype=np.float32)
            )
            self.records = [self.records[i] for i in keep]
        self.manifest.pop(source, None)
        return removed

    # ---------- search ----------

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 4,
        source_filter: str | None = None,
    ) -> list[tuple[int, float]]:
        """Return [(record_index, cosine_score)] sorted by score descending."""
        if not len(self) or top_k <= 0:
            return []
        if query_vector.shape[0] != self.dimension:
            raise DimensionMismatch(
                f"Query vector is {query_vector.shape[0]}-dim but the store holds "
                f"{self.dimension}-dim vectors. Re-ingest with the current model."
            )

        # Both sides are L2-normalized, so the dot product is cosine similarity.
        scores = self.embeddings @ query_vector.astype(np.float32)

        candidates = np.arange(len(self))
        if source_filter:
            mask = np.array(
                [r["source"] == source_filter for r in self.records], dtype=bool
            )
            candidates = candidates[mask]
            if candidates.size == 0:
                return []
            scores = scores[mask]

        k = min(top_k, candidates.size)
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        return [(int(candidates[i]), float(scores[i])) for i in top]

    def record(self, index: int) -> dict[str, Any]:
        return self.records[index]
