"""Ingestion pipeline: documents -> chunks -> embeddings -> vector store.

Incremental by default: a file whose sha256 matches the manifest is skipped,
a changed file has its old chunks dropped and replaced, and chunks belonging
to deleted files are pruned.

    python -m src.ingest            # incremental
    python -m src.ingest --rebuild  # wipe the store and start clean
"""

from __future__ import annotations

import argparse
import sys
import time
from typing import Any

from .chunking import split_text
from .config import settings
from .embeddings import embed_documents
from .loaders import iter_documents
from .vector_store import VectorStore


def _records_for(document, chunks: list[str]) -> list[dict[str, Any]]:
    return [
        {
            "id": f"{document.source}#{index}",
            "text": chunk,
            "source": document.source,
            "chunk_index": index,
            "title": document.metadata.get("title", document.source),
            "source_url": document.metadata.get("source_url", ""),
            "license": document.metadata.get("license", ""),
        }
        for index, chunk in enumerate(chunks)
    ]


def ingest(rebuild: bool = False, verbose: bool = True) -> dict[str, Any]:
    """Run ingestion. Returns a summary dict (also used by the /ingest route)."""
    started = time.time()
    store = VectorStore(directory=settings.vector_store_dir) if rebuild else VectorStore.load()

    seen: set[str] = set()
    changed = skipped = new_chunks = 0

    def log(message: str) -> None:
        if verbose:
            print(message, flush=True)

    for document in iter_documents(settings.documents_dir):
        seen.add(document.source)
        known = store.manifest.get(document.source)
        if known and known.get("sha256") == document.sha256:
            skipped += 1
            log(f"skip     {document.source}")
            continue

        chunks = split_text(document.text, settings.chunk_size, settings.chunk_overlap)
        if not chunks:
            continue

        store.delete_by_source(document.source)
        vectors = embed_documents(chunks)
        store.add(vectors, _records_for(document, chunks))
        store.manifest[document.source] = {
            "sha256": document.sha256,
            "mtime": document.mtime,
            "chunks": len(chunks),
            "title": document.metadata.get("title", document.source),
        }
        changed += 1
        new_chunks += len(chunks)
        log(f"indexed  {document.source} -> {len(chunks)} chunks")

    removed_files = [source for source in list(store.manifest) if source not in seen]
    removed_chunks = sum(store.delete_by_source(source) for source in removed_files)
    for source in removed_files:
        log(f"removed  {source}")

    store.save()
    summary = {
        "files_indexed": changed,
        "files_skipped": skipped,
        "files_removed": len(removed_files),
        "chunks_added": new_chunks,
        "chunks_removed": removed_chunks,
        "total_chunks": len(store),
        "dimension": store.dimension,
        "embed_model": settings.embed_model,
        "elapsed_seconds": round(time.time() - started, 2),
    }
    log(
        f"\n{summary['total_chunks']} chunks / {summary['dimension']}-dim in "
        f"{settings.vector_store_dir} ({summary['elapsed_seconds']}s)"
    )
    return summary


def main() -> int:
    # Document names and text are UTF-8; Windows consoles often are not.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="Ingest documents into the vector store")
    parser.add_argument(
        "--rebuild", action="store_true", help="discard the existing index and rebuild"
    )
    args = parser.parse_args()

    if not settings.documents_dir.exists():
        print(f"Documents directory not found: {settings.documents_dir}", file=sys.stderr)
        return 1

    ingest(rebuild=args.rebuild)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
