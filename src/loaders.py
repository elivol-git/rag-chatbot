"""Read source documents into plain text plus metadata.

Supported: .md, .txt, .pdf. Markdown/text files may start with a small header
block terminated by a '---' line:

    title: Gothic architecture
    source_url: https://...
    license: CC BY-SA 4.0
    ---

Those keys become chunk metadata; the header itself is not embedded.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

SUPPORTED_SUFFIXES = {".md", ".txt", ".pdf"}
_HEADER_KEYS = {"title", "source_url", "license"}


@dataclass
class LoadedDocument:
    source: str  # path relative to the documents dir, used as the chunk key
    text: str
    sha256: str
    mtime: float
    metadata: dict[str, Any] = field(default_factory=dict)


def _parse_header(text: str) -> tuple[dict[str, Any], str]:
    lines = text.splitlines()
    metadata: dict[str, Any] = {}
    for index, line in enumerate(lines[:10]):
        stripped = line.strip()
        if stripped == "---":
            return metadata, "\n".join(lines[index + 1 :]).lstrip()
        if ":" in stripped:
            key, _, value = stripped.partition(":")
            key = key.strip().lower()
            if key in _HEADER_KEYS:
                metadata[key] = value.strip()
                continue
        break
    return {}, text


def _read_pdf(path: Path) -> str:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(pages)


def load_document(path: Path, documents_dir: Path) -> LoadedDocument | None:
    """Load one file. Returns None if it has no usable text."""
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        return None

    raw = _read_pdf(path) if suffix == ".pdf" else path.read_text(
        encoding="utf-8", errors="replace"
    )
    metadata, body = _parse_header(raw) if suffix != ".pdf" else ({}, raw)
    body = body.strip()
    if not body:
        return None

    metadata.setdefault("title", path.stem.replace("-", " ").replace("_", " ").title())
    stat = path.stat()
    return LoadedDocument(
        source=path.relative_to(documents_dir).as_posix(),
        text=body,
        sha256=hashlib.sha256(body.encode("utf-8")).hexdigest(),
        mtime=stat.st_mtime,
        metadata=metadata,
    )


def iter_documents(documents_dir: Path):
    """Yield every loadable document under documents_dir, sorted by path."""
    if not documents_dir.exists():
        return
    for path in sorted(documents_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        document = load_document(path, documents_dir)
        if document is not None:
            yield document
