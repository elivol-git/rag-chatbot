"""MCP server exposing the architecture knowledge base as a callable tool.

It imports the same retrieve() the Flask API uses — the retrieval logic exists
in exactly one place (src/retrieval.py).

Run standalone (stdio transport):
    python -m src.mcp_server

Register with Claude Code:
    claude mcp add arch-kb -- C:/projects/RAG_chatbot/.venv/Scripts/python.exe \
        -m src.mcp_server
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Allow `python src/mcp_server.py` as well as `python -m src.mcp_server`.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src.retrieval import retrieve, store_stats
else:
    from .retrieval import retrieve, store_stats

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("architecture-knowledge-base")


@mcp.tool()
def search_knowledge_base(query: str, top_k: int = 4, source: str = "") -> str:
    """Search the architecture knowledge base for passages relevant to a query.

    Covers architectural movements, architects, building elements, construction
    technique and design theory. Returns the matching passages with their source
    file and cosine similarity score.

    Args:
        query: A natural-language question or topic.
        top_k: How many passages to return (default 4).
        source: Optional filename filter, e.g. "wiki-bauhaus.md".
    """
    chunks = retrieve(query, top_k=top_k, source_filter=source or None)
    if not chunks:
        return "No passages in the knowledge base matched that query."

    blocks = [
        f"[{n}] {chunk.title} (source: {chunk.source}, score: {chunk.score:.3f})\n{chunk.text}"
        for n, chunk in enumerate(chunks, start=1)
    ]
    return "\n\n---\n\n".join(blocks)


@mcp.tool()
def knowledge_base_stats() -> str:
    """Report what the architecture knowledge base currently contains."""
    stats = store_stats()
    return json.dumps(
        {
            "chunks": stats["chunks"],
            "documents": stats["documents"],
            "embedding_dimension": stats["dimension"],
            "sources": stats["sources"],
        },
        indent=2,
    )


if __name__ == "__main__":
    mcp.run(transport="stdio")
