"""Prompt assembly: question + retrieved chunks -> grounded chat messages."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .retrieval import RetrievedChunk

SYSTEM_PROMPT = """You are an assistant answering questions about architecture \
(buildings, movements, architects, construction and design theory).

Rules you must follow:
1. Answer ONLY from the numbered context passages given below. Do not use outside knowledge.
2. Cite the passages you used inline with their numbers, like [1] or [2][3].
3. If the context does not contain the answer, reply exactly: \
"I don't have that in my knowledge base." — do not guess and do not apologise at length.
4. Never invent citation numbers that are not in the context.
5. Be concise and factual: a short paragraph, or bullets when listing.
"""

NO_CONTEXT_ANSWER = "I don't have that in my knowledge base."


def format_context(chunks: list["RetrievedChunk"]) -> str:
    return "\n\n".join(
        f"[{n}] (source: {chunk.source} | {chunk.title})\n{chunk.text}"
        for n, chunk in enumerate(chunks, start=1)
    )


def build_prompt(question: str, chunks: list["RetrievedChunk"]) -> list[dict[str, Any]]:
    """Return chat messages ready for the LLM."""
    user_content = (
        f"Context passages:\n\n{format_context(chunks)}\n\n"
        f"Question: {question.strip()}\n\n"
        "Answer using only the passages above, with inline [n] citations."
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
