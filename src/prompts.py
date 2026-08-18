"""Prompt assembly: question + retrieved chunks -> grounded chat messages."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .retrieval import RetrievedChunk

_HEBREW = re.compile(r"[֐-׿]")

# The corpus is bilingual, so a question in either language can retrieve
# passages in the other. The answer follows the question, never the sources.
REFUSALS = {
    "en": "I don't have that in my knowledge base.",
    "he": "אין לי את זה במאגר הידע שלי.",
}
NO_CONTEXT_ANSWER = REFUSALS["en"]

_LANGUAGE_RULE = {
    "en": "Answer in English, even when the passages are in Hebrew.",
    "he": "ענה בעברית, גם כאשר הקטעים באנגלית. תרגם את המידע הדרוש.",
}

SYSTEM_PROMPT = """You are an assistant answering questions about architecture \
(buildings, movements, architects, construction, building materials and design theory).

Rules you must follow:
1. Answer ONLY from the numbered context passages given below. Do not use outside knowledge.
2. Cite the passages you used inline with their numbers, like [1] or [2][3].
3. If the context does not contain the answer, reply exactly: "{refusal}" \
— do not guess and do not apologise at length.
4. Never invent citation numbers that are not in the context.
5. Be concise and factual: a short paragraph, or bullets when listing.
6. {language_rule}
"""


def detect_language(text: str) -> str:
    """'he' if most of the question's words are Hebrew, otherwise 'en'.

    Questions here mix scripts in both directions: Hebrew ones carry Latin
    technical terms ("מה זה ICF"), English ones quote Hebrew terms ("What is
    בטון מזוין?"). Counting letters makes that second case a coin flip, since
    Hebrew words are short; counting words keeps the sentence's own language
    in charge. Ties go to English.
    """
    hebrew = latin = 0
    for word in re.findall(r"[^\W\d_]+", text, flags=re.UNICODE):
        if _HEBREW.search(word):
            hebrew += 1
        elif re.search(r"[A-Za-z]", word):
            latin += 1
    return "he" if hebrew > latin else "en"


def refusal_for(question: str) -> str:
    return REFUSALS[detect_language(question)]


def system_prompt_for(question: str) -> str:
    language = detect_language(question)
    return SYSTEM_PROMPT.format(
        refusal=REFUSALS[language], language_rule=_LANGUAGE_RULE[language]
    )


def format_context(chunks: list["RetrievedChunk"]) -> str:
    return "\n\n".join(
        f"[{n}] (source: {chunk.source} | {chunk.title})\n{chunk.text}"
        for n, chunk in enumerate(chunks, start=1)
    )


def build_prompt(question: str, chunks: list["RetrievedChunk"]) -> list[dict[str, Any]]:
    """Return chat messages ready for the LLM, in the question's language."""
    question = question.strip()
    closing = {
        "en": "Answer using only the passages above, with inline [n] citations, in English.",
        "he": "ענה רק על סמך הקטעים שלמעלה, עם ציטוטים [n] בתוך הטקסט, בעברית.",
    }[detect_language(question)]

    user_content = (
        f"Context passages:\n\n{format_context(chunks)}\n\n"
        f"Question: {question}\n\n{closing}"
    )
    return [
        {"role": "system", "content": system_prompt_for(question)},
        {"role": "user", "content": user_content},
    ]
