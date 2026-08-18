import pytest

from src.prompts import (
    NO_CONTEXT_ANSWER,
    REFUSALS,
    build_prompt,
    detect_language,
    format_context,
    refusal_for,
)
from src.retrieval import RetrievedChunk


def _chunk(text: str, source: str, score: float) -> RetrievedChunk:
    return RetrievedChunk(
        text=text, source=source, title=source, score=score, chunk_index=0
    )


def test_context_is_numbered_from_one():
    context = format_context(
        [_chunk("first passage", "a.md", 0.9), _chunk("second passage", "b.md", 0.8)]
    )
    assert "[1]" in context and "[2]" in context
    assert context.index("[1]") < context.index("[2]")


def test_prompt_contains_question_and_every_chunk():
    chunks = [_chunk("flying buttresses transfer thrust", "wiki-gothic.md", 0.7)]
    messages = build_prompt("What is a flying buttress?", chunks)

    assert [m["role"] for m in messages] == ["system", "user"]
    assert "only" in messages[0]["content"].lower()
    assert NO_CONTEXT_ANSWER in messages[0]["content"]
    assert "What is a flying buttress?" in messages[1]["content"]
    assert "flying buttresses transfer thrust" in messages[1]["content"]
    assert "wiki-gothic.md" in messages[1]["content"]


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("What defines Brutalist architecture?", "en"),
        ("מהי אדריכלות מדברית?", "he"),
        ("מה זה ICF ואיך משתמשים בו בבנייה?", "he"),  # Hebrew with a Latin term
        ("What is בטון מזוין?", "en"),  # English with a Hebrew term
    ],
)
def test_language_follows_the_question(question, expected):
    assert detect_language(question) == expected
    assert refusal_for(question) == REFUSALS[expected]


def test_hebrew_question_over_english_sources_asks_for_a_hebrew_answer():
    chunks = [_chunk("Reinforced concrete combines concrete and steel.", "materials.md", 0.8)]
    messages = build_prompt("מה זה בטון מזוין?", chunks)

    assert "ענה בעברית" in messages[0]["content"]
    assert REFUSALS["he"] in messages[0]["content"]
    assert "בעברית" in messages[1]["content"]
    # The English passage is still the only source of facts.
    assert "Reinforced concrete combines concrete and steel." in messages[1]["content"]


def test_english_question_over_hebrew_sources_asks_for_an_english_answer():
    chunks = [_chunk("בטון מזוין משלב בטון ופלדה.", "michlala/materials.md", 0.8)]
    messages = build_prompt("What is reinforced concrete?", chunks)

    assert "Answer in English" in messages[0]["content"]
    assert NO_CONTEXT_ANSWER in messages[0]["content"]
    assert "בטון מזוין משלב בטון ופלדה." in messages[1]["content"]
