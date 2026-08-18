from src.prompts import NO_CONTEXT_ANSWER, build_prompt, format_context
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
