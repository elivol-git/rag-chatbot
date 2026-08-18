"""End-to-end retrieval checks against the real index and a live Ollama.

Skipped automatically when the vector store has not been built yet or Ollama
is not running, so `pytest tests` stays green on a fresh clone.
"""

import pytest

from src.llm import ollama_reachable
from src.retrieval import retrieve, store_stats

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module", autouse=True)
def requires_index():
    if not ollama_reachable():
        pytest.skip("Ollama is not running")
    if store_stats()["chunks"] == 0:
        pytest.skip("vector store is empty - run `python -m src.ingest` first")


def test_scores_are_descending_and_within_range():
    chunks = retrieve("What is a flying buttress?", top_k=4)
    assert chunks, "expected at least one hit for a core corpus topic"
    scores = [c.score for c in chunks]
    assert scores == sorted(scores, reverse=True)
    assert all(0.0 <= s <= 1.0001 for s in scores)


def test_retrieval_finds_the_expected_document():
    chunks = retrieve("Who founded the Bauhaus school?", top_k=5)
    assert any("bauhaus" in c.source.lower() for c in chunks)


def test_source_filter_restricts_to_one_document():
    sources = store_stats()["sources"]
    target = next((s for s in sources if "bauhaus" in s.lower()), sources[0])
    chunks = retrieve("architecture", top_k=3, source_filter=target)
    assert chunks and all(c.source == target for c in chunks)


def test_top_k_is_respected():
    assert len(retrieve("gothic cathedral", top_k=2, min_score=0.0)) == 2


def test_off_topic_question_returns_nothing_above_floor():
    chunks = retrieve("How do I configure a Kubernetes ingress controller?", top_k=4)
    assert not chunks, f"off-topic query leaked chunks: {[c.source for c in chunks]}"
