import numpy as np
import pytest

from src import embeddings


def test_zero_vector_is_retried_then_accepted(monkeypatch):
    """Ollama's first response in a process can be a zero vector; retry past it."""
    responses = [[[0.0] * 4], [[0.0] * 4], [[1.0, 2.0, 2.0, 0.0]]]
    calls = []

    def fake_request(texts):
        calls.append(texts)
        return responses[len(calls) - 1]

    monkeypatch.setattr(embeddings, "_request_batch", fake_request)
    monkeypatch.setattr(embeddings.time, "sleep", lambda _: None)

    vector = embeddings._embed_batch(["query"])
    assert len(calls) == 3
    assert vector == [[1.0, 2.0, 2.0, 0.0]]


def test_persistent_zero_vector_raises(monkeypatch):
    monkeypatch.setattr(embeddings, "_request_batch", lambda texts: [[0.0] * 4])
    monkeypatch.setattr(embeddings.time, "sleep", lambda _: None)

    with pytest.raises(embeddings.EmbeddingError, match="zero embedding vector"):
        embeddings._embed_batch(["query"])


def test_embed_texts_normalizes(monkeypatch):
    monkeypatch.setattr(embeddings, "_request_batch", lambda texts: [[3.0, 4.0, 0.0]] * len(texts))

    matrix = embeddings.embed_texts(["a", "b"])
    assert matrix.shape == (2, 3)
    np.testing.assert_allclose(np.linalg.norm(matrix, axis=1), [1.0, 1.0], rtol=1e-6)
    assert matrix.dtype == np.float32


def test_document_and_query_prefixes_differ(monkeypatch):
    seen = []
    monkeypatch.setattr(
        embeddings,
        "_request_batch",
        lambda texts: (seen.extend(texts), [[1.0, 0.0]] * len(texts))[1],
    )
    # settings is a frozen dataclass, so patch the predicate rather than the field.
    monkeypatch.setattr(embeddings, "_uses_task_prefixes", lambda: True)

    embeddings.embed_documents(["a vaulted nave"])
    embeddings.embed_query("what is a nave?")

    assert seen[0].startswith("search_document: ")
    assert seen[1].startswith("search_query: ")


def test_empty_input_short_circuits(monkeypatch):
    monkeypatch.setattr(
        embeddings, "_request_batch", lambda texts: pytest.fail("should not call Ollama")
    )
    assert embeddings.embed_texts([]).shape == (0, 0)
