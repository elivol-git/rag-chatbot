"""The embed client must survive Ollama reloading a model under VRAM pressure."""

import pytest
import requests

from src import embeddings


class _Response:
    def __init__(self, vectors):
        self._vectors = vectors

    def raise_for_status(self):
        return None

    def json(self):
        return {"embeddings": self._vectors}


def test_timeout_is_retried(monkeypatch):
    calls = []

    def fake_post(url, json=None, timeout=None):
        calls.append(url)
        if len(calls) == 1:
            raise requests.Timeout("read timed out")
        return _Response([[1.0, 0.0]])

    monkeypatch.setattr(embeddings.requests, "post", fake_post)

    assert embeddings._request_batch(["query"]) == [[1.0, 0.0]]
    assert len(calls) == 2


def test_persistent_timeout_reports_memory_pressure(monkeypatch):
    def always_timeout(url, json=None, timeout=None):
        raise requests.Timeout("read timed out")

    monkeypatch.setattr(embeddings.requests, "post", always_timeout)

    with pytest.raises(embeddings.EmbeddingError, match="timed out after"):
        embeddings._request_batch(["query"])


def test_other_request_errors_are_not_retried(monkeypatch):
    calls = []

    def refuse(url, json=None, timeout=None):
        calls.append(url)
        raise requests.ConnectionError("connection refused")

    monkeypatch.setattr(embeddings.requests, "post", refuse)

    with pytest.raises(embeddings.EmbeddingError, match="Is Ollama running"):
        embeddings._request_batch(["query"])
    assert len(calls) == 1
