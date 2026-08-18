import numpy as np
import pytest

from src.vector_store import DimensionMismatch, VectorStore


def _unit(*values: float) -> np.ndarray:
    vector = np.array(values, dtype=np.float32)
    return vector / np.linalg.norm(vector)


@pytest.fixture
def store(tmp_path):
    store = VectorStore(directory=tmp_path)
    vectors = np.vstack([_unit(1, 0, 0), _unit(0, 1, 0), _unit(1, 1, 0)])
    records = [
        {"id": "a#0", "text": "gothic vaults", "source": "a.md", "chunk_index": 0, "title": "A"},
        {"id": "b#0", "text": "bauhaus school", "source": "b.md", "chunk_index": 0, "title": "B"},
        {"id": "a#1", "text": "mixed topic", "source": "a.md", "chunk_index": 1, "title": "A"},
    ]
    store.add(vectors, records)
    return store


def test_search_ranks_by_cosine_similarity(store):
    hits = store.search(_unit(1, 0, 0), top_k=3)
    assert [index for index, _ in hits] == [0, 2, 1]
    assert hits[0][1] == pytest.approx(1.0, abs=1e-5)
    scores = [score for _, score in hits]
    assert scores == sorted(scores, reverse=True)


def test_source_filter_restricts_results(store):
    hits = store.search(_unit(1, 0, 0), top_k=5, source_filter="a.md")
    assert {store.record(i)["source"] for i, _ in hits} == {"a.md"}
    assert len(hits) == 2


def test_save_load_roundtrip(store, tmp_path):
    store.manifest["a.md"] = {"sha256": "deadbeef", "chunks": 2}
    store.save()

    reloaded = VectorStore.load(tmp_path)
    assert len(reloaded) == 3
    assert reloaded.dimension == 3
    assert reloaded.manifest["a.md"]["sha256"] == "deadbeef"
    np.testing.assert_allclose(reloaded.embeddings, store.embeddings, rtol=1e-6)
    assert reloaded.search(_unit(0, 1, 0), top_k=1)[0][0] == 1


def test_delete_by_source_removes_only_that_file(store):
    removed = store.delete_by_source("a.md")
    assert removed == 2
    assert len(store) == 1
    assert store.records[0]["source"] == "b.md"
    assert store.embeddings.shape == (1, 3)


def test_dimension_mismatch_is_reported(store):
    with pytest.raises(DimensionMismatch):
        store.search(_unit(1, 0, 0, 0), top_k=1)
    with pytest.raises(DimensionMismatch):
        store.add(np.zeros((1, 8), dtype=np.float32), [{"source": "c.md", "text": "x"}])


def test_empty_store_search_returns_nothing(tmp_path):
    assert VectorStore(directory=tmp_path).search(_unit(1, 0, 0), top_k=3) == []
