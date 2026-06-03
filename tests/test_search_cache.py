"""Tests for EmbeddingSearchCache — vectorized cosine search."""

from __future__ import annotations

import numpy as np
import pytest

from corbell.core.embeddings.search_cache import EmbeddingSearchCache
from corbell.core.embeddings.sqlite_store import SQLiteEmbeddingStore
from corbell.core.embeddings.extractor import EmbeddingRecord


def _make_record(chunk_id: str, embedding: list) -> EmbeddingRecord:
    return EmbeddingRecord(
        id=chunk_id,
        service_id="svc",
        repo="/repo",
        file_path="test.py",
        start_line=1,
        end_line=10,
        content="code",
        language="python",
        chunk_type="function",
        embedding=embedding,
    )


@pytest.fixture
def store(tmp_db):
    return SQLiteEmbeddingStore(tmp_db)


# ---------------------------------------------------------------------------
# is_loaded property
# ---------------------------------------------------------------------------

def test_not_loaded_initially():
    cache = EmbeddingSearchCache()
    assert not cache.is_loaded


def test_loaded_after_load(store):
    dim = 384
    vec = [0.1] * dim
    store.upsert(_make_record("chunk::1", vec))

    cache = EmbeddingSearchCache()
    cache.load(store)
    assert cache.is_loaded


def test_not_loaded_when_store_empty(store):
    cache = EmbeddingSearchCache()
    cache.load(store)
    assert not cache.is_loaded


# ---------------------------------------------------------------------------
# load correctness
# ---------------------------------------------------------------------------

def test_load_multiple_vectors(store):
    dim = 384
    for i in range(10):
        vec = [float(i) / 10] * dim
        store.upsert(_make_record(f"chunk::{i}", vec))

    cache = EmbeddingSearchCache()
    cache.load(store)
    assert cache.is_loaded
    assert cache._matrix is not None
    assert cache._matrix.shape == (10, dim)


# ---------------------------------------------------------------------------
# search correctness
# ---------------------------------------------------------------------------

def test_search_returns_correct_top_k(store):
    """Top-K results are the K most similar chunks."""
    # Create 5 vectors, where vec4 is most similar to query [1,0,0,0]
    store.upsert(_make_record("a", [1.0, 0, 0, 0]))
    store.upsert(_make_record("b", [0, 1.0, 0, 0]))
    store.upsert(_make_record("c", [0, 0, 1.0, 0]))
    store.upsert(_make_record("d", [0, 0, 0, 1.0]))
    store.upsert(_make_record("e", [0.5, 0.5, 0, 0]))

    cache = EmbeddingSearchCache()
    cache.load(store)

    query = np.array([1.0, 0, 0, 0], dtype=np.float32)
    results = cache.search(query, top_k=3)

    assert len(results) == 3
    # "a" should be the most similar (cosine = 1.0)
    assert results[0][0] == "a"
    assert abs(results[0][1] - 1.0) < 1e-4


def test_search_scores_descending(store):
    """Results are returned in descending score order."""
    store.upsert(_make_record("high", [1.0, 0, 0, 0]))
    store.upsert(_make_record("mid", [0.5, 0.5, 0, 0]))
    store.upsert(_make_record("low", [0, 1.0, 0, 0]))

    cache = EmbeddingSearchCache()
    cache.load(store)

    query = np.array([1.0, 0, 0, 0], dtype=np.float32)
    results = cache.search(query, top_k=3)

    assert len(results) == 3
    scores = [r[1] for r in results]
    assert scores == sorted(scores, reverse=True)


def test_search_empty_cache():
    """Search returns empty list when cache is not loaded."""
    cache = EmbeddingSearchCache()
    query = np.array([1.0] * 384, dtype=np.float32)
    results = cache.search(query, top_k=10)
    assert results == []


def test_search_zero_query_vector(store):
    """Search returns empty list for zero query vector."""
    store.upsert(_make_record("a", [1.0] * 384))
    cache = EmbeddingSearchCache()
    cache.load(store)

    query = np.zeros(384, dtype=np.float32)
    results = cache.search(query, top_k=5)
    assert results == []


def test_search_top_k_capped(store):
    """top_k is capped at the number of available chunks."""
    dim = 384
    for i in range(3):
        store.upsert(_make_record(f"chunk::{i}", [float(i) / 10 + 0.01] * dim))

    cache = EmbeddingSearchCache()
    cache.load(store)

    query = np.ones(dim, dtype=np.float32)
    results = cache.search(query, top_k=100)  # more than available
    assert len(results) == 3
