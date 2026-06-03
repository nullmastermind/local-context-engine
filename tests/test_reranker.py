"""Tests for reranker — LLM-based result reranking."""

from __future__ import annotations

from unittest.mock import MagicMock

from corbell.core.query.reranker import rerank_chunks
from corbell.core.query.graph_expander import ScoredChunk


def _make_chunk(chunk_id: str, score: float = 0.8) -> ScoredChunk:
    return ScoredChunk(
        chunk_id=chunk_id,
        score=score,
        file_path="/repo/auth.py",
        start_line=1,
        end_line=10,
        content="def auth(): pass",
        repo_id="repo",
        symbol="auth",
        chunk_type="function",
        language="python",
    )


# ---------------------------------------------------------------------------
# No LLM
# ---------------------------------------------------------------------------

def test_no_llm_returns_all_ids_in_order():
    """Without LLM, all chunk_ids are returned in original order."""
    chunks = [_make_chunk(f"c{i}") for i in range(5)]
    result = rerank_chunks("query", chunks, llm_client=None)
    assert result == [c.chunk_id for c in chunks]


def test_unconfigured_llm_returns_all_ids():
    """Unconfigured LLM (is_configured=False) falls back to original order."""
    m = MagicMock()
    m.is_configured = False

    chunks = [_make_chunk(f"c{i}") for i in range(3)]
    result = rerank_chunks("query", chunks, m)
    assert result == ["c0", "c1", "c2"]


def test_empty_chunks_returns_empty():
    result = rerank_chunks("query", [], llm_client=None)
    assert result == []


# ---------------------------------------------------------------------------
# LLM success paths
# ---------------------------------------------------------------------------

def test_valid_json_response_reorders():
    """Valid JSON array response reorders chunks."""
    m = MagicMock()
    m.is_configured = True
    m.call.return_value = '["c2", "c0", "c1"]'

    chunks = [_make_chunk("c0"), _make_chunk("c1"), _make_chunk("c2")]
    result = rerank_chunks("query", chunks, m)
    assert result == ["c2", "c0", "c1"]


def test_valid_json_with_markdown_fences():
    """JSON wrapped in markdown code fences is parsed correctly."""
    m = MagicMock()
    m.is_configured = True
    m.call.return_value = '```json\n["c1", "c0"]\n```'

    chunks = [_make_chunk("c0"), _make_chunk("c1")]
    result = rerank_chunks("query", chunks, m)
    assert result == ["c1", "c0"]


def test_partial_response_filters_to_returned_ids():
    """If LLM returns only a subset, only those IDs are included."""
    m = MagicMock()
    m.is_configured = True
    m.call.return_value = '["c2", "c0"]'  # omits c1

    chunks = [_make_chunk("c0"), _make_chunk("c1"), _make_chunk("c2")]
    result = rerank_chunks("query", chunks, m)
    assert result == ["c2", "c0"]
    assert "c1" not in result


# ---------------------------------------------------------------------------
# Fallback paths
# ---------------------------------------------------------------------------

def test_invalid_json_falls_back_to_all():
    """Invalid JSON response falls back to all IDs in original order."""
    m = MagicMock()
    m.is_configured = True
    m.call.return_value = "NOT VALID JSON AT ALL"

    chunks = [_make_chunk(f"c{i}") for i in range(3)]
    result = rerank_chunks("query", chunks, m)
    assert result == ["c0", "c1", "c2"]


def test_empty_json_array_falls_back_to_all():
    """Empty JSON array [] falls back to all IDs."""
    m = MagicMock()
    m.is_configured = True
    m.call.return_value = "[]"

    chunks = [_make_chunk(f"c{i}") for i in range(3)]
    result = rerank_chunks("query", chunks, m)
    assert result == ["c0", "c1", "c2"]


def test_wrong_type_response_falls_back():
    """JSON object (not array) falls back to all IDs."""
    m = MagicMock()
    m.is_configured = True
    m.call.return_value = '{"ids": ["c0"]}'

    chunks = [_make_chunk(f"c{i}") for i in range(3)]
    result = rerank_chunks("query", chunks, m)
    assert result == ["c0", "c1", "c2"]


def test_llm_exception_falls_back():
    """Exception from LLM call falls back to all IDs."""
    m = MagicMock()
    m.is_configured = True
    m.call.side_effect = RuntimeError("network error")

    chunks = [_make_chunk(f"c{i}") for i in range(3)]
    result = rerank_chunks("query", chunks, m)
    assert result == ["c0", "c1", "c2"]


def test_invalid_ids_in_response_filtered():
    """IDs in response that don't exist in input are filtered out."""
    m = MagicMock()
    m.is_configured = True
    m.call.return_value = '["c0", "nonexistent_id", "c1"]'

    chunks = [_make_chunk("c0"), _make_chunk("c1")]
    result = rerank_chunks("query", chunks, m)
    assert result == ["c0", "c1"]
    assert "nonexistent_id" not in result
