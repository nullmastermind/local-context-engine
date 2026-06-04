"""Tests for reranker — LLM-based result reranking."""

from __future__ import annotations

from unittest.mock import MagicMock

from corbell.core.query.reranker import RerankResult, rerank_chunks
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
    assert result.chunk_ids == [c.chunk_id for c in chunks]


def test_unconfigured_llm_returns_all_ids():
    """Unconfigured LLM (is_configured=False) falls back to original order."""
    m = MagicMock()
    m.is_configured = False

    chunks = [_make_chunk(f"c{i}") for i in range(3)]
    result = rerank_chunks("query", chunks, m)
    assert result.chunk_ids == ["c0", "c1", "c2"]


def test_empty_chunks_returns_empty():
    result = rerank_chunks("query", [], llm_client=None)
    assert result.chunk_ids == []


# ---------------------------------------------------------------------------
# LLM success paths
# ---------------------------------------------------------------------------

def test_valid_json_response_reorders():
    """Valid JSON integer-index array response reorders chunks."""
    m = MagicMock()
    m.is_configured = True
    # LLM returns integer indices: chunk 2 first, then 0, then 1
    m.call.return_value = "[2, 0, 1]"

    chunks = [_make_chunk("c0"), _make_chunk("c1"), _make_chunk("c2")]
    result = rerank_chunks("query", chunks, m)
    assert result.chunk_ids == ["c2", "c0", "c1"]


def test_valid_json_with_markdown_fences():
    """JSON wrapped in markdown code fences is parsed correctly."""
    m = MagicMock()
    m.is_configured = True
    m.call.return_value = "```json\n[1, 0]\n```"

    chunks = [_make_chunk("c0"), _make_chunk("c1")]
    result = rerank_chunks("query", chunks, m)
    assert result.chunk_ids == ["c1", "c0"]


def test_partial_response_filters_to_returned_ids():
    """If LLM returns only a subset of indices, only those IDs are included."""
    m = MagicMock()
    m.is_configured = True
    # Omit index 1 (c1)
    m.call.return_value = "[2, 0]"

    chunks = [_make_chunk("c0"), _make_chunk("c1"), _make_chunk("c2")]
    result = rerank_chunks("query", chunks, m)
    assert result.chunk_ids == ["c2", "c0"]
    assert "c1" not in result.chunk_ids


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
    assert result.chunk_ids == ["c0", "c1", "c2"]


def test_empty_json_array_falls_back_to_all():
    """Empty JSON array [] falls back to all IDs."""
    m = MagicMock()
    m.is_configured = True
    m.call.return_value = "[]"

    chunks = [_make_chunk(f"c{i}") for i in range(3)]
    result = rerank_chunks("query", chunks, m)
    assert result.chunk_ids == ["c0", "c1", "c2"]


def test_wrong_type_response_falls_back():
    """JSON object (not array) falls back to all IDs."""
    m = MagicMock()
    m.is_configured = True
    m.call.return_value = '{"ids": [0]}'

    chunks = [_make_chunk(f"c{i}") for i in range(3)]
    result = rerank_chunks("query", chunks, m)
    assert result.chunk_ids == ["c0", "c1", "c2"]


def test_llm_exception_falls_back():
    """Exception from LLM call falls back to all IDs."""
    m = MagicMock()
    m.is_configured = True
    m.call.side_effect = RuntimeError("network error")

    chunks = [_make_chunk(f"c{i}") for i in range(3)]
    result = rerank_chunks("query", chunks, m)
    assert result.chunk_ids == ["c0", "c1", "c2"]


def test_invalid_ids_in_response_filtered():
    """Out-of-range indices in response are filtered out."""
    m = MagicMock()
    m.is_configured = True
    # Index 99 is out of range for a 2-chunk list
    m.call.return_value = "[0, 99, 1]"

    chunks = [_make_chunk("c0"), _make_chunk("c1")]
    result = rerank_chunks("query", chunks, m)
    assert result.chunk_ids == ["c0", "c1"]
    assert len(result.chunk_ids) == 2


# ---------------------------------------------------------------------------
# RerankResult content tests
# ---------------------------------------------------------------------------

def test_rerank_result_contains_system_and_user_prompts():
    """RerankResult captures the system and user prompts sent to the LLM."""
    m = MagicMock()
    m.is_configured = True
    m.call.return_value = "[0]"

    chunks = [_make_chunk("c0")]
    result = rerank_chunks("my test query", chunks, m)

    assert isinstance(result, RerankResult)
    assert len(result.system_prompt) > 0
    assert "my test query" in result.user_prompt
    assert "ranker" in result.system_prompt.lower()


def test_rerank_result_fallback_used_on_parse_failure():
    """RerankResult.fallback_used is True when JSON parse fails."""
    m = MagicMock()
    m.is_configured = True
    m.call.return_value = "INVALID JSON"

    chunks = [_make_chunk(f"c{i}") for i in range(3)]
    result = rerank_chunks("query", chunks, m)

    assert result.fallback_used is True
    assert result.chunk_ids == ["c0", "c1", "c2"]


def test_rerank_result_fallback_used_on_exception():
    """RerankResult.fallback_used is True when LLM raises an exception."""
    m = MagicMock()
    m.is_configured = True
    m.call.side_effect = RuntimeError("timeout")

    chunks = [_make_chunk(f"c{i}") for i in range(3)]
    result = rerank_chunks("query", chunks, m)

    assert result.fallback_used is True


def test_rerank_result_fallback_not_used_on_success():
    """RerankResult.fallback_used is False on a successful parse."""
    m = MagicMock()
    m.is_configured = True
    m.call.return_value = "[0]"

    chunks = [_make_chunk("c0")]
    result = rerank_chunks("query", chunks, m)

    assert result.fallback_used is False


def test_rerank_result_raw_response_contains_llm_output():
    """RerankResult.raw_response contains the raw text returned by the LLM."""
    m = MagicMock()
    m.is_configured = True
    llm_output = "[0, 1]"
    m.call.return_value = llm_output

    chunks = [_make_chunk("c0"), _make_chunk("c1")]
    result = rerank_chunks("query", chunks, m)

    assert result.raw_response == llm_output


def test_rerank_result_raw_response_none_on_exception():
    """RerankResult.raw_response is None when LLM raises before returning."""
    m = MagicMock()
    m.is_configured = True
    m.call.side_effect = RuntimeError("connection refused")

    chunks = [_make_chunk("c0")]
    result = rerank_chunks("query", chunks, m)

    assert result.raw_response is None


def test_no_llm_result_has_empty_prompts():
    """When LLM is None, prompts and raw_response are empty/None."""
    chunks = [_make_chunk("c0"), _make_chunk("c1")]
    result = rerank_chunks("query", chunks, llm_client=None)

    assert result.system_prompt == ""
    assert result.user_prompt == ""
    assert result.raw_response is None
    assert result.elapsed_seconds == 0.0
    assert result.fallback_used is False
