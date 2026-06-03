"""Tests for merger — deduplication and adjacent chunk merging."""

from __future__ import annotations

from corbell.core.query.merger import merge_and_dedup
from corbell.core.query.graph_expander import ScoredChunk


def _make_chunk(
    chunk_id: str,
    file_path: str = "/repo/auth.py",
    start: int = 1,
    end: int = 10,
    score: float = 0.8,
    repo_id: str = "repo",
    content: str | None = None,
) -> ScoredChunk:
    if content is None:
        lines = [f"line {i}" for i in range(start, end + 1)]
        content = "\n".join(lines)
    return ScoredChunk(
        chunk_id=chunk_id,
        score=score,
        file_path=file_path,
        start_line=start,
        end_line=end,
        content=content,
        repo_id=repo_id,
    )


# ---------------------------------------------------------------------------
# Empty input
# ---------------------------------------------------------------------------

def test_empty_input():
    assert merge_and_dedup([]) == []


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------

def test_dedup_keeps_max_score():
    """Same chunk_id appearing twice: keep the higher score."""
    c1_low = _make_chunk("c1", score=0.5)
    c1_high = _make_chunk("c1", score=0.9)

    result = merge_and_dedup([c1_low, c1_high])
    assert len(result) == 1
    assert result[0].score == 0.9


def test_dedup_different_ids_kept():
    """Different chunk_ids are kept as separate entries."""
    c1 = _make_chunk("c1", start=1, end=10)
    c2 = _make_chunk("c2", start=20, end=30)

    result = merge_and_dedup([c1, c2])
    assert len(result) == 2


# ---------------------------------------------------------------------------
# Adjacent merging
# ---------------------------------------------------------------------------

def test_adjacent_chunks_merged(tmp_path):
    """Chunks with next.start <= current.end + 1 are merged."""
    f = tmp_path / "auth.py"
    lines = [f"code line {i}" for i in range(100)]
    f.write_text("\n".join(lines))

    c1 = _make_chunk("c1", str(f), start=1, end=20)
    c2 = _make_chunk("c2", str(f), start=21, end=40)  # adjacent

    result = merge_and_dedup([c1, c2])
    assert len(result) == 1
    assert result[0].start_line == 1
    assert result[0].end_line == 40


def test_overlapping_chunks_merged(tmp_path):
    """Overlapping chunks (from extractor overlap) are merged."""
    f = tmp_path / "module.py"
    lines = [f"line {i}" for i in range(100)]
    f.write_text("\n".join(lines))

    c1 = _make_chunk("c1", str(f), start=1, end=30)
    c2 = _make_chunk("c2", str(f), start=25, end=50)  # overlaps

    result = merge_and_dedup([c1, c2])
    assert len(result) == 1
    assert result[0].start_line == 1
    assert result[0].end_line == 50


def test_non_adjacent_chunks_not_merged(tmp_path):
    """Chunks with a gap between them are not merged."""
    f = tmp_path / "module.py"
    lines = [f"line {i}" for i in range(100)]
    f.write_text("\n".join(lines))

    c1 = _make_chunk("c1", str(f), start=1, end=10)
    c2 = _make_chunk("c2", str(f), start=15, end=25)  # gap of 4 lines

    result = merge_and_dedup([c1, c2])
    assert len(result) == 2


def test_100_line_cap(tmp_path):
    """Merging stops when merged block would exceed 100 lines."""
    f = tmp_path / "big.py"
    lines = [f"line {i}" for i in range(200)]
    f.write_text("\n".join(lines))

    c1 = _make_chunk("c1", str(f), start=1, end=80)
    c2 = _make_chunk("c2", str(f), start=81, end=130)  # merge would be 130 lines > 100

    result = merge_and_dedup([c1, c2])
    # Should not merge because merged size would be 130 > 100
    assert len(result) == 2


def test_merge_within_100_lines(tmp_path):
    """Merging is allowed when result is exactly 100 lines."""
    f = tmp_path / "ok.py"
    lines = [f"line {i}" for i in range(200)]
    f.write_text("\n".join(lines))

    c1 = _make_chunk("c1", str(f), start=1, end=50)
    c2 = _make_chunk("c2", str(f), start=51, end=100)  # merge = 100 lines exactly

    result = merge_and_dedup([c1, c2])
    assert len(result) == 1
    assert result[0].start_line == 1
    assert result[0].end_line == 100


# ---------------------------------------------------------------------------
# Multi-file grouping
# ---------------------------------------------------------------------------

def test_multi_file_chunks_not_merged(tmp_path):
    """Chunks from different files are never merged."""
    f1 = tmp_path / "a.py"
    f1.write_text("code\n" * 50)
    f2 = tmp_path / "b.py"
    f2.write_text("code\n" * 50)

    c1 = _make_chunk("c1", str(f1), start=1, end=20)
    c2 = _make_chunk("c2", str(f2), start=21, end=30)  # different file, adjacent lines

    result = merge_and_dedup([c1, c2])
    assert len(result) == 2
    files = {c.file_path for c in result}
    assert len(files) == 2


def test_results_sorted_by_score_descending(tmp_path):
    """Output is sorted by score descending."""
    f = tmp_path / "test.py"
    lines = [f"line {i}" for i in range(100)]
    f.write_text("\n".join(lines))

    c1 = _make_chunk("c1", str(f), start=1, end=5, score=0.9)
    c2 = _make_chunk("c2", str(f), start=10, end=15, score=0.3)
    c3 = _make_chunk("c3", str(f), start=20, end=25, score=0.7)

    result = merge_and_dedup([c2, c3, c1])  # out of order input
    scores = [c.score for c in result]
    assert scores == sorted(scores, reverse=True)
