"""Deduplication and adjacent chunk merging for query results."""

from __future__ import annotations

from typing import Dict, List, TYPE_CHECKING

if TYPE_CHECKING:
    from corbell.core.query.graph_expander import ScoredChunk

# Maximum number of lines in a merged chunk block
_MAX_MERGED_LINES = 100


def merge_and_dedup(chunks: List["ScoredChunk"]) -> List["ScoredChunk"]:
    """Deduplicate by chunk_id (keep max score), group by file, merge adjacent chunks.

    Steps:
    1. Deduplicate: for duplicate chunk_ids, keep the one with the highest score.
    2. Group chunks by file path.
    3. Within each file, sort by start_line and merge adjacent/overlapping chunks.
    4. Cap merged blocks at 100 lines.

    Args:
        chunks: List of ScoredChunk objects (may have duplicates from multiple queries).

    Returns:
        Deduplicated and merged list of ScoredChunk objects.
    """
    if not chunks:
        return []

    # Step 1: Dedup by chunk_id, keeping max score
    best: Dict[str, "ScoredChunk"] = {}
    for chunk in chunks:
        if chunk.chunk_id not in best or chunk.score > best[chunk.chunk_id].score:
            best[chunk.chunk_id] = chunk

    deduped = list(best.values())

    # Step 2: Group by file path
    by_file: Dict[str, List["ScoredChunk"]] = {}
    for chunk in deduped:
        by_file.setdefault(chunk.file_path, []).append(chunk)

    # Step 3+4: Sort and merge adjacent/overlapping chunks per file
    result: List["ScoredChunk"] = []
    for file_path, file_chunks in by_file.items():
        merged = _merge_file_chunks(file_chunks)
        result.extend(merged)

    # Sort final result by score descending for output ordering
    result.sort(key=lambda c: c.score, reverse=True)
    return result


def _merge_file_chunks(chunks: List["ScoredChunk"]) -> List["ScoredChunk"]:
    """Merge adjacent/overlapping chunks within a single file.

    Two chunks are merged if next.start_line <= current.end_line + 1.
    The merged chunk gets the max score and a fresh chunk_id combining both.
    Merged blocks are capped at 100 lines.
    """
    # Sort by start line
    sorted_chunks = sorted(chunks, key=lambda c: c.start_line)

    merged: List["ScoredChunk"] = []
    if not sorted_chunks:
        return merged

    current = sorted_chunks[0]

    for next_chunk in sorted_chunks[1:]:
        # Check adjacency: next starts before or at current.end + 1
        if next_chunk.start_line <= current.end_line + 1:
            # Check if merging would exceed 100-line cap
            merged_lines = next_chunk.end_line - current.start_line + 1
            if merged_lines <= _MAX_MERGED_LINES:
                # Merge: extend current to cover next_chunk
                current = _merge_two(current, next_chunk)
            else:
                # Would exceed cap: flush current, start new
                merged.append(current)
                current = next_chunk
        else:
            # Not adjacent: flush current, start new
            merged.append(current)
            current = next_chunk

    merged.append(current)
    return merged


def _merge_two(a: "ScoredChunk", b: "ScoredChunk") -> "ScoredChunk":
    """Merge two chunks into one, combining their content and taking the max score."""
    from corbell.core.query.graph_expander import ScoredChunk  # local import to avoid circular

    new_start = min(a.start_line, b.start_line)
    new_end = max(a.end_line, b.end_line)

    # Rebuild content by merging lines if both have content
    # We read the combined content from file to avoid duplication.
    # If reading fails, concatenate with a separator.
    try:
        lines = _read_lines(a.file_path, new_start, new_end)
        content = "\n".join(lines)
    except Exception:
        # Fall back: concatenate without overlap
        content = a.content + "\n" + b.content

    combined_id = f"{a.chunk_id}+{b.chunk_id}"

    return ScoredChunk(
        chunk_id=combined_id,
        score=max(a.score, b.score),
        file_path=a.file_path,
        start_line=new_start,
        end_line=new_end,
        content=content,
        repo_id=a.repo_id,
        symbol=a.symbol,
        chunk_type=a.chunk_type,
        language=a.language,
    )


def _read_lines(file_path: str, start_line: int, end_line: int) -> List[str]:
    """Read specific lines from a file (1-based, inclusive)."""
    from pathlib import Path
    path = Path(file_path)
    all_lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    start_idx = max(0, start_line - 1)
    end_idx = min(len(all_lines), end_line)
    return all_lines[start_idx:end_idx]
