"""Output formatter for code retrieval results."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, TYPE_CHECKING

if TYPE_CHECKING:
    from corbell.core.query.graph_expander import ScoredChunk


def format_results(
    chunks: List["ScoredChunk"],
    repo_paths: Dict[str, str],
    max_output_bytes: int = 80_000,
    max_line_chars: int = 1000,
) -> str:
    """Format scored chunks as annotated code blocks for LLM context injection.

    Output format:
        <absolute_path>#L<start>-<end>
        <start>: <code line>
        <start+1>: <code line>
        ...
        <end>: <code line>

    Args:
        chunks: Scored chunks to format (pre-sorted by score descending).
        repo_paths: Mapping of repo_id -> absolute repo path string.
                    Used to resolve relative file paths to absolute paths.
        max_output_bytes: Maximum total output size in bytes. Truncation stops at the
                          last complete chunk boundary that fits. Defaults to 80 000 (~20K tokens).
        max_line_chars: Maximum characters per source line before inline truncation.
                        Defaults to 1000.

    Returns:
        Formatted string with all chunks, separated by blank lines. If the output
        exceeds max_output_bytes, a trailing note reports how many results were shown.
    """
    if not chunks:
        return ""

    total = len(chunks)
    blocks: List[str] = []
    accumulated_bytes = 0
    truncation_footer = ""

    for n, chunk in enumerate(chunks):
        abs_path = _resolve_absolute_path(chunk.file_path, chunk.repo_id, repo_paths)

        # Read the actual lines for this chunk range
        lines = _read_chunk_lines(abs_path, chunk.start_line, chunk.end_line)
        if lines is None:
            # File not readable — use content from chunk object
            lines = chunk.content.splitlines()

        # Build the header: path#Lstart-end
        header = f"{abs_path}#L{chunk.start_line}-{chunk.end_line}"

        # Build numbered lines with per-line truncation
        numbered_lines: List[str] = []
        for i, line in enumerate(lines):
            line_num = chunk.start_line + i
            if len(line) > max_line_chars:
                line = line[:max_line_chars] + " [truncated — use Read tool for full content]"
            numbered_lines.append(f"{line_num}: {line}")

        block = header + "\n" + "\n".join(numbered_lines)

        # Per-output size gate: check if adding this block would exceed the limit
        # Account for the separator ("\n\n") between blocks
        separator_size = 2 if blocks else 0
        block_bytes = len(block.encode("utf-8"))
        if accumulated_bytes + separator_size + block_bytes > max_output_bytes:
            # Collect remaining chunk headers so the agent knows what else is relevant
            remaining_headers: List[str] = []
            for remaining in chunks[n:]:
                rp = _resolve_absolute_path(remaining.file_path, remaining.repo_id, repo_paths)
                remaining_headers.append(f"{rp}#L{remaining.start_line}-{remaining.end_line}")
            truncation_footer = (
                f"\n\n[Showing {n}/{total} results. "
                f"Remaining (use Read tool):\n"
                + "\n".join(remaining_headers)
                + "]"
            )
            break

        blocks.append(block)
        accumulated_bytes += separator_size + block_bytes

    return "\n\n".join(blocks) + truncation_footer


def _resolve_absolute_path(
    file_path: str,
    repo_id: str,
    repo_paths: Dict[str, str],
) -> str:
    """Resolve a file_path to an absolute path string.

    If file_path is already absolute, returns it as-is.
    Otherwise, joins it with the repo's root path.
    """
    p = Path(file_path)
    if p.is_absolute():
        return str(p)

    repo_root = repo_paths.get(repo_id, "")
    if repo_root:
        return str((Path(repo_root) / file_path).resolve())

    return file_path  # best effort


def _read_chunk_lines(
    file_path: str,
    start_line: int,
    end_line: int,
) -> List[str] | None:
    """Read lines start_line..end_line (1-based, inclusive) from a file.

    Returns None if the file cannot be read.
    """
    try:
        all_lines = Path(file_path).read_text(encoding="utf-8", errors="ignore").splitlines()
        start_idx = max(0, start_line - 1)
        end_idx = min(len(all_lines), end_line)
        return all_lines[start_idx:end_idx]
    except Exception:
        return None
