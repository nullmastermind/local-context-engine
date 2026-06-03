"""LLM-based result reranker for the query pipeline."""

from __future__ import annotations

import json
from typing import Any, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from corbell.core.query.graph_expander import ScoredChunk


def rerank_chunks(
    query: str,
    chunks: List["ScoredChunk"],
    llm_client: Optional[Any],
) -> List[str]:
    """Rerank and filter query results using an LLM.

    Sends chunk content + metadata to the LLM. The LLM returns a JSON array
    of 0-based chunk indices ordered by relevance, omitting irrelevant chunks.

    On any failure (parse error, LLM timeout), all chunk IDs are returned
    in their original order (graceful fallback).

    Args:
        query: The original user query.
        chunks: Scored chunks to rerank.
        llm_client: An LLMClient instance (or None / unconfigured).

    Returns:
        List of chunk_ids in reranked order (most relevant first).
        Irrelevant chunks are excluded.
        Falls back to original order on any failure.
    """
    if not chunks:
        return []

    all_ids = [c.chunk_id for c in chunks]

    if llm_client is None or not getattr(llm_client, "is_configured", False):
        return all_ids

    # Build payload with code content, indexed for compact LLM output
    entries = []
    for i, chunk in enumerate(chunks):
        entry = (
            f"[{i}] {chunk.file_path}:{chunk.start_line}-{chunk.end_line}"
            f" ({chunk.chunk_type}, {chunk.symbol or 'no symbol'})\n"
            f"{chunk.content}"
        )
        entries.append(entry)

    system = (
        "You are a code search relevance ranker. "
        "Given a query and numbered code chunks, return a JSON array of chunk "
        "indices (integers) ordered from most relevant to least relevant. "
        "OMIT chunks that are not relevant to the query. "
        "Return ONLY a valid JSON array of integers, e.g. [2,0,5]."
    )

    separator = "---\n"
    chunks_text = separator.join(entries)

    user = (
        f"Query: {query}\n\n"
        f"Chunks:\n{chunks_text}\n\n"
        "Return JSON array of relevant chunk indices (most relevant first):"
    )

    try:
        response = llm_client.call(
            system, user,
            max_tokens=200,
            temperature=0.0,
        )

        text = response.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            text = "\n".join(
                line for line in lines
                if not line.startswith("```")
            ).strip()

        indices = json.loads(text)

        if not isinstance(indices, list):
            return all_ids

        # Validate indices are ints within range
        n = len(chunks)
        filtered = [
            all_ids[idx] for idx in indices
            if isinstance(idx, int) and 0 <= idx < n
        ]

        if not filtered:
            return all_ids

        return filtered

    except Exception:
        return all_ids
