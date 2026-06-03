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
    """Rerank query results using an LLM for relevance filtering.

    Sends only metadata (no full source code) to the LLM to keep prompts small.
    The LLM returns a JSON array of chunk_ids ordered by relevance.

    On any failure (parse error, invalid IDs, LLM timeout), all chunk IDs are
    returned in their original order (graceful fallback — never loses results).

    Args:
        query: The original user query.
        chunks: Scored chunks to rerank.
        llm_client: An LLMClient instance (or None / unconfigured).

    Returns:
        List of chunk_ids in reranked order (most relevant first).
        Falls back to original order on any failure.
    """
    if not chunks:
        return []

    all_ids = [c.chunk_id for c in chunks]

    if llm_client is None or not getattr(llm_client, "is_configured", False):
        return all_ids

    # Build metadata-only payload (no source code — keeps prompt small)
    metadata = []
    for chunk in chunks:
        meta = {
            "chunk_id": chunk.chunk_id,
            "file": chunk.file_path,
            "symbol": chunk.symbol or "",
            "type": chunk.chunk_type,
            "lines": f"{chunk.start_line}-{chunk.end_line}",
        }
        metadata.append(meta)

    system = (
        "You are a code search relevance ranker. "
        "Given a user query and a list of code chunks (metadata only), "
        "return a JSON array of chunk_ids ordered from most relevant to least relevant. "
        "Include only chunk_ids that are actually relevant to the query. "
        "Return ONLY a valid JSON array of strings, nothing else."
    )

    user = (
        f"Query: {query}\n\n"
        f"Chunks:\n{json.dumps(metadata, indent=2)}\n\n"
        "Return JSON array of relevant chunk_ids (most relevant first):"
    )

    try:
        response = llm_client.call(
            system, user,
            max_tokens=1000,
            temperature=0.0,
        )

        # Parse the JSON response
        text = response.strip()
        # Handle markdown code blocks
        if text.startswith("```"):
            lines = text.splitlines()
            text = "\n".join(
                line for line in lines
                if not line.startswith("```")
            ).strip()

        reranked_ids = json.loads(text)

        if not isinstance(reranked_ids, list):
            return all_ids

        # Validate that returned IDs are strings and exist in the original set
        valid_id_set = set(all_ids)
        filtered = [
            chunk_id for chunk_id in reranked_ids
            if isinstance(chunk_id, str) and chunk_id in valid_id_set
        ]

        if not filtered:
            return all_ids

        return filtered

    except Exception:
        # Graceful fallback: return all IDs in original order
        return all_ids
