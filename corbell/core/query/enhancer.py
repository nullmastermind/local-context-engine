"""Query enhancement: LLM-based query expansion and keyword extraction."""

from __future__ import annotations

import re
from typing import Any, List, Optional, Tuple


def enhance_query(
    query: str,
    llm_client: Optional[Any],
) -> Tuple[List[str], List[str]]:
    """Expand a user query into search queries and keywords.

    With LLM configured: generates 3 natural-language search queries describing
    what the relevant code would *do* (not technology names).

    Without LLM: returns the original query as the sole search query and
    extracts simple keywords via regex.

    Args:
        query: The user's natural language query.
        llm_client: An LLMClient instance (or None / unconfigured).

    Returns:
        Tuple of (search_queries, keywords):
        - search_queries: List of strings to embed and search with.
        - keywords: List of extracted keywords for graph expansion hints.
    """
    if llm_client is not None and getattr(llm_client, "is_configured", False):
        return _enhance_with_llm(query, llm_client)
    else:
        return _enhance_without_llm(query)


def _enhance_with_llm(
    query: str, llm_client: Any
) -> Tuple[List[str], List[str]]:
    """Use LLM to generate 3 code-oriented search queries."""
    system = (
        "You are a code search assistant. Given a user query about code, "
        "generate exactly 3 different natural-language search queries that describe "
        "what the relevant implementation code *does* (not technology names or framework names). "
        "Each query should describe behavior, logic, or data transformations. "
        "Return exactly 3 queries, one per line, no numbering, no extra text."
    )
    user = f"User query: {query}\n\nGenerate 3 code search queries:"

    try:
        response = llm_client.call(system, user, max_tokens=300, temperature=0.1)
        lines = [line.strip() for line in response.strip().splitlines() if line.strip()]
        # Take up to 3 non-empty lines
        search_queries = lines[:3]
        if not search_queries:
            search_queries = [query]
    except Exception:
        search_queries = [query]

    # Extract keywords from the original query for graph expansion
    keywords = _extract_keywords(query)
    return search_queries, keywords


def _enhance_without_llm(query: str) -> Tuple[List[str], List[str]]:
    """Simple enhancement without LLM: use query as-is, extract keywords via regex."""
    keywords = _extract_keywords(query)
    return [query], keywords


def _extract_keywords(text: str) -> List[str]:
    """Extract meaningful keywords from text via regex."""
    # Extract words that look like identifiers (camelCase, snake_case, etc.)
    words = re.findall(r"[a-zA-Z][a-zA-Z0-9_]*", text)

    # Filter stop words and short words
    stop_words = {
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
        "of", "with", "by", "from", "up", "about", "into", "through", "during",
        "is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
        "do", "does", "did", "will", "would", "could", "should", "may", "might",
        "must", "shall", "can", "how", "what", "when", "where", "which", "who",
        "that", "this", "these", "those", "it", "its", "get", "set", "use",
        "new", "return", "class", "function", "method", "var", "let", "const",
        "def", "import", "from", "as", "if", "else", "while", "for", "try",
        "except", "raise", "pass", "break", "continue", "not", "and", "or",
    }

    keywords = [
        w for w in words
        if len(w) > 2 and w.lower() not in stop_words
    ]

    # Remove duplicates while preserving order
    seen: set = set()
    unique_keywords = []
    for kw in keywords:
        lower = kw.lower()
        if lower not in seen:
            seen.add(lower)
            unique_keywords.append(kw)

    return unique_keywords[:20]  # cap at 20 keywords
