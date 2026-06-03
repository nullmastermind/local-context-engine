"""Embedding model interface + SentenceTransformers implementation."""

from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from typing import List, Optional

logger = logging.getLogger(__name__)


class EmbeddingModel(ABC):
    """Abstract embedding model interface."""

    @abstractmethod
    def encode(self, texts: List[str]) -> List[List[float]]:
        """Encode a list of texts into embedding vectors.

        Args:
            texts: List of text strings to encode.

        Returns:
            List of float vectors (one per input text).
        """
        ...

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the embedding dimension."""
        ...


class SentenceTransformerModel(EmbeddingModel):
    """Wraps ``sentence-transformers`` with lazy loading.

    Uses ``all-MiniLM-L6-v2`` by default (384-dim, fast, no API key).
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None  # lazy-loaded

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(f"sentence-transformers/{self.model_name}")
        return self._model

    def encode(self, texts: List[str]) -> List[List[float]]:
        model = self._get_model()
        vecs = model.encode(texts, show_progress_bar=False)
        return [v.tolist() for v in vecs]

    @property
    def dimension(self) -> int:
        return self._get_model().get_sentence_embedding_dimension()


def _is_google_key_error(e: Exception) -> bool:
    """Return True when a Google API error is caused by the key, not the request."""
    code = getattr(e, "code", None)
    if code in (401, 403, 429):
        return True
    if code == 400:
        msg = (getattr(e, "message", None) or str(e)).lower()
        return "api key" in msg
    return False


def _is_rate_limit_error(e: Exception) -> bool:
    """Return True when a Google API error is a 429 RESOURCE_EXHAUSTED rate limit."""
    return getattr(e, "code", None) == 429


def _parse_gemini_version(model_name: str) -> int:
    """Parse the version number from a gemini-embedding model name.

    Examples:
        ``gemini-embedding-001`` → 1
        ``gemini-embedding-2`` → 2

    Returns 0 if the version cannot be parsed.
    """
    prefix = "gemini-embedding-"
    if not model_name.startswith(prefix):
        return 0
    suffix = model_name[len(prefix):]
    try:
        return int(suffix)
    except ValueError:
        return 0


class GoogleEmbeddingModel(EmbeddingModel):
    """Google AI (Gemini) embedding model via the google-genai SDK.

    Uses ``gemini-embedding-001`` by default (768-dim, text-only).
    Requires ``pip install corbell[google]`` and ``GOOGLE_API_KEY``.

    Supports a comma-separated list of API keys for round-robin distribution
    and automatic failover when a key is invalid or quota-exhausted.

    Supports ``task_type`` to improve retrieval quality:
    - ``RETRIEVAL_DOCUMENT`` for indexing (default)
    - ``RETRIEVAL_QUERY`` for query-time encoding

    For ``gemini-embedding-2`` and later, inline text prefixes are used
    instead of relying solely on ``task_type`` for better retrieval quality.
    Use ``prepare_query`` and ``prepare_document`` to format texts before
    passing them to ``encode``.
    """

    def __init__(self, model_name: str = "gemini-embedding-001", api_key: Optional[str] = None):
        self.model_name = model_name
        raw = api_key or os.environ.get("GOOGLE_API_KEY") or ""
        self._api_keys: List[str] = [k.strip() for k in raw.split(",") if k.strip()]
        if not self._api_keys:
            raise ValueError(
                "GOOGLE_API_KEY is not set. "
                "Set it in your environment or workspace.yaml:\n"
                "  export GOOGLE_API_KEY=AIza...\n"
                "Or use a local embedding model (e.g. all-MiniLM-L6-v2) in storage.model."
            )
        self._key_index: int = 0
        # kept for backwards-compat with tests that read _api_key directly
        self._api_key: str = self._api_keys[0]

    @property
    def uses_prefix_format(self) -> bool:
        """Return True when the model requires inline text prefixes for best quality.

        Activated for ``gemini-embedding-2`` and all later versions (version >= 2).
        """
        return _parse_gemini_version(self.model_name) >= 2

    def prepare_query(self, query: str) -> str:
        """Format a query string with a task prefix for retrieval.

        Only applies the prefix when ``uses_prefix_format`` is True.
        """
        if self.uses_prefix_format:
            return f"task: code retrieval | query: {query}"
        return query

    def prepare_document(self, content: str, title: Optional[str] = None) -> str:
        """Format a document chunk with a title prefix for indexing.

        Only applies the prefix when ``uses_prefix_format`` is True.

        Args:
            content: Raw chunk text.
            title: Descriptive title, typically ``"{file_path}:{symbol}"``
                   or ``"{file_path}:L{start}-{end}"``. Defaults to ``"none"``.
        """
        if self.uses_prefix_format:
            resolved_title = title or "none"
            return f"title: {resolved_title} | text: {content}"
        return content

    _BATCH_SIZE = 100
    _BASE_DELAY = 2.0
    _MAX_BACKOFF = 60.0

    def encode(self, texts: List[str], task_type: str = "RETRIEVAL_DOCUMENT") -> List[List[float]]:
        """Encode a list of texts into embedding vectors.

        Batches requests (100 texts/batch) and retries on rate limit (429)
        with exponential backoff.

        Args:
            texts: List of text strings to encode.
            task_type: Task type hint for the embedding model.
                Use ``RETRIEVAL_DOCUMENT`` when indexing, ``RETRIEVAL_QUERY`` at query time.

        Returns:
            List of float vectors (one per input text).
        """
        try:
            from google import genai
            from google.genai import types
        except ImportError:
            raise ImportError("pip install corbell[google]")

        all_embeddings: List[List[float]] = []
        for batch_start in range(0, len(texts), self._BATCH_SIZE):
            batch = texts[batch_start:batch_start + self._BATCH_SIZE]
            contents = [types.Content(parts=[types.Part(text=t)]) for t in batch]
            batch_result = self._embed_batch_with_retry(
                contents, task_type, genai, types
            )
            all_embeddings.extend(batch_result)

        return all_embeddings

    def _embed_batch_with_retry(
        self, contents, task_type: str, genai, types
    ) -> List[List[float]]:
        """Embed a single batch, rotating keys and retrying on rate limit.

        - If all keys fail with 429 (rate limit), waits with capped exponential
          backoff and retries indefinitely until the quota is restored.
        - If any key fails with a non-key error, raises immediately.
        - If all keys fail with auth errors (401/403/400+apikey), raises immediately.
        """
        import time

        start = self._key_index
        rate_limit_attempt = 0

        while True:
            errors: List[str] = []
            all_rate_limited = True

            for i in range(len(self._api_keys)):
                idx = (start + i) % len(self._api_keys)
                key = self._api_keys[idx]
                try:
                    client = genai.Client(api_key=key)
                    result = client.models.embed_content(
                        model=self.model_name,
                        contents=contents,
                        config=types.EmbedContentConfig(
                            task_type=task_type,
                            output_dimensionality=768,
                        ),
                    )
                    self._key_index = (idx + 1) % len(self._api_keys)
                    return [emb.values for emb in result.embeddings]
                except Exception as e:
                    if _is_google_key_error(e):
                        if not _is_rate_limit_error(e):
                            # Auth failure (401/403/400+apikey) — not a transient error
                            all_rate_limited = False
                        errors.append(f"key[{idx}]: {e}")
                        continue
                    raise

            if not all_rate_limited:
                raise RuntimeError(
                    f"All {len(self._api_keys)} Google API key(s) failed with auth errors:\n"
                    + "\n".join(errors)
                )

            # All keys are rate-limited (429) — wait and retry indefinitely
            delay = min(self._BASE_DELAY * (2 ** rate_limit_attempt), self._MAX_BACKOFF)
            rate_limit_attempt += 1
            logger.warning(
                "All %d Google API key(s) rate-limited (429). "
                "Retrying in %.0fs (attempt %d)...",
                len(self._api_keys),
                delay,
                rate_limit_attempt,
            )
            time.sleep(delay)

    @property
    def dimension(self) -> int:
        return 768
