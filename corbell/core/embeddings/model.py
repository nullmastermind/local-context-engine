"""Embedding model interface + SentenceTransformers implementation."""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import List, Optional


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


class GoogleEmbeddingModel(EmbeddingModel):
    """Google AI (Gemini) embedding model via the google-genai SDK.

    Uses ``gemini-embedding-001`` by default (768-dim, text-only).
    Requires ``pip install corbell[google]`` and ``GOOGLE_API_KEY``.

    Supports a comma-separated list of API keys for round-robin distribution
    and automatic failover when a key is invalid or quota-exhausted.

    Supports ``task_type`` to improve retrieval quality:
    - ``RETRIEVAL_DOCUMENT`` for indexing (default)
    - ``RETRIEVAL_QUERY`` for query-time encoding
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

    def encode(self, texts: List[str], task_type: str = "RETRIEVAL_DOCUMENT") -> List[List[float]]:
        """Encode a list of texts into embedding vectors.

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

        start = self._key_index
        errors: List[str] = []
        for i in range(len(self._api_keys)):
            idx = (start + i) % len(self._api_keys)
            key = self._api_keys[idx]
            try:
                client = genai.Client(api_key=key)
                result = client.models.embed_content(
                    model=self.model_name,
                    contents=texts,
                    config=types.EmbedContentConfig(
                        task_type=task_type,
                        output_dimensionality=768,
                    ),
                )
                self._key_index = (idx + 1) % len(self._api_keys)
                return [emb.values for emb in result.embeddings]
            except Exception as e:
                if _is_google_key_error(e):
                    errors.append(f"key[{idx}]: {e}")
                    continue
                raise

        raise RuntimeError(
            f"All {len(self._api_keys)} Google API key(s) failed for embedding:\n"
            + "\n".join(errors)
        )

    @property
    def dimension(self) -> int:
        return 768
