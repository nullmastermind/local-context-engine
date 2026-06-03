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


class GoogleEmbeddingModel(EmbeddingModel):
    """Google AI (Gemini) embedding model via the google-genai SDK.

    Uses ``gemini-embedding-001`` by default (768-dim, text-only).
    Requires ``pip install corbell[google]`` and ``GOOGLE_API_KEY``.

    Supports ``task_type`` to improve retrieval quality:
    - ``RETRIEVAL_DOCUMENT`` for indexing (default)
    - ``RETRIEVAL_QUERY`` for query-time encoding
    """

    def __init__(self, model_name: str = "gemini-embedding-001", api_key: Optional[str] = None):
        self.model_name = model_name
        self._api_key = api_key or os.environ.get("GOOGLE_API_KEY")
        if self._api_key is None:
            raise ValueError(
                "GOOGLE_API_KEY is not set. "
                "Set it in your environment or workspace.yaml:\n"
                "  export GOOGLE_API_KEY=AIza...\n"
                "Or use a local embedding model (e.g. all-MiniLM-L6-v2) in storage.model."
            )

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

        client = genai.Client(api_key=self._api_key)
        result = client.models.embed_content(
            model=self.model_name,
            contents=texts,
            config=types.EmbedContentConfig(
                task_type=task_type,
                output_dimensionality=768,
            ),
        )
        return [emb.values for emb in result.embeddings]

    @property
    def dimension(self) -> int:
        return 768
