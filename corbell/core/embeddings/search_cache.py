"""In-memory numpy matrix cache for fast vectorized cosine similarity search."""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional, Tuple

import numpy as np

if TYPE_CHECKING:
    from corbell.core.embeddings.sqlite_store import SQLiteEmbeddingStore


class EmbeddingSearchCache:
    """Process-resident cache of all embedding vectors as a numpy matrix.

    Loads all vectors from the embedding store once and performs vectorized
    cosine similarity using matrix multiplication (O(n) instead of O(n) with
    constant factors ~150x faster than row-by-row).

    For 30K chunks @ 384 dims: ~46 MB memory, ~2ms per query.
    """

    def __init__(self) -> None:
        self._ids: List[str] = []
        self._matrix: Optional[np.ndarray] = None  # shape (n_chunks, dim)

    @property
    def is_loaded(self) -> bool:
        """True if the cache has been loaded from the store."""
        return self._matrix is not None and len(self._ids) > 0

    def load(self, store: "SQLiteEmbeddingStore") -> None:
        """Load all embedding vectors from the store into a numpy matrix.

        Args:
            store: The embedding store to load vectors from.
        """
        rows = store.get_all_vectors()
        if not rows:
            self._ids = []
            self._matrix = None
            return

        ids: List[str] = []
        vecs: List[np.ndarray] = []

        for chunk_id, blob in rows:
            vec = np.frombuffer(blob, dtype=np.float32)
            nrm = np.linalg.norm(vec)
            if nrm > 0:
                vecs.append(vec / nrm)  # pre-normalize for cosine via dot product
            else:
                vecs.append(vec)
            ids.append(chunk_id)

        self._ids = ids
        self._matrix = np.stack(vecs, axis=0)  # shape (n_chunks, dim)

    def search(self, query_vec: np.ndarray, top_k: int = 50) -> List[Tuple[str, float]]:
        """Search for the top-K most similar chunks.

        Args:
            query_vec: Query embedding vector (will be normalized internally).
            top_k: Number of results to return.

        Returns:
            List of ``(chunk_id, score)`` tuples ordered by descending similarity.
            Returns empty list if cache is not loaded.
        """
        if not self.is_loaded or self._matrix is None:
            return []

        qvec = np.array(query_vec, dtype=np.float32)
        qnorm = float(np.linalg.norm(qvec))
        if qnorm == 0:
            return []
        qvec = qvec / qnorm  # normalize query

        # Vectorized cosine similarity: matrix @ query_vec
        # (matrix rows are already normalized, so dot product = cosine similarity)
        scores: np.ndarray = self._matrix @ qvec  # shape (n_chunks,)

        n = len(self._ids)
        actual_k = min(top_k, n)

        # Use argpartition for O(n) top-K selection instead of full sort
        if actual_k < n:
            # argpartition gives the indices of the top-k (unsorted)
            top_indices = np.argpartition(scores, -actual_k)[-actual_k:]
            # Sort only the top-k indices by score (descending)
            top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]
        else:
            top_indices = np.argsort(scores)[::-1]

        return [(self._ids[int(i)], float(scores[int(i)])) for i in top_indices]
