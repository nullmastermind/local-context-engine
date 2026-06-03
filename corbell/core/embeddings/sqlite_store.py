"""SQLite backing store for code embeddings.

Stores embedding vectors as binary blobs and provides cosine-similarity search.
Implements :class:`~corbell.core.embeddings.base.EmbeddingStore`.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from corbell.core.embeddings.base import EmbeddingStore
from corbell.core.embeddings.extractor import EmbeddingRecord

_CREATE_CHUNKS = """
CREATE TABLE IF NOT EXISTS embedding_chunks (
    id TEXT PRIMARY KEY,
    service_id TEXT NOT NULL,
    repo TEXT NOT NULL,
    file_path TEXT NOT NULL,
    start_line INTEGER,
    end_line INTEGER,
    content TEXT NOT NULL,
    language TEXT NOT NULL,
    chunk_type TEXT NOT NULL,
    symbol TEXT,
    embedding BLOB
);
"""
_CREATE_IDX = "CREATE INDEX IF NOT EXISTS idx_chunks_service ON embedding_chunks(service_id);"


class SQLiteEmbeddingStore(EmbeddingStore):
    """SQLite-backed embedding store with cosine-similarity search.

    The embedding vector is stored as a raw float32 blob for compactness.
    Implements :class:`~corbell.core.embeddings.base.EmbeddingStore`.
    """

    def __init__(self, db_path: Path | str):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._conn() as conn:
            conn.execute(_CREATE_CHUNKS)
            conn.execute(_CREATE_IDX)
            conn.commit()

    # ------------------------------------------------------------------ #
    # Write                                                                #
    # ------------------------------------------------------------------ #

    def upsert(self, record: EmbeddingRecord) -> None:
        """Insert or replace a single embedding record."""
        emb_blob = self._vec_to_blob(record.embedding) if record.embedding else None
        with self._conn() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO embedding_chunks
                   (id, service_id, repo, file_path, start_line, end_line,
                    content, language, chunk_type, symbol, embedding)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    record.id,
                    record.service_id,
                    record.repo,
                    record.file_path,
                    record.start_line,
                    record.end_line,
                    record.content,
                    record.language,
                    record.chunk_type,
                    record.symbol,
                    emb_blob,
                ),
            )
            conn.commit()

    def upsert_batch(self, records: List[EmbeddingRecord]) -> None:
        """Bulk-upsert a list of records."""
        with self._conn() as conn:
            for record in records:
                emb_blob = self._vec_to_blob(record.embedding) if record.embedding else None
                conn.execute(
                    """INSERT OR REPLACE INTO embedding_chunks
                       (id, service_id, repo, file_path, start_line, end_line,
                        content, language, chunk_type, symbol, embedding)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        record.id,
                        record.service_id,
                        record.repo,
                        record.file_path,
                        record.start_line,
                        record.end_line,
                        record.content,
                        record.language,
                        record.chunk_type,
                        record.symbol,
                        emb_blob,
                    ),
                )
            conn.commit()

    # ------------------------------------------------------------------ #
    # Read / Search                                                        #
    # ------------------------------------------------------------------ #

    def query(
        self,
        query_embedding: List[float],
        service_ids: Optional[List[str]] = None,
        top_k: int = 10,
    ) -> List[EmbeddingRecord]:
        """Return top-K most similar records by cosine similarity.

        Args:
            query_embedding: Query vector.
            service_ids: Restrict search to these service IDs (None = all).
            top_k: Number of results to return.

        Returns:
            List of :class:`EmbeddingRecord` ordered by descending similarity.
        """
        qvec = np.array(query_embedding, dtype=np.float32)
        qnorm = np.linalg.norm(qvec)
        if qnorm == 0:
            return []

        with self._conn() as conn:
            if service_ids:
                placeholders = ",".join("?" * len(service_ids))
                rows = conn.execute(
                    f"SELECT * FROM embedding_chunks WHERE service_id IN ({placeholders}) "
                    f"AND embedding IS NOT NULL",
                    service_ids,
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM embedding_chunks WHERE embedding IS NOT NULL"
                ).fetchall()

        if not rows:
            return []

        # Compute cosine similarities
        scored: List[Tuple[float, sqlite3.Row]] = []
        for row in rows:
            vec = self._blob_to_vec(row["embedding"])
            if vec is None:
                continue
            sim = float(np.dot(qvec, vec) / (qnorm * np.linalg.norm(vec) + 1e-10))
            scored.append((sim, row))

        scored.sort(key=lambda x: x[0], reverse=True)
        top = scored[:top_k]

        return [self._row_to_record(row) for _, row in top]

    def count(self, service_id: Optional[str] = None) -> int:
        """Return number of stored chunks."""
        with self._conn() as conn:
            if service_id:
                return conn.execute(
                    "SELECT COUNT(*) FROM embedding_chunks WHERE service_id = ?", (service_id,)
                ).fetchone()[0]
            return conn.execute("SELECT COUNT(*) FROM embedding_chunks").fetchone()[0]

    def clear(self, service_id: Optional[str] = None) -> None:
        """Delete all chunks, or only those for a service."""
        with self._conn() as conn:
            if service_id:
                conn.execute(
                    "DELETE FROM embedding_chunks WHERE service_id = ?", (service_id,)
                )
            else:
                conn.execute("DELETE FROM embedding_chunks")
            conn.commit()

    def delete_by_file(self, file_path: str, repo_id: str) -> int:
        """Delete all chunks for a specific file in a repo.

        Args:
            file_path: Relative file path within the repo.
            repo_id: The service/repo ID.

        Returns:
            Number of rows deleted.
        """
        with self._conn() as conn:
            cursor = conn.execute(
                "DELETE FROM embedding_chunks WHERE file_path = ? AND service_id = ?",
                (file_path, repo_id),
            )
            conn.commit()
            return cursor.rowcount

    def get_all_vectors(self) -> List[Tuple[str, bytes]]:
        """Return all chunk IDs and their raw embedding blobs.

        Used by :class:`~corbell.core.embeddings.search_cache.EmbeddingSearchCache`
        to load all vectors into memory at once.

        Returns:
            List of ``(chunk_id, raw_blob)`` tuples for rows that have an embedding.
        """
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT id, embedding FROM embedding_chunks WHERE embedding IS NOT NULL"
            ).fetchall()
        return [(row["id"], row["embedding"]) for row in rows]

    def get_chunks_by_ids(self, chunk_ids: List[str]) -> List[EmbeddingRecord]:
        """Fetch full EmbeddingRecord objects for the given IDs.

        Args:
            chunk_ids: List of chunk IDs to retrieve.

        Returns:
            List of :class:`EmbeddingRecord` objects (order not guaranteed).
        """
        if not chunk_ids:
            return []
        with self._conn() as conn:
            placeholders = ",".join("?" * len(chunk_ids))
            rows = conn.execute(
                f"SELECT * FROM embedding_chunks WHERE id IN ({placeholders})",
                chunk_ids,
            ).fetchall()
        return [self._row_to_record(row) for row in rows]

    # ------------------------------------------------------------------ #
    # Serialization helpers                                                #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _vec_to_blob(vec: List[float]) -> bytes:
        arr = np.array(vec, dtype=np.float32)
        return arr.tobytes()

    @staticmethod
    def _blob_to_vec(blob: bytes) -> Optional[np.ndarray]:
        if not blob:
            return None
        return np.frombuffer(blob, dtype=np.float32)

    @staticmethod
    def _row_to_record(row: sqlite3.Row) -> EmbeddingRecord:
        vec = SQLiteEmbeddingStore._blob_to_vec(row["embedding"])
        return EmbeddingRecord(
            id=row["id"],
            service_id=row["service_id"],
            repo=row["repo"],
            file_path=row["file_path"],
            start_line=row["start_line"] or 0,
            end_line=row["end_line"] or 0,
            content=row["content"],
            language=row["language"],
            chunk_type=row["chunk_type"],
            symbol=row["symbol"],
            embedding=vec.tolist() if vec is not None else None,
        )
