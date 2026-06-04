"""Index tracking: file-level mtime tracking and global index metadata."""

from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from corbell.core.workspace import WorkspaceConfig

_CREATE_FILE_META = """
CREATE TABLE IF NOT EXISTS file_index_meta (
    file_path TEXT NOT NULL,
    repo_id TEXT NOT NULL,
    mtime REAL NOT NULL,
    indexed_at REAL NOT NULL,
    PRIMARY KEY (file_path, repo_id)
);
"""

_CREATE_INDEX_META = """
CREATE TABLE IF NOT EXISTS index_meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


@dataclass
class StaleResult:
    """Result of a stale-file detection scan."""

    added: List[tuple] = field(default_factory=list)      # [(file_path, repo_id), ...]
    modified: List[tuple] = field(default_factory=list)   # [(file_path, repo_id), ...]
    deleted: List[tuple] = field(default_factory=list)    # [(file_path, repo_id), ...]

    @property
    def has_changes(self) -> bool:
        """True if any files need to be reindexed."""
        return bool(self.added or self.modified or self.deleted)

    @property
    def changed_repo_ids(self) -> set:
        """Set of repo IDs that have at least one changed file."""
        result = set()
        for _, repo_id in self.added:
            result.add(repo_id)
        for _, repo_id in self.modified:
            result.add(repo_id)
        for _, repo_id in self.deleted:
            result.add(repo_id)
        return result


class IndexTracker:
    """Manages file_index_meta and index_meta tables for incremental indexing.

    The file_index_meta table records each indexed file's path, repo_id, and
    mtime so the builder can detect added/modified/deleted files on subsequent
    runs without a full re-scan of every file's content.

    The index_meta table stores global key-value metadata (embedding_model,
    last_build_at, chunk_size, overlap).
    """

    def __init__(self, db_path: Path | str) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._create_tables()

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _create_tables(self) -> None:
        """Ensure the tracking tables exist."""
        with self._conn() as conn:
            conn.execute(_CREATE_FILE_META)
            conn.execute(_CREATE_INDEX_META)
            conn.commit()

    # Alias for external callers that prefer create_tables()
    def create_tables(self) -> None:
        self._create_tables()

    def get_last_build_at(self) -> Optional[float]:
        """Return the Unix timestamp of the last successful build, or None."""
        with self._conn() as conn:
            row = conn.execute(
                "SELECT value FROM index_meta WHERE key = 'last_build_at'"
            ).fetchone()
            if row:
                try:
                    return float(row["value"])
                except (ValueError, TypeError):
                    return None
        return None

    def get_stored_model(self) -> Optional[str]:
        """Return the model name stored at last build, or None."""
        with self._conn() as conn:
            row = conn.execute(
                "SELECT value FROM index_meta WHERE key = 'embedding_model'"
            ).fetchone()
            return row["value"] if row else None

    def set_meta(self, key: str, value: str) -> None:
        """Upsert a key-value pair in index_meta."""
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO index_meta (key, value) VALUES (?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (key, value),
            )
            conn.commit()

    def get_stale_files(
        self,
        repos: List,
        config: "WorkspaceConfig",
    ) -> StaleResult:
        """Detect added, modified, and deleted files across all repos.

        Args:
            repos: List of RepoConfig objects with resolved_path set.
            config: WorkspaceConfig for skip_dirs and file extension settings.

        Returns:
            StaleResult with added, modified, deleted lists of (file_path, repo_id).
        """
        from corbell.core.constants import EXTENSION_LANG, SKIP_DIRS
        from corbell.core.gitignore import load_gitignore
        from corbell.core.workspace import IndexingConfig

        indexing: IndexingConfig = config.indexing
        extra_skip = set(indexing.skip_dirs)
        all_skip = SKIP_DIRS | extra_skip

        result = StaleResult()

        # Load all currently tracked entries from DB
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT file_path, repo_id, mtime FROM file_index_meta"
            ).fetchall()

        tracked: Dict[tuple, float] = {}
        for row in rows:
            tracked[(row["file_path"], row["repo_id"])] = row["mtime"]

        # Scan current files on disk
        seen_keys: set = set()
        for repo in repos:
            repo_id = repo.id
            repo_path = repo.resolved_path
            if not repo_path or not repo_path.exists():
                continue

            gitignore_spec = load_gitignore(repo_path)

            for fp in repo_path.rglob("*"):
                if not fp.is_file():
                    continue
                # Check skip dirs
                rel = fp.relative_to(repo_path)
                if any(part in all_skip for part in rel.parts):
                    continue
                if any(part.startswith(".") for part in rel.parts[:-1]):
                    continue
                # Check gitignore
                if gitignore_spec.match_file(str(rel).replace("\\", "/")):
                    continue
                # Check extension
                if fp.suffix not in EXTENSION_LANG:
                    continue
                # Check file size
                try:
                    stat = fp.stat()
                    if stat.st_size > indexing.max_file_bytes:
                        continue
                    current_mtime = stat.st_mtime
                except OSError:
                    continue

                rel_str = str(rel)
                key = (rel_str, repo_id)
                seen_keys.add(key)

                if key not in tracked:
                    result.added.append(key)
                elif abs(current_mtime - tracked[key]) > 0.001:
                    result.modified.append(key)

        # Detect deleted files
        for key in tracked:
            if key not in seen_keys:
                result.deleted.append(key)

        return result

    def mark_indexed(self, file_path: str, repo_id: str, mtime: float) -> None:
        """Record a file as indexed with its current mtime.

        Should be called AFTER the embedding chunks have been committed to DB
        to ensure crash safety (next run will re-detect the file if mtime not updated).

        Args:
            file_path: Relative file path within the repo.
            repo_id: The repository ID.
            mtime: File modification time (from os.path.getmtime).
        """
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO file_index_meta (file_path, repo_id, mtime, indexed_at) "
                "VALUES (?, ?, ?, ?) "
                "ON CONFLICT(file_path, repo_id) DO UPDATE SET "
                "mtime = excluded.mtime, indexed_at = excluded.indexed_at",
                (file_path, repo_id, mtime, time.time()),
            )
            conn.commit()

    def remove_tracked(self, file_paths: List[tuple]) -> None:
        """Remove tracking entries for deleted files.

        Args:
            file_paths: List of ``(file_path, repo_id)`` tuples to remove.
        """
        if not file_paths:
            return
        with self._conn() as conn:
            for file_path, repo_id in file_paths:
                conn.execute(
                    "DELETE FROM file_index_meta WHERE file_path = ? AND repo_id = ?",
                    (file_path, repo_id),
                )
            conn.commit()

    def clear_all(self) -> None:
        """Remove all tracking data (used during --rebuild)."""
        with self._conn() as conn:
            conn.execute("DELETE FROM file_index_meta")
            conn.execute("DELETE FROM index_meta")
            conn.commit()
