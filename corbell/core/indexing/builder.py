"""Index builder: orchestrates full and incremental builds of the code index."""

from __future__ import annotations

import logging
import os
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from corbell.core.gitignore import load_gitignore
from corbell.core.indexing.tracker import IndexTracker

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Module-level worker function (must be picklable for multiprocessing)        #
# --------------------------------------------------------------------------- #

def _extract_file_worker(
    args: Tuple,
) -> List[Any]:
    """Extract embedding chunks from a single file.

    This is a module-level function so it can be pickled by ProcessPoolExecutor.

    Args:
        args: Tuple of
            (abs_path_str, rel_path, lang, service_id, repo_str,
             chunk_size, overlap, max_file_bytes)

    Returns:
        List of EmbeddingRecord objects (embeddings are None at this stage).
    """
    (
        abs_path_str,
        rel_path,
        lang,
        service_id,
        repo_str,
        chunk_size,
        overlap,
        max_file_bytes,  # noqa: F841 — kept for signature completeness
    ) = args

    from corbell.core.embeddings.extractor import CodeChunkExtractor

    extractor = CodeChunkExtractor(chunk_size=chunk_size, overlap=overlap)
    fp = Path(abs_path_str)
    return extractor._extract_file(fp, rel_path, lang, service_id, repo_str)


def _get_worker_count() -> int:
    """Return the number of parallel workers for indexing.

    Reads ``CORBELL_INDEX_WORKERS`` env var; defaults to ``min(cpu_count, 8)``.
    """
    env_val = os.environ.get("CORBELL_INDEX_WORKERS", "").strip()
    if env_val:
        try:
            return max(1, int(env_val))
        except ValueError:
            pass
    return min(os.cpu_count() or 4, 8)


# --------------------------------------------------------------------------- #
# Batch encoding helpers                                                       #
# --------------------------------------------------------------------------- #

_API_BATCH_SIZE = 100  # conservative limit for API-backed embedding models

# Super-batch size: number of chunks streamed to the embedding store at a time.
# Keeps peak memory bounded regardless of total corpus size.
# Override via CORBELL_EMBED_BATCH env var.
_SUPER_BATCH_SIZE: int = max(1, int(os.environ.get("CORBELL_EMBED_BATCH", "500") or "500"))

_ENCODE_MAX_RETRIES = 3
_ENCODE_BASE_DELAY = 2.0


def _encode_batch_with_retry(model: Any, batch: List[str]) -> List[Any]:
    """Encode a single text batch, retrying on transient errors.

    Rate-limit (429) is already handled inside model.encode() via key rotation
    and backoff.  This wrapper catches transient network/server errors (timeouts,
    connection resets, 5xx) and retries up to ``_ENCODE_MAX_RETRIES`` times with
    exponential backoff before propagating.
    """
    for attempt in range(_ENCODE_MAX_RETRIES):
        try:
            return model.encode(batch)
        except Exception as e:
            status = getattr(e, "status_code", None) or getattr(e, "code", None)
            is_transient = (
                isinstance(e, (OSError, TimeoutError, ConnectionError))
                or (isinstance(status, int) and status >= 500)
            )
            if not is_transient or attempt == _ENCODE_MAX_RETRIES - 1:
                raise
            delay = _ENCODE_BASE_DELAY * (2 ** attempt)
            logger.warning(
                "Transient embedding error (attempt %d/%d): %s. Retrying in %.0fs...",
                attempt + 1, _ENCODE_MAX_RETRIES, e, delay,
            )
            time.sleep(delay)
    return []  # unreachable, satisfies type checker


def _get_embed_concurrency(model: Any) -> int:
    """Return the number of concurrent embedding API threads to use.

    Checks ``CORBELL_EMBED_CONCURRENCY`` env var first; falls back to a
    provider-aware default scaled by the number of API keys configured:

    - VoyageEmbeddingModel  → 30 per key (paid tier handles 300 req/s per key)
    - GoogleEmbeddingModel  → 8 per key  (1500 req/min ≈ 25/s per key)
    - unknown               → 4

    With multiple keys the model round-robins requests across them, so each key
    sees 1/N of the traffic.  Scaling concurrency by key count exploits the full
    aggregate rate-limit budget.
    """
    env_val = os.environ.get("CORBELL_EMBED_CONCURRENCY", "").strip()
    if env_val:
        try:
            return max(1, int(env_val))
        except ValueError:
            pass

    from corbell.core.embeddings.model import GoogleEmbeddingModel, VoyageEmbeddingModel

    num_keys = max(1, len(getattr(model, "_api_keys", []) or [1]))

    if isinstance(model, VoyageEmbeddingModel):
        return 30 * num_keys
    if isinstance(model, GoogleEmbeddingModel):
        return 8 * num_keys
    return 4


def _encode_chunks(model: Any, chunks: List[Any]) -> List[Any]:
    """Encode chunks and attach embeddings in-place.

    Batches into groups of ``_API_BATCH_SIZE`` and submits each batch as a
    concurrent task via :class:`~concurrent.futures.ThreadPoolExecutor`.
    The number of concurrent threads is determined by :func:`_get_embed_concurrency`.

    Args:
        model: An EmbeddingModel instance.
        chunks: List of EmbeddingRecord objects without embeddings.

    Returns:
        The same list with ``embedding`` fields populated.
    """
    from corbell.core.embeddings.model import GoogleEmbeddingModel

    if not chunks:
        return chunks

    if isinstance(model, GoogleEmbeddingModel) and model.uses_prefix_format:
        texts = [
            model.prepare_document(
                c.content,
                title=(
                    f"{c.file_path}:{c.symbol}"
                    if c.symbol
                    else f"{c.file_path}:L{c.start_line}-{c.end_line}"
                ),
            )
            for c in chunks
        ]
    else:
        texts = [c.content for c in chunks]

    # Split texts into API-sized batches and encode concurrently.
    batches = [texts[i : i + _API_BATCH_SIZE] for i in range(0, len(texts), _API_BATCH_SIZE)]
    concurrency = _get_embed_concurrency(model)

    # Use a list pre-sized to the number of batches so we can store results in order.
    batch_vectors: List[Any] = [None] * len(batches)

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        future_to_idx = {
            executor.submit(_encode_batch_with_retry, model, batch): idx
            for idx, batch in enumerate(batches)
        }
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            batch_vectors[idx] = future.result()

    # Flatten ordered results and assign to chunks.
    vectors: List[Any] = []
    for vecs in batch_vectors:
        vectors.extend(vecs)

    for chunk, vec in zip(chunks, vectors):
        chunk.embedding = vec

    return chunks


# --------------------------------------------------------------------------- #
# Parallel file collection helpers                                             #
# --------------------------------------------------------------------------- #

def _collect_repo_files(
    repo_path: Path,
    repo_id: str,
    max_file_bytes: int,
    gitignore_spec: Any,
) -> List[Tuple[str, str, str]]:
    """Walk repo_path and return (abs_path_str, rel_path, lang) for each indexable file.

    Replicates the filtering logic from CodeChunkExtractor.extract_from_repo so
    we can dispatch individual files to worker processes.

    Args:
        repo_path: Absolute path to the repo root.
        repo_id: Repository identifier (unused here but kept for symmetry).
        max_file_bytes: Maximum file size in bytes.
        gitignore_spec: Pre-loaded PathSpec for gitignore filtering.

    Returns:
        List of (abs_path_str, rel_path, lang) tuples for picklable dispatch.
    """
    from corbell.core.constants import EXTENSION_LANG, SKIP_DIRS

    file_list: List[Tuple[str, str, str]] = []
    for fp in repo_path.rglob("*"):
        if not fp.is_file():
            continue
        rel_path = fp.relative_to(repo_path)
        if any(part in SKIP_DIRS for part in rel_path.parts):
            continue
        if any(part.startswith(".") for part in rel_path.parts[:-1]):
            continue
        lang = EXTENSION_LANG.get(fp.suffix)
        if not lang:
            continue
        try:
            if fp.stat().st_size > max_file_bytes:
                continue
        except OSError:
            continue
        rel = str(rel_path)
        if gitignore_spec.match_file(rel.replace("\\", "/")):
            continue
        file_list.append((str(fp), rel, lang))
    return file_list


class IndexBuilder:
    """Orchestrates building and maintaining the code search index.

    Handles both full builds (--rebuild) and incremental builds (changed files only).
    Uses crash-safe ordering: meta is updated AFTER chunk commits so failed runs
    self-heal on the next invocation.

    Concurrent builds are serialised by an ``IndexLock`` (file-based lock).  On
    acquiring the lock the builder re-checks the stale state so a second caller
    that waited for the lock can skip redundant work.
    """

    def build(
        self,
        cfg: Any,  # WorkspaceConfig
        db_path: Path,
        rebuild: bool = False,
        repo_filter: Optional[str] = None,
        progress_fn: Optional[Callable[[str], None]] = None,
    ) -> Dict[str, Any]:
        """Build or incrementally update the code search index.

        Args:
            cfg: WorkspaceConfig (from build_config()).
            db_path: Absolute path to the SQLite database file.
            rebuild: If True, clears all tables and does a full rebuild.
            repo_filter: If set, only process the repo with this ID.

        Returns:
            Summary dict with stats about the build.

        Raises:
            ValueError: If the embedding model has changed and --rebuild is not set.
        """
        from corbell.core.embeddings.extractor import CodeChunkExtractor
        from corbell.core.embeddings.model import (
            GoogleEmbeddingModel, VoyageEmbeddingModel, EmbeddingModel,
        )
        from corbell.core.embeddings.sqlite_store import SQLiteEmbeddingStore
        from corbell.core.graph.sqlite_store import SQLiteGraphStore

        emb_store = SQLiteEmbeddingStore(db_path)
        graph_store = SQLiteGraphStore(db_path)
        tracker = IndexTracker(db_path)

        # Filter repos if requested
        repos = cfg.repos
        if repo_filter:
            repos = [r for r in repos if r.id == repo_filter]
            if not repos:
                raise ValueError(f"Repo '{repo_filter}' not found in workspace config")

        model_name = cfg.storage.resolved_model()

        # Model safety check (skip on full rebuild)
        if not rebuild:
            stored_model = tracker.get_stored_model()
            if stored_model and stored_model != model_name:
                raise ValueError(
                    f"Model changed from '{stored_model}' to '{model_name}'. "
                    f"Run 'corbell index build --rebuild' to re-index."
                )

        indexing = cfg.indexing
        extractor = CodeChunkExtractor(
            chunk_size=indexing.chunk_size,
            overlap=indexing.chunk_overlap,
        )
        model: EmbeddingModel
        if model_name.startswith("gemini-"):
            model = GoogleEmbeddingModel(model_name)
        elif model_name.startswith("voyage-"):
            model = VoyageEmbeddingModel(model_name)
        else:
            raise ValueError(
                f"Unsupported embedding model '{model_name}'. "
                f"Set CORBELL_EMBEDDING_MODEL to a cloud model:\n"
                f"  - voyage-code-3 or voyage-4-lite (requires VOYAGE_API_KEY)\n"
                f"  - gemini-embedding-001 (requires GOOGLE_API_KEY)"
            )

        if rebuild:
            return self._full_build(
                repos, emb_store, graph_store, tracker, extractor, model,
                indexing, model_name, db_path, repo_filter=repo_filter,
                progress_fn=progress_fn,
            )
        else:
            stale = tracker.get_stale_files(repos, cfg)
            if not stale.has_changes:
                return {"status": "clean", "chunks_added": 0, "repos_rebuilt": 0}

            return self._incremental_build(
                repos, stale, emb_store, graph_store, tracker,
                extractor, model, cfg, indexing, model_name, db_path,
                progress_fn=progress_fn,
            )

    def _full_build(
        self,
        repos: List,
        emb_store: Any,
        graph_store: Any,
        tracker: IndexTracker,
        extractor: Any,
        model: Any,
        indexing: Any,
        model_name: str,
        db_path: Path,
        repo_filter: Optional[str] = None,
        progress_fn: Optional[Callable[[str], None]] = None,
    ) -> Dict[str, Any]:
        """Run a full (re)build across all repos, serialised by IndexLock."""
        from corbell.core.indexing.lock import IndexLock

        lock = IndexLock(db_path.parent / "index.lock")
        with lock:
            return self._full_build_locked(
                repos, emb_store, graph_store, tracker, extractor, model,
                indexing, model_name, repo_filter=repo_filter, progress_fn=progress_fn,
            )

    def _full_build_locked(
        self,
        repos: List,
        emb_store: Any,
        graph_store: Any,
        tracker: IndexTracker,
        extractor: Any,
        model: Any,
        indexing: Any,
        model_name: str,
        repo_filter: Optional[str] = None,
        progress_fn: Optional[Callable[[str], None]] = None,
    ) -> Dict[str, Any]:
        """Inner full build — called while holding IndexLock."""
        # C2: Skip if another process already completed a build very recently.
        last_build = tracker.get_last_build_at()
        if last_build is not None and (time.time() - last_build) < 30:
            return {"status": "already_built", "chunks_added": 0, "repos_rebuilt": 0}

        # C1: Clear the index inside the lock to eliminate the race window.
        if repo_filter:
            emb_store.clear(service_id=repo_filter)
            graph_store.delete_service_data(repo_filter)
            tracker.remove_tracked([(r.id, r.id) for r in repos])
        else:
            emb_store.clear()
            graph_store.clear()
            tracker.clear_all()

        total_chunks = 0
        total_repos = 0
        services_data = []
        workers = _get_worker_count()
        repo_file_lists: Dict[str, List[Path]] = {}  # repo_path str -> list of Path objects

        for repo in repos:
            repo_id = repo.id
            repo_path = repo.resolved_path
            if not repo_path or not repo_path.exists():
                continue

            language = repo.language or "python"

            services_data.append({
                "id": repo_id,
                "resolved_path": repo_path,
                "repo": str(repo_path),
                "language": language,
                "tags": [],
            })

            gitignore_spec = load_gitignore(repo_path)

            # Collect file list for parallel dispatch
            file_list = _collect_repo_files(
                repo_path, repo_id, indexing.max_file_bytes, gitignore_spec
            )

            # Store Path objects for graph builders (reuse same walk)
            repo_file_lists[str(repo_path)] = [Path(abs_path) for abs_path, rel, lang in file_list]

            if not file_list:
                total_repos += 1
                continue

            if progress_fn:
                progress_fn(
                    f"Extracting {repo_id} ({len(file_list)} files, {workers} workers)..."
                )

            # Build worker args (all strings — picklable)
            worker_args = [
                (
                    abs_path_str,
                    rel_path,
                    lang,
                    repo_id,
                    str(repo_path),
                    indexing.chunk_size,
                    indexing.chunk_overlap,
                    indexing.max_file_bytes,
                )
                for abs_path_str, rel_path, lang in file_list
            ]

            # Parallel extraction
            chunks: List[Any] = []
            with ProcessPoolExecutor(max_workers=workers) as pool:
                futures = {pool.submit(_extract_file_worker, arg): arg for arg in worker_args}
                for future in as_completed(futures):
                    try:
                        chunks.extend(future.result())
                    except Exception as exc:
                        file_info = futures[future]
                        logger.warning("Failed to extract %s: %s", file_info[0], exc)

            if progress_fn:
                progress_fn(f"Indexing {repo_id} ({len(chunks)} chunks)...")

            if chunks:
                # Stream in super-batches: encode + commit one bounded slice at a time.
                # Committing per super-batch keeps SQLite WAL bounded (not corpus-sized).
                # CRASH-SAFE: mark_indexed is called AFTER all super-batches are committed,
                # so partial commits from a crash are cleaned up via delete_by_file on restart.
                with emb_store.connection() as conn:
                    for i in range(0, len(chunks), _SUPER_BATCH_SIZE):
                        super_batch = chunks[i : i + _SUPER_BATCH_SIZE]
                        _encode_chunks(model, super_batch)
                        emb_store.upsert_batch(super_batch, conn=conn)
                        conn.commit()
                total_chunks += len(chunks)

                # Mark each file as indexed AFTER chunks are committed
                file_mtimes = self._collect_file_mtimes(repo_path, chunks)
                for file_path, mtime in file_mtimes.items():
                    tracker.mark_indexed(file_path, repo_id, mtime)

            total_repos += 1

        # Build graph
        from corbell.core.graph.builder import ServiceGraphBuilder
        from corbell.core.graph.method_graph import MethodGraphBuilder, _EXT_LANG as _GRAPH_EXT_LANG
        if progress_fn:
            progress_fn("Building call graph...")
        sgb = ServiceGraphBuilder(graph_store)
        mgb = MethodGraphBuilder(graph_store)

        # Collect all indexed files across repos for ServiceGraphBuilder
        all_indexed_files: List[Path] = []
        for file_paths in repo_file_lists.values():
            all_indexed_files.extend(file_paths)

        sgb.build_from_workspace(
            services_data, clear_existing=False, method_level=False,
            file_list=all_indexed_files,
        )
        for svc in services_data:
            repo_path_str = str(svc["resolved_path"])
            all_files = repo_file_lists.get(repo_path_str, [])
            graph_file_list = [fp for fp in all_files if fp.suffix in _GRAPH_EXT_LANG]
            mgb.build_for_service(svc["id"], svc["resolved_path"], file_list=graph_file_list)

        # Store global metadata LAST (after all commits)
        tracker.set_meta("embedding_model", model_name)
        tracker.set_meta("last_build_at", str(time.time()))
        tracker.set_meta("chunk_size", str(indexing.chunk_size))
        tracker.set_meta("overlap", str(indexing.chunk_overlap))

        return {
            "status": "full_build",
            "chunks_added": total_chunks,
            "repos_rebuilt": total_repos,
        }

    def _incremental_build(
        self,
        repos: List,
        stale: Any,
        emb_store: Any,
        graph_store: Any,
        tracker: IndexTracker,
        extractor: Any,
        model: Any,
        cfg: Any,
        indexing: Any,
        model_name: str,
        db_path: Path,
        progress_fn: Optional[Callable[[str], None]] = None,
    ) -> Dict[str, Any]:
        """Re-embed changed files and rebuild graph for affected repos, serialised by IndexLock."""
        from corbell.core.indexing.lock import IndexLock

        lock = IndexLock(db_path.parent / "index.lock")
        with lock:
            # Re-check after acquiring lock — another process may have built already
            fresh_stale = tracker.get_stale_files(repos, cfg)
            if not fresh_stale.has_changes:
                return {"status": "clean", "chunks_added": 0, "repos_rebuilt": 0}

            return self._incremental_build_locked(
                repos, fresh_stale, emb_store, graph_store, tracker,
                extractor, model, indexing, model_name, progress_fn=progress_fn,
            )

    def _incremental_build_locked(
        self,
        repos: List,
        stale: Any,
        emb_store: Any,
        graph_store: Any,
        tracker: IndexTracker,
        extractor: Any,
        model: Any,
        indexing: Any,
        model_name: str,
        progress_fn: Optional[Callable[[str], None]] = None,
    ) -> Dict[str, Any]:
        """Inner incremental build — called while holding IndexLock."""
        from corbell.core.graph.builder import ServiceGraphBuilder
        from corbell.core.graph.method_graph import MethodGraphBuilder

        total_chunks = 0
        changed_repo_ids = stale.changed_repo_ids
        workers = _get_worker_count()

        # Build a lookup of repo_id → repo
        repo_map = {r.id: r for r in repos}

        # Handle deleted files
        for file_path, repo_id in stale.deleted:
            emb_store.delete_by_file(file_path, repo_id)
        tracker.remove_tracked(stale.deleted)

        # Re-embed modified + added files
        files_to_reindex: Dict[str, List] = {}
        for file_path, repo_id in stale.added + stale.modified:
            files_to_reindex.setdefault(repo_id, []).append(file_path)

        for repo_id, file_paths in files_to_reindex.items():
            repo = repo_map.get(repo_id)
            if not repo or not repo.resolved_path:
                continue
            repo_path = repo.resolved_path
            if progress_fn:
                progress_fn(
                    f"Re-indexing {len(file_paths)} files in {repo_id} ({workers} workers)..."
                )

            # Delete old chunks for all files in this repo before re-extracting
            for rel_path in file_paths:
                emb_store.delete_by_file(rel_path, repo_id)

            # Build worker args for all files in this repo
            from corbell.core.constants import EXTENSION_LANG

            worker_args = []
            for rel_path in file_paths:
                abs_path = repo_path / rel_path
                if not abs_path.exists():
                    continue
                lang = EXTENSION_LANG.get(abs_path.suffix, "python")
                worker_args.append((
                    str(abs_path),
                    rel_path,
                    lang,
                    repo_id,
                    str(repo_path),
                    indexing.chunk_size,
                    indexing.chunk_overlap,
                    indexing.max_file_bytes,
                ))

            if not worker_args:
                continue

            # Parallel extraction for this repo's changed files
            chunks: List[Any] = []
            with ProcessPoolExecutor(max_workers=workers) as pool:
                futures = {pool.submit(_extract_file_worker, arg): arg for arg in worker_args}
                for future in as_completed(futures):
                    try:
                        chunks.extend(future.result())
                    except Exception as exc:
                        file_info = futures[future]
                        logger.warning("Failed to extract %s: %s", file_info[0], exc)

            if chunks:
                # Stream in super-batches: encode + commit one bounded slice at a time.
                # CRASH-SAFE: mark_indexed is called AFTER all super-batches are committed,
                # so partial commits from a crash are cleaned up via delete_by_file on restart.
                with emb_store.connection() as conn:
                    for i in range(0, len(chunks), _SUPER_BATCH_SIZE):
                        super_batch = chunks[i : i + _SUPER_BATCH_SIZE]
                        _encode_chunks(model, super_batch)
                        emb_store.upsert_batch(super_batch, conn=conn)
                        conn.commit()
                total_chunks += len(chunks)

            # Mark all processed files as indexed AFTER commit
            for rel_path in file_paths:
                abs_path = repo_path / rel_path
                if not abs_path.exists():
                    continue
                try:
                    mtime = abs_path.stat().st_mtime
                except OSError:
                    mtime = time.time()
                tracker.mark_indexed(rel_path, repo_id, mtime)

        # Rebuild graph for affected repos
        sgb = ServiceGraphBuilder(graph_store)
        mgb = MethodGraphBuilder(graph_store)
        if progress_fn:
            progress_fn("Rebuilding call graph...")

        for repo_id in changed_repo_ids:
            repo = repo_map.get(repo_id)
            if not repo or not repo.resolved_path:
                continue
            repo_path = repo.resolved_path
            language = repo.language or "python"

            # Remove old graph data for this repo
            graph_store.delete_service_data(repo_id)

            # Rebuild graph for this repo
            svc_data = [{
                "id": repo_id,
                "resolved_path": repo_path,
                "repo": str(repo_path),
                "language": language,
                "tags": [],
            }]
            sgb.build_from_workspace(svc_data, clear_existing=False, method_level=False)
            mgb.build_for_service(repo_id, repo_path)

        # Update metadata LAST
        tracker.set_meta("embedding_model", model_name)
        tracker.set_meta("last_build_at", str(time.time()))
        tracker.set_meta("chunk_size", str(indexing.chunk_size))
        tracker.set_meta("overlap", str(indexing.chunk_overlap))

        return {
            "status": "incremental",
            "chunks_added": total_chunks,
            "repos_rebuilt": len(changed_repo_ids),
            "files_added": len(stale.added),
            "files_modified": len(stale.modified),
            "files_deleted": len(stale.deleted),
        }

    def _collect_file_mtimes(self, repo_path: Path, chunks: List) -> Dict[str, float]:
        """Collect mtimes for all files represented in the chunks list."""
        result: Dict[str, float] = {}
        for chunk in chunks:
            rel_path = chunk.file_path
            if rel_path not in result:
                abs_path = repo_path / rel_path
                try:
                    result[rel_path] = abs_path.stat().st_mtime
                except OSError:
                    result[rel_path] = time.time()
        return result

