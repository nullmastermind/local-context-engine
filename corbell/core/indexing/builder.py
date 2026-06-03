"""Index builder: orchestrates full and incremental builds of the code index."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from corbell.core.gitignore import load_gitignore
from corbell.core.indexing.tracker import IndexTracker


class IndexBuilder:
    """Orchestrates building and maintaining the code search index.

    Handles both full builds (--rebuild) and incremental builds (changed files only).
    Uses crash-safe ordering: meta is updated AFTER chunk commits so failed runs
    self-heal on the next invocation.
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
        from corbell.core.embeddings.model import SentenceTransformerModel, GoogleEmbeddingModel, EmbeddingModel
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

        # Full rebuild: clear everything
        if rebuild:
            if repo_filter:
                # Only clear the specific repo's data
                emb_store.clear(service_id=repo_filter)
                graph_store.delete_service_data(repo_filter)
                tracker.remove_tracked(
                    [(r.id, r.id) for r in repos]  # placeholder; real cleanup below
                )
            else:
                emb_store.clear()
                graph_store.clear()
                tracker.clear_all()

        indexing = cfg.indexing
        extractor = CodeChunkExtractor(
            chunk_size=indexing.chunk_size,
            overlap=indexing.chunk_overlap,
        )
        model: EmbeddingModel
        if model_name.startswith("gemini-"):
            model = GoogleEmbeddingModel(model_name)
        else:
            model = SentenceTransformerModel(model_name)

        if rebuild:
            return self._full_build(
                repos, emb_store, graph_store, tracker, extractor, model,
                indexing, model_name, progress_fn=progress_fn,
            )
        else:
            stale = tracker.get_stale_files(repos, cfg)
            if not stale.has_changes:
                return {"status": "clean", "chunks_added": 0, "repos_rebuilt": 0}

            return self._incremental_build(
                repos, stale, emb_store, graph_store, tracker,
                extractor, model, cfg, indexing, model_name,
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
        progress_fn: Optional[Callable[[str], None]] = None,
    ) -> Dict[str, Any]:
        """Run a full (re)build across all repos."""
        total_chunks = 0
        total_repos = 0
        services_data = []

        for repo in repos:
            repo_id = repo.id
            repo_path = repo.resolved_path
            if not repo_path or not repo_path.exists():
                continue

            language = repo.language or "python"

            # Build graph for this repo
            services_data.append({
                "id": repo_id,
                "resolved_path": repo_path,
                "repo": str(repo_path),
                "language": language,
                "tags": [],
            })

            # Load gitignore once per repo, share with extractor
            gitignore_spec = load_gitignore(repo_path)

            # Extract and embed chunks
            chunks = extractor.extract_from_repo(
                repo_path, repo_id,
                max_file_bytes=indexing.max_file_bytes,
                gitignore_spec=gitignore_spec,
            )
            if progress_fn:
                progress_fn(f"Indexing {repo_id} ({len(chunks)} chunks)...")
            if chunks:
                from corbell.core.embeddings.model import GoogleEmbeddingModel
                if isinstance(model, GoogleEmbeddingModel) and model.uses_prefix_format:
                    texts = [
                        model.prepare_document(
                            c.content,
                            title=f"{c.file_path}:{c.symbol}"
                            if c.symbol
                            else f"{c.file_path}:L{c.start_line}-{c.end_line}",
                        )
                        for c in chunks
                    ]
                else:
                    texts = [c.content for c in chunks]
                vectors = model.encode(texts)
                for chunk, vec in zip(chunks, vectors):
                    chunk.embedding = vec

                # CRASH-SAFE: commit chunks first, then update meta
                emb_store.upsert_batch(chunks)
                total_chunks += len(chunks)

                # Mark each file as indexed AFTER chunks are committed
                file_mtimes = self._collect_file_mtimes(repo_path, chunks)
                for file_path, mtime in file_mtimes.items():
                    tracker.mark_indexed(file_path, repo_id, mtime)

            total_repos += 1

        # Build graph
        from corbell.core.graph.builder import ServiceGraphBuilder
        from corbell.core.graph.method_graph import MethodGraphBuilder
        if progress_fn:
            progress_fn("Building call graph...")
        sgb = ServiceGraphBuilder(graph_store)
        mgb = MethodGraphBuilder(graph_store)
        sgb.build_from_workspace(services_data, clear_existing=False, method_level=False)
        for svc in services_data:
            mgb.build_for_service(svc["id"], svc["resolved_path"])

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
        progress_fn: Optional[Callable[[str], None]] = None,
    ) -> Dict[str, Any]:
        """Re-embed changed files and rebuild graph for affected repos."""
        from corbell.core.graph.builder import ServiceGraphBuilder
        from corbell.core.graph.method_graph import MethodGraphBuilder

        total_chunks = 0
        changed_repo_ids = stale.changed_repo_ids

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
                progress_fn(f"Re-indexing {len(file_paths)} files in {repo_id}...")

            for rel_path in file_paths:
                abs_path = repo_path / rel_path
                if not abs_path.exists():
                    continue

                # Delete old chunks for this file
                emb_store.delete_by_file(rel_path, repo_id)

                # Extract chunks from file
                from corbell.core.constants import EXTENSION_LANG
                lang = EXTENSION_LANG.get(abs_path.suffix, "python")
                chunks = extractor._extract_file(abs_path, rel_path, lang, repo_id, str(repo_path))

                if chunks:
                    from corbell.core.embeddings.model import GoogleEmbeddingModel
                    if isinstance(model, GoogleEmbeddingModel) and model.uses_prefix_format:
                        texts = [
                            model.prepare_document(
                                c.content,
                                title=f"{c.file_path}:{c.symbol}"
                                if c.symbol
                                else f"{c.file_path}:L{c.start_line}-{c.end_line}",
                            )
                            for c in chunks
                        ]
                    else:
                        texts = [c.content for c in chunks]
                    vectors = model.encode(texts)
                    for chunk, vec in zip(chunks, vectors):
                        chunk.embedding = vec

                    # CRASH-SAFE: commit chunks first
                    emb_store.upsert_batch(chunks)
                    total_chunks += len(chunks)

                # Mark file as indexed AFTER commit
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
