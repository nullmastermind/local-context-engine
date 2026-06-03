"""Main query engine orchestrator for codebase retrieval."""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Optional


def codebase_retrieval(
    query: str,
    workspace_path: str | Path,
    top_k: int = 50,
    use_llm: bool = True,
    rerank: bool = True,
) -> str:
    """Execute the full code retrieval pipeline.

    Pipeline:
    1. Load workspace config and open stores.
    2. Auto-index check (empty → error, stale+old → blocking rebuild,
       stale+recent → background rebuild).
    3. LLM query enhancement (or use raw query).
    4. Embedding search via EmbeddingSearchCache.
    5. Graph call-chain expansion.
    6. Merge + dedup.
    7. LLM rerank (optional).
    8. Format results.

    Args:
        query: Natural language query string.
        workspace_path: Path to workspace.yaml or its containing directory.
        top_k: Maximum number of chunks to pass to reranker.
        use_llm: If False, skip LLM enhancement and reranking.
        rerank: If False, skip reranking even when LLM is configured.

    Returns:
        Formatted code snippet string ready for LLM context injection.
        Returns an error string (prefixed with "Error:") on failure.
    """
    from corbell.core.workspace import load_workspace
    from corbell.core.embeddings.sqlite_store import SQLiteEmbeddingStore
    from corbell.core.embeddings.search_cache import EmbeddingSearchCache
    from corbell.core.embeddings.model import SentenceTransformerModel, GoogleEmbeddingModel, EmbeddingModel
    from corbell.core.graph.sqlite_store import SQLiteGraphStore
    from corbell.core.indexing.builder import IndexBuilder
    from corbell.core.indexing.tracker import IndexTracker
    from corbell.core.query.diagnostics import QueryDiagnostics
    from corbell.core.query.enhancer import enhance_query
    from corbell.core.query.graph_expander import ScoredChunk, expand_via_graph
    from corbell.core.query.merger import merge_and_dedup
    from corbell.core.query.reranker import rerank_chunks
    from corbell.core.query.formatter import format_results

    workspace_path = Path(workspace_path)
    config_dir = workspace_path if workspace_path.is_dir() else workspace_path.parent

    try:
        cfg = load_workspace(workspace_path)
    except FileNotFoundError:
        return f"Error: workspace.yaml not found at {workspace_path}. Run 'corbell init' first."

    db_path = cfg.db_path(config_dir)
    emb_store = SQLiteEmbeddingStore(db_path)
    graph_store = SQLiteGraphStore(db_path)
    tracker = IndexTracker(db_path)

    # --- Auto-index check ---
    chunk_count = emb_store.count()
    if chunk_count == 0:
        return "No index found. Run 'corbell index build' first."

    stale_result = tracker.get_stale_files(cfg.repos, cfg)
    if stale_result.has_changes:
        last_build = tracker.get_last_build_at()
        age_seconds = time.time() - (last_build or 0)
        one_day = 86400

        if age_seconds > one_day:
            # Stale + old → blocking incremental rebuild
            builder = IndexBuilder()
            builder.build(cfg, config_dir, rebuild=False)
        else:
            # Stale + recent → background subprocess
            _spawn_background_worker(workspace_path, config_dir, db_path)

    # --- LLM client setup ---
    llm_client: Optional[Any] = None
    if use_llm:
        from corbell.core.llm_client import LLMClient
        llm_cfg = cfg.llm
        llm_client = LLMClient(
            provider=llm_cfg.provider,
            model=llm_cfg.model,
            api_key=llm_cfg.resolved_api_key(),
            aws_region=llm_cfg.aws_region,
            azure_endpoint=llm_cfg.azure_endpoint,
            azure_deployment=llm_cfg.azure_deployment,
            azure_api_version=llm_cfg.azure_api_version,
            gcp_project=llm_cfg.gcp_project,
            gcp_region=llm_cfg.gcp_region,
        )

    # --- Query enhancement ---
    search_queries, _keywords = enhance_query(query, llm_client if use_llm else None)

    # --- Embedding model ---
    model_name = cfg.storage.model
    emb_model: EmbeddingModel
    if model_name.startswith("gemini-"):
        emb_model = GoogleEmbeddingModel(model_name)
    else:
        emb_model = SentenceTransformerModel(model_name)

    # --- Load search cache ---
    cache = EmbeddingSearchCache()
    cache.load(emb_store)

    if not cache.is_loaded:
        return "No index found. Run 'corbell index build' first."

    # --- Embedding search ---
    import numpy as np

    all_embedding_results: dict[str, ScoredChunk] = {}
    query_config = cfg.query

    for sq in search_queries:
        try:
            if isinstance(emb_model, GoogleEmbeddingModel):
                q_vecs = emb_model.encode([sq], task_type="RETRIEVAL_QUERY")
            else:
                q_vecs = emb_model.encode([sq])
        except Exception as exc:
            return f"Error: Failed to load embedding model '{model_name}'. Ensure 'sentence-transformers' is installed. ({exc})"

        q_vec = np.array(q_vecs[0], dtype=np.float32)
        hits = cache.search(q_vec, top_k=top_k)

        if not hits:
            continue

        # Fetch full records for top hits
        hit_ids = [h[0] for h in hits]
        hit_scores = {h[0]: h[1] for h in hits}

        try:
            records = emb_store.get_chunks_by_ids(hit_ids)
        except Exception:
            continue

        # Build repo_path map for resolving absolute paths
        repo_path_map = {
            r.id: str(r.resolved_path) for r in cfg.repos if r.resolved_path
        }

        for record in records:
            score = hit_scores.get(record.id, 0.0)
            # Resolve absolute file path
            abs_path = record.file_path
            repo_root = repo_path_map.get(record.service_id, "")
            if repo_root and not Path(abs_path).is_absolute():
                abs_path = str((Path(repo_root) / abs_path).resolve())

            chunk = ScoredChunk(
                chunk_id=record.id,
                score=score,
                file_path=abs_path,
                start_line=record.start_line,
                end_line=record.end_line,
                content=record.content,
                repo_id=record.service_id,
                symbol=record.symbol,
                chunk_type=record.chunk_type,
                language=record.language,
            )

            # Keep max score for deduplication across queries
            existing = all_embedding_results.get(record.id)
            if existing is None or score > existing.score:
                all_embedding_results[record.id] = chunk

    if not all_embedding_results:
        return "No relevant code found for the given query."

    base_chunks = list(all_embedding_results.values())

    # --- Graph expansion ---
    diagnostics = QueryDiagnostics()
    bonus_chunks = expand_via_graph(
        embedding_results=base_chunks,
        graph_store=graph_store,
        repos=cfg.repos,
        max_depth=query_config.expand_call_depth,
        max_chunks=query_config.expand_max_chunks,
        diagnostics=diagnostics,
    )

    all_chunks = base_chunks + bonus_chunks

    # --- Merge + dedup ---
    merged = merge_and_dedup(all_chunks)

    # --- Apply top_k cap ---
    merged = merged[:top_k]

    # --- LLM rerank ---
    do_rerank = use_llm and rerank and query_config.rerank
    if do_rerank:
        reranked_ids = rerank_chunks(query, merged, llm_client)
        # Reorder merged by reranked_ids
        id_to_chunk = {c.chunk_id: c for c in merged}
        reranked = [id_to_chunk[cid] for cid in reranked_ids if cid in id_to_chunk]
        # Add any chunks not returned by reranker at the end
        reranked_set = set(reranked_ids)
        leftover = [c for c in merged if c.chunk_id not in reranked_set]
        merged = reranked + leftover

    # --- Format output ---
    repo_paths = {r.id: str(r.resolved_path) for r in cfg.repos if r.resolved_path}
    output = format_results(merged, repo_paths)

    # Prepend diagnostics warning if thresholds exceeded
    warning = diagnostics.summary()
    if warning:
        output = f"[warnings: {warning}]\n\n{output}"

    return output


def _spawn_background_worker(
    workspace_path: Path,
    config_dir: Path,
    db_path: Path,
) -> None:
    """Spawn a background subprocess for incremental index rebuild.

    Uses PID file deduplication to prevent double-spawning.
    """
    pid_file = config_dir / ".corbell" / "index.pid"
    pid_file.parent.mkdir(parents=True, exist_ok=True)

    # Check if a worker is already running
    if pid_file.exists():
        try:
            pid = int(pid_file.read_text().strip())
            # Check if process is alive
            if _is_process_alive(pid):
                return  # worker already running
        except (ValueError, OSError):
            pass
        # Stale PID file — remove it
        try:
            pid_file.unlink()
        except OSError:
            pass

    # Spawn background worker
    try:
        proc = subprocess.Popen(
            [sys.executable, "-m", "corbell.core.indexing._worker", str(workspace_path)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        pid_file.write_text(str(proc.pid))
    except Exception:
        pass  # Background worker failure is non-fatal


def _is_process_alive(pid: int) -> bool:
    """Check if a process with the given PID is still running."""
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except (OSError, ProcessLookupError):
        return False
