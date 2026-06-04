"""Main query engine orchestrator for codebase retrieval."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class DebugResult:
    """Rich result from codebase_retrieval_debug(), including pipeline internals."""

    final_output: str
    pre_rerank_chunks: List[Any]
    rerank_detail: Optional[Any]
    diagnostics: Any
    error: Optional[str]


def _execute_pipeline(
    query: str,
    workspace_path: Path,
    top_k: int = 50,
    use_llm: bool = True,
    rerank: bool = True,
    diagnostics: Optional[Any] = None,
) -> Tuple[str, Any]:
    """Execute the full code retrieval pipeline and return (output, diagnostics).

    Extracted from codebase_retrieval() to allow debug callers to pass in a
    pre-configured QueryDiagnostics (e.g. with collect_debug=True).

    Args:
        query: Natural language query string.
        workspace_path: Resolved absolute path to the workspace root.
        top_k: Maximum number of chunks to pass to reranker.
        use_llm: If False, skip reranking.
        rerank: If False, skip reranking even when LLM is configured.
        diagnostics: Optional QueryDiagnostics instance; created if None.

    Returns:
        Tuple of (formatted_output_string, diagnostics).
    """
    from corbell.core.workspace import build_config, db_path_for_workspace
    from corbell.core.embeddings.sqlite_store import SQLiteEmbeddingStore
    from corbell.core.embeddings.search_cache import EmbeddingSearchCache
    from corbell.core.embeddings.model import GoogleEmbeddingModel, VoyageEmbeddingModel, EmbeddingModel
    from corbell.core.graph.sqlite_store import SQLiteGraphStore
    from corbell.core.indexing.builder import IndexBuilder
    from corbell.core.indexing.tracker import IndexTracker
    from corbell.core.query.diagnostics import QueryDiagnostics
    from corbell.core.query.graph_expander import ScoredChunk, expand_via_graph
    from corbell.core.query.merger import merge_and_dedup
    from corbell.core.query.reranker import rerank_chunks
    from corbell.core.query.formatter import format_results

    if diagnostics is None:
        diagnostics = QueryDiagnostics()

    if not workspace_path.exists():
        return (
            f"Error: Workspace directory not found: {workspace_path}. "
            "Run 'corbell index build' first.",
            diagnostics,
        )

    cfg = build_config(workspace_path)
    db_path = db_path_for_workspace(workspace_path, model=cfg.storage.resolved_model())
    emb_store = SQLiteEmbeddingStore(db_path)
    graph_store = SQLiteGraphStore(db_path)
    tracker = IndexTracker(db_path)

    # --- Auto-index check ---
    chunk_count = emb_store.count()
    if chunk_count == 0:
        logger.info("Index is empty — running full build now (this may take a while)...")
        builder = IndexBuilder()
        builder.build(cfg, db_path, rebuild=True, progress_fn=lambda msg: logger.info(msg))

    # Short-circuit: skip stale check if a build finished within the last 30 seconds
    last_build = tracker.get_last_build_at()
    if last_build is None or (time.time() - last_build) >= 30:
        stale_result = tracker.get_stale_files(cfg.repos, cfg)
        if stale_result.has_changes:
            # Always do a blocking incremental rebuild when stale
            builder = IndexBuilder()
            builder.build(cfg, db_path, rebuild=False, progress_fn=lambda msg: logger.info(msg))

    # --- LLM client setup ---
    llm_client: Optional[Any] = None
    if use_llm:
        from corbell.core.llm_client import LLMClient
        llm_cfg = cfg.llm
        llm_client = LLMClient(
            provider=llm_cfg.provider,
            model=llm_cfg.resolved_model(),
            api_key=llm_cfg.resolved_api_key(),
            aws_region=llm_cfg.aws_region,
            azure_endpoint=llm_cfg.azure_endpoint,
            azure_deployment=llm_cfg.azure_deployment,
            azure_api_version=llm_cfg.azure_api_version,
            gcp_project=llm_cfg.gcp_project,
            gcp_region=llm_cfg.gcp_region,
        )

    # --- Search queries ---
    search_queries = [query]

    # --- Embedding model ---
    model_name = cfg.storage.resolved_model()
    emb_model: EmbeddingModel
    if model_name.startswith("gemini-"):
        emb_model = GoogleEmbeddingModel(model_name)
    elif model_name.startswith("voyage-"):
        emb_model = VoyageEmbeddingModel(model_name)
    else:
        raise ValueError(
            f"Unsupported embedding model '{model_name}'. "
            f"Set CORBELL_EMBEDDING_MODEL to a cloud model:\n"
            f"  - voyage-code-3 or voyage-4-lite (requires VOYAGE_API_KEY)\n"
            f"  - gemini-embedding-001 (requires GOOGLE_API_KEY)"
        )

    # --- Load search cache ---
    cache = EmbeddingSearchCache()
    cache.load(emb_store)

    if not cache.is_loaded:
        return "No index found. Run 'corbell index build' first.", diagnostics

    # --- Embedding search ---
    import numpy as np

    all_embedding_results: dict[str, ScoredChunk] = {}
    query_config = cfg.query

    t0 = time.time()
    try:
        for sq in search_queries:
            try:
                if isinstance(emb_model, GoogleEmbeddingModel):
                    formatted_query = (
                        emb_model.prepare_query(sq) if emb_model.uses_prefix_format else sq
                    )
                    q_vecs = emb_model.encode([formatted_query], task_type="RETRIEVAL_QUERY")
                elif isinstance(emb_model, VoyageEmbeddingModel):
                    q_vecs = emb_model.encode([sq], input_type="query")
                else:
                    q_vecs = emb_model.encode([sq])
            except Exception as exc:
                return (
                    f"Error: Failed to encode query with embedding model '{model_name}': {exc}",
                    diagnostics,
                )

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
    finally:
        diagnostics.record_time("embedding_search", time.time() - t0)

    if not all_embedding_results:
        return "No relevant code found for the given query.", diagnostics

    base_chunks = list(all_embedding_results.values())

    # --- Graph expansion ---
    t0 = time.time()
    try:
        bonus_chunks = expand_via_graph(
            embedding_results=base_chunks,
            graph_store=graph_store,
            repos=cfg.repos,
            max_depth=query_config.expand_call_depth,
            max_chunks=query_config.expand_max_chunks,
            diagnostics=diagnostics,
        )
    finally:
        diagnostics.record_time("graph_expansion", time.time() - t0)

    all_chunks = base_chunks + bonus_chunks

    # Track which chunks came from graph expansion
    if diagnostics.collect_debug:
        diagnostics.graph_chunk_ids = {c.chunk_id for c in bonus_chunks}

    # --- Merge + dedup ---
    t0 = time.time()
    try:
        merged = merge_and_dedup(all_chunks)
        # Apply top_k cap with reserved slots for pure-graph chunks.
        # Pure-graph chunks have low scores (cascaded 0.6x/0.5x) and would
        # be dropped by a naive score-based cap. Reserve up to 30% of slots.
        if bonus_chunks and len(merged) > top_k:
            graph_set = {c.chunk_id for c in bonus_chunks}
            graph_only = [c for c in merged if all(
                p in graph_set for p in c.chunk_id.split('+')
            )]
            the_rest = [c for c in merged if not all(
                p in graph_set for p in c.chunk_id.split('+')
            )]
            graph_budget = min(len(graph_only), top_k * 3 // 10)
            rest_budget = top_k - graph_budget
            merged = the_rest[:rest_budget] + graph_only[:graph_budget]
            merged.sort(key=lambda c: c.score, reverse=True)
        else:
            merged = merged[:top_k]
    finally:
        diagnostics.record_time("merge_dedup", time.time() - t0)

    # Capture pre-rerank state for debug mode
    if diagnostics.collect_debug:
        diagnostics.pre_rerank_chunks = list(merged)

    # --- LLM rerank ---
    t0 = time.time()
    try:
        do_rerank = use_llm and rerank and query_config.rerank
        if do_rerank:
            # Annotate chunks with graph metadata before sending to the reranker
            graph_meta = _annotate_with_graph_meta(merged, graph_store, cfg.repos)

            rerank_result = rerank_chunks(query, merged, llm_client, graph_meta=graph_meta)
            reranked_ids = rerank_result.chunk_ids

            if diagnostics.collect_debug:
                diagnostics.rerank_detail = rerank_result

            logger.info(
                "Rerank complete: %.3fs, %d/%d chunks kept, order: %s",
                rerank_result.elapsed_seconds,
                len(reranked_ids),
                len(merged),
                reranked_ids,
            )
            # Reorder merged, keeping only chunks selected by the reranker
            id_to_chunk = {c.chunk_id: c for c in merged}
            merged = [id_to_chunk[cid] for cid in reranked_ids if cid in id_to_chunk]
    finally:
        diagnostics.record_time("rerank", time.time() - t0)

    # --- Format output ---
    t0 = time.time()
    try:
        repo_paths = {r.id: str(r.resolved_path) for r in cfg.repos if r.resolved_path}
        output = format_results(merged, repo_paths)

        # Prepend diagnostics warning if thresholds exceeded
        warning = diagnostics.summary()
        if warning:
            output = f"[warnings: {warning}]\n\n{output}"
    finally:
        diagnostics.record_time("format", time.time() - t0)

    return output, diagnostics


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
    2. Auto-index check (empty → full build, stale → blocking incremental rebuild).
       Skipped entirely when last build completed within the past 30 seconds.
    3. Embedding search via EmbeddingSearchCache (raw query used directly).
    4. Graph call-chain expansion.
    5. Merge + dedup.
    6. LLM rerank (optional).
    7. Format results.

    Args:
        query: Natural language query string.
        workspace_path: Path to the workspace (repository) root directory.
        top_k: Maximum number of chunks to pass to reranker.
        use_llm: If False, skip reranking.
        rerank: If False, skip reranking even when LLM is configured.

    Returns:
        Formatted code snippet string ready for LLM context injection.
        Returns an error string (prefixed with "Error:") on failure.
    """
    output, _ = _execute_pipeline(
        query,
        Path(workspace_path).resolve(),
        top_k=top_k,
        use_llm=use_llm,
        rerank=rerank,
    )
    return output


def codebase_retrieval_debug(
    query: str,
    workspace_path: str | Path,
    top_k: int = 50,
) -> DebugResult:
    """Execute the retrieval pipeline in debug mode, returning full internals.

    Same pipeline as codebase_retrieval(), but captures pre-rerank chunks,
    rerank prompts/response, per-phase timing, and any error.

    Args:
        query: Natural language query string.
        workspace_path: Path to the workspace root directory.
        top_k: Maximum number of chunks to pass to reranker.

    Returns:
        DebugResult with final_output, pre_rerank_chunks, rerank_detail,
        diagnostics (including timing), and error (if any).
    """
    from corbell.core.query.diagnostics import QueryDiagnostics

    diag = QueryDiagnostics(collect_debug=True)
    try:
        output, diag = _execute_pipeline(
            query,
            Path(workspace_path).resolve(),
            top_k=top_k,
            diagnostics=diag,
        )
        error = None
        if output.startswith("Error:") or output.startswith("No "):
            error = output
            output = ""
    except Exception as exc:
        output = ""
        error = f"{type(exc).__name__}: {exc}"

    return DebugResult(
        final_output=output,
        pre_rerank_chunks=diag.pre_rerank_chunks or [],
        rerank_detail=diag.rerank_detail,
        diagnostics=diag,
        error=error,
    )


def _annotate_with_graph_meta(
    chunks: List[Any],
    graph_store: Any,
    repos: List[Any],
) -> Dict[str, Dict]:
    """Build a graph metadata dict keyed by chunk_id for each chunk.

    For each chunk, finds overlapping MethodNodes (by file_path + line range)
    and collects:
      - callers: number of methods that call into this chunk's method
      - callees: number of method_call edges outgoing from this chunk's method
      - flow: name of the first FlowNode that includes this method (or None)

    Args:
        chunks: List of ScoredChunk objects.
        graph_store: SQLiteGraphStore instance.
        repos: List of RepoConfig objects for path resolution.

    Returns:
        Dict mapping chunk_id -> {"callers": int, "callees": int, "flow": str | None}.
        Chunks with no matching MethodNode are omitted.
    """
    from corbell.core.query.graph_expander import _find_matching_methods

    # Build repo_id → absolute path mapping (same as graph_expander)
    repo_path_map: Dict[str, Path] = {}
    for repo in repos:
        if repo.resolved_path:
            repo_path_map[repo.id] = repo.resolved_path

    try:
        all_services = graph_store.get_all_services()
        service_ids = [s.id for s in all_services]
    except Exception:
        return {}

    graph_meta: Dict[str, Dict] = {}

    for chunk in chunks:
        try:
            matching_methods = _find_matching_methods(
                chunk, graph_store, repo_path_map, service_ids
            )
        except Exception:
            continue

        if not matching_methods:
            continue

        # Aggregate across all overlapping methods (e.g. nested lambdas)
        total_callers = 0
        total_callees = 0
        flow_name: Optional[str] = None

        for method in matching_methods:
            try:
                callers = graph_store.get_callers_of_method(method.id)
                total_callers += len(callers)
            except Exception:
                pass

            try:
                outgoing = graph_store.get_dependencies(method.id)
                total_callees += sum(1 for e in outgoing if e.kind == "method_call")
            except Exception:
                pass

            if flow_name is None:
                try:
                    flows = graph_store.get_flows_for_method(method.id)
                    if flows:
                        flow_name = flows[0].get("flow_name") or None
                except Exception:
                    pass

        graph_meta[chunk.chunk_id] = {
            "callers": total_callers,
            "callees": total_callees,
            "flow": flow_name,
        }

    return graph_meta
