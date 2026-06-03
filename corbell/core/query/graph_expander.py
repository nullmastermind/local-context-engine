"""Graph-based call-chain expansion for query results."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, TYPE_CHECKING

if TYPE_CHECKING:
    from corbell.core.query.diagnostics import QueryDiagnostics


@dataclass
class ScoredChunk:
    """An embedding chunk with a relevance score."""

    chunk_id: str
    score: float
    file_path: str      # absolute path
    start_line: int
    end_line: int
    content: str
    repo_id: str
    symbol: Optional[str] = None
    chunk_type: str = "block"
    language: str = "python"


def expand_via_graph(
    embedding_results: List[ScoredChunk],
    graph_store: Any,
    repos: List,
    max_depth: int = 2,
    max_chunks: int = 30,
    diagnostics: Optional["QueryDiagnostics"] = None,
) -> List[ScoredChunk]:
    """Expand embedding results by following call-chain edges in the graph.

    For each embedding result that overlaps with a MethodNode (by file path +
    line range), BFS is performed over callers (score * 0.6) and callees
    (score * 0.5). A global cap of max_chunks expanded chunks is enforced.
    Score floor of 0.15 stops expansion of low-confidence chains.

    Args:
        embedding_results: Initial scored chunks from embedding search.
        graph_store: SQLiteGraphStore instance.
        repos: List of RepoConfig objects (for resolving relative paths).
        max_depth: BFS depth limit for expansion.
        max_chunks: Maximum total expanded (bonus) chunks to add.
        diagnostics: Optional diagnostics object for tracking warnings.

    Returns:
        List of bonus ScoredChunk objects (excluding the original results).
    """
    if not embedding_results:
        return []

    # Build repo_id → absolute path mapping
    repo_path_map: Dict[str, Path] = {}
    for repo in repos:
        if repo.resolved_path:
            repo_path_map[repo.id] = repo.resolved_path

    bonus_chunks: List[ScoredChunk] = []
    visited: Set[str] = set()  # visited method IDs

    # Get all method nodes from the graph (cached as dict keyed by file+lines)
    try:
        all_services = graph_store.get_all_services()
        service_ids = [s.id for s in all_services]
    except Exception:
        return []

    for base_chunk in embedding_results:
        if len(bonus_chunks) >= max_chunks:
            break

        # Find MethodNodes that overlap with this chunk's file+line range
        matching_methods = _find_matching_methods(
            base_chunk, graph_store, repo_path_map, service_ids
        )

        for method_node in matching_methods:
            if len(bonus_chunks) >= max_chunks:
                break
            if method_node.id in visited:
                continue

            visited.add(method_node.id)
            _bfs_expand(
                method_node=method_node,
                parent_score=base_chunk.score,
                depth=0,
                max_depth=max_depth,
                graph_store=graph_store,
                repo_path_map=repo_path_map,
                bonus_chunks=bonus_chunks,
                max_chunks=max_chunks,
                visited=visited,
                diagnostics=diagnostics,
            )

    return bonus_chunks


def _find_matching_methods(
    chunk: ScoredChunk,
    graph_store: Any,
    repo_path_map: Dict[str, Path],
    service_ids: List[str],
) -> List[Any]:
    """Find MethodNodes whose file_path and line range overlap with the given chunk."""
    results = []

    try:
        for service_id in service_ids:
            methods = graph_store.get_methods_for_service(service_id)
            for method in methods:
                # Normalize method file_path to absolute
                method_abs = Path(method.file_path)
                chunk_abs = Path(chunk.file_path)

                # Compare absolute paths
                if not method_abs.is_absolute():
                    repo_path = repo_path_map.get(service_id)
                    if repo_path:
                        method_abs = (repo_path / method.file_path).resolve()

                if method_abs != chunk_abs:
                    continue

                # Check line overlap: method and chunk lines overlap
                if method.line_end < chunk.start_line:
                    continue
                if method.line_start > chunk.end_line:
                    continue

                results.append(method)
    except Exception:
        pass

    return results


def _bfs_expand(
    method_node: Any,
    parent_score: float,
    depth: int,
    max_depth: int,
    graph_store: Any,
    repo_path_map: Dict[str, Path],
    bonus_chunks: List[ScoredChunk],
    max_chunks: int,
    visited: Set[str],
    diagnostics: Optional["QueryDiagnostics"],
) -> None:
    """BFS expansion from a method node, adding callers and callees as bonus chunks."""
    if depth >= max_depth:
        return
    if len(bonus_chunks) >= max_chunks:
        return

    # Expand callers (score * 0.6)
    try:
        callers = graph_store.get_callers_of_method(method_node.id)
        for caller in callers:
            if len(bonus_chunks) >= max_chunks:
                return
            caller_score = parent_score * 0.6
            if caller_score < 0.15:
                continue
            if caller.id in visited:
                continue
            visited.add(caller.id)

            bonus = _method_to_scored_chunk(
                caller, caller_score, repo_path_map, diagnostics
            )
            if bonus is not None:
                bonus_chunks.append(bonus)
                _bfs_expand(
                    method_node=caller,
                    parent_score=caller_score,
                    depth=depth + 1,
                    max_depth=max_depth,
                    graph_store=graph_store,
                    repo_path_map=repo_path_map,
                    bonus_chunks=bonus_chunks,
                    max_chunks=max_chunks,
                    visited=visited,
                    diagnostics=diagnostics,
                )
    except Exception:
        if diagnostics:
            diagnostics.graph_expansion_failures += 1

    # Expand callees (score * 0.5)
    try:
        outgoing = graph_store.get_dependencies(method_node.id)
        for edge in outgoing:
            if edge.kind != "method_call":
                continue
            if len(bonus_chunks) >= max_chunks:
                return
            callee_score = parent_score * 0.5
            if callee_score < 0.15:
                continue

            callee_id = edge.target_id
            if callee_id in visited:
                continue
            visited.add(callee_id)

            callee_node = graph_store.get_method(callee_id)
            if callee_node is None:
                if diagnostics:
                    diagnostics.skipped_methods += 1
                continue

            bonus = _method_to_scored_chunk(
                callee_node, callee_score, repo_path_map, diagnostics
            )
            if bonus is not None:
                bonus_chunks.append(bonus)
                _bfs_expand(
                    method_node=callee_node,
                    parent_score=callee_score,
                    depth=depth + 1,
                    max_depth=max_depth,
                    graph_store=graph_store,
                    repo_path_map=repo_path_map,
                    bonus_chunks=bonus_chunks,
                    max_chunks=max_chunks,
                    visited=visited,
                    diagnostics=diagnostics,
                )
    except Exception:
        if diagnostics:
            diagnostics.graph_expansion_failures += 1


def _method_to_scored_chunk(
    method_node: Any,
    score: float,
    repo_path_map: Dict[str, Path],
    diagnostics: Optional["QueryDiagnostics"],
) -> Optional[ScoredChunk]:
    """Convert a MethodNode to a ScoredChunk by reading its source lines.

    Returns None if the file doesn't exist (increments diagnostics counter).
    """
    file_path = Path(method_node.file_path)
    if not file_path.is_absolute():
        repo_path = repo_path_map.get(method_node.service_id)
        if repo_path:
            file_path = (repo_path / method_node.file_path).resolve()

    if not file_path.exists():
        if diagnostics:
            diagnostics.skipped_files += 1
        return None

    try:
        lines = file_path.read_text(encoding="utf-8", errors="ignore").splitlines()
        start = max(0, method_node.line_start - 1)
        end = min(len(lines), method_node.line_end)
        content = "\n".join(lines[start:end])
    except Exception:
        if diagnostics:
            diagnostics.skipped_files += 1
        return None

    return ScoredChunk(
        chunk_id=method_node.id,
        score=score,
        file_path=str(file_path),
        start_line=method_node.line_start,
        end_line=method_node.line_end,
        content=content,
        repo_id=method_node.service_id,
        symbol=method_node.method_name,
        chunk_type="method",
        language="python",  # best effort; MethodNode doesn't store language
    )
