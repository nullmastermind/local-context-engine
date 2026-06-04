"""MCP Server for Corbell code retrieval engine.

Exposes a single tool `context_engine_codebase_retrieval` via FastMCP,
supporting both stdio and SSE transports.
"""

from __future__ import annotations

import os
from typing import Optional

from mcp.server.fastmcp import FastMCP


# Create the FastMCP server
mcp = FastMCP("corbell", dependencies=["corbell"])


# ---------------------------------------------------------------------------
# Tool: context_engine_codebase_retrieval
# ---------------------------------------------------------------------------

@mcp.tool()
def context_engine_codebase_retrieval(
    query: str,
    workspace_full_path: str = "",
) -> str:
    """Search the indexed codebase and return relevant code snippets.

    Returns formatted code blocks with absolute file paths and line numbers,
    ready for injection into an LLM context window.

    Args:
        query: Natural language description of the code you're looking for.
        workspace_full_path: Full path to the workspace (repository) root directory.
            Falls back to CORBELL_WORKSPACE env var if empty.

    Returns:
        Formatted code snippets, or an error string on failure.
    """
    try:
        workspace_path_str = _resolve_workspace(workspace_full_path)
        if workspace_path_str is None:
            return (
                "Error: workspace_full_path is required. "
                "Pass the full path to the workspace (repository) root directory."
            )

        from pathlib import Path
        from corbell.core.workspace import build_config, db_path_for_workspace
        from corbell.core.embeddings.sqlite_store import SQLiteEmbeddingStore
        from corbell.core.indexing.tracker import IndexTracker
        from corbell.core.indexing.builder import IndexBuilder

        ws_path = Path(workspace_path_str).resolve()

        if not ws_path.exists():
            return (
                f"Error: Workspace directory not found: {ws_path}. "
                "Ensure the path points to a valid repository root."
            )

        cfg = build_config(ws_path)
        db_path = db_path_for_workspace(ws_path, model=cfg.storage.resolved_model())

        try:
            emb_store = SQLiteEmbeddingStore(db_path)
        except Exception:
            return (
                f"Error: Database corrupted at {db_path}. "
                "Run 'corbell index build --rebuild' to recreate."
            )

        # Check index status
        try:
            chunk_count = emb_store.count()
        except Exception:
            return (
                f"Error: Database corrupted at {db_path}. "
                "Run 'corbell index build --rebuild' to recreate."
            )

        if chunk_count == 0:
            import logging
            logging.getLogger(__name__).info(
                "Index is empty — running full build now (this may take a while)..."
            )
            builder = IndexBuilder()
            builder.build(cfg, db_path, rebuild=True)

        # Blocking incremental rebuild if stale (MCP never does full build)
        tracker = IndexTracker(db_path)
        stale_result = tracker.get_stale_files(cfg.repos, cfg)
        if stale_result.has_changes:
            try:
                builder = IndexBuilder()
                builder.build(cfg, db_path, rebuild=False)
            except Exception:
                # Non-fatal: proceed with current index
                pass

        # Run the retrieval pipeline
        from corbell.core.query.engine import codebase_retrieval

        result = codebase_retrieval(
            query=query,
            workspace_path=ws_path,
            top_k=50,
            use_llm=True,
            rerank=True,
        )

        return result

    except Exception as exc:
        return f"Error: Unexpected failure in codebase_retrieval: {exc}"


def _resolve_workspace(workspace_full_path: str) -> Optional[str]:
    """Resolve the workspace path from parameter or env var."""
    # 1. Explicit path provided
    if workspace_full_path and workspace_full_path.strip():
        return workspace_full_path.strip()

    # 2. Environment variable
    env_path = os.environ.get("CORBELL_WORKSPACE")
    if env_path:
        return env_path

    return None


# ---------------------------------------------------------------------------
# Server entry point
# ---------------------------------------------------------------------------

def serve(transport: str = "stdio", port: int = 8000) -> None:
    """Run the MCP server.

    Args:
        transport: 'stdio' for pipe-based IDE integration, 'sse' for HTTP server.
        port: Port number for SSE transport (ignored for stdio).
    """
    if transport == "sse":
        mcp.settings.port = port
    mcp.run(transport=transport)


def main() -> None:
    """Entry point for `uvx codebase-retrieval-context-engine`."""
    import argparse

    parser = argparse.ArgumentParser(description="Codebase Retrieval Context Engine MCP Server")
    parser.add_argument(
        "--transport", "-t", default="stdio", choices=["stdio", "sse"],
        help="Transport mode (default: stdio)",
    )
    parser.add_argument(
        "--port", "-p", type=int, default=8000,
        help="Port for SSE transport (default: 8000)",
    )
    args = parser.parse_args()
    serve(transport=args.transport, port=args.port)
