"""MCP Server for Corbell code retrieval engine.

Exposes a single tool `context_engine_codebase_retrieval` via FastMCP,
supporting both stdio and SSE transports.
"""

from __future__ import annotations

import asyncio
import os
import sys
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
    top_k: int = 50,
    rerank: bool = True,
) -> str:
    """Search the indexed codebase and return relevant code snippets.

    Returns formatted code blocks with absolute file paths and line numbers,
    ready for injection into an LLM context window.

    Args:
        query: Natural language description of the code you're looking for.
        workspace_full_path: Full path to the workspace (repository) root directory.
            Falls back to CORBELL_WORKSPACE env var if empty.
        top_k: Maximum number of code chunks to return (default 50).
        rerank: Whether to use LLM reranking for better relevance (default true).

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
        db_path = db_path_for_workspace(ws_path)

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
            top_k=top_k,
            use_llm=True,
            rerank=rerank,
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
# Filtered stdin wrapper — prevents empty-line crashes in MCP SDK
# ---------------------------------------------------------------------------

class _FilteredStdin:
    """Async iterator over stdin that silently drops empty/whitespace lines.

    The MCP SDK's stdio transport passes every raw line from sys.stdin to
    Pydantic's JSONRPCMessage.model_validate_json(). Empty newlines fail
    validation and crash the server. This wrapper filters them out.
    """

    def __init__(self) -> None:
        self._reader = None

    def __aiter__(self):
        return self

    async def __anext__(self) -> str:
        loop = asyncio.get_event_loop()
        while True:
            line = await loop.run_in_executor(None, sys.stdin.readline)
            if not line:  # EOF
                raise StopAsyncIteration
            if line.strip():  # Only forward non-empty lines
                return line
            # Empty/whitespace lines are silently dropped


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
        print(f"Corbell MCP server starting on http://localhost:{port}/sse ...", file=sys.stderr)
        mcp.settings.port = port
        mcp.run(transport="sse")
    else:
        print("Corbell MCP server starting on stdio...", file=sys.stderr)

        async def _run():
            from mcp.server.stdio import stdio_server

            filtered = _FilteredStdin()
            async with stdio_server(stdin=filtered) as (read_stream, write_stream):
                await mcp._mcp_server.run(
                    read_stream,
                    write_stream,
                    mcp._mcp_server.create_initialization_options(),
                )

        asyncio.run(_run())
