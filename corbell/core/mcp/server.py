"""MCP Server for Corbell code retrieval engine.

Exposes a single tool `context_engine_codebase_retrieval` via FastMCP,
supporting both stdio and SSE transports.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Optional

import anyio
from mcp.server.fastmcp import FastMCP

# File logger for MCP debugging — always writes to a fixed path
_LOG_PATH = os.environ.get("CORBELL_MCP_LOG", r"D:\projects\Python\local-context-engine\mcp.log")
_mcp_logger = logging.getLogger("corbell.mcp")
_mcp_logger.setLevel(logging.DEBUG)
_fh = logging.FileHandler(_LOG_PATH, mode="a", encoding="utf-8")
_fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] [PID:%(process)d] %(message)s"))
_mcp_logger.addHandler(_fh)


# Create the FastMCP server
mcp = FastMCP("corbell", dependencies=["corbell"])

_mcp_logger.info("=" * 60)
_mcp_logger.info("MCP server module loaded")
_mcp_logger.info("PID: %s", os.getpid())
_mcp_logger.info("CWD: %s", os.getcwd())
_mcp_logger.info("ENV snapshot: CORBELL_WORKSPACE=%s", os.environ.get("CORBELL_WORKSPACE"))
_mcp_logger.info("ENV snapshot: CORBELL_LLM_PROVIDER=%s", os.environ.get("CORBELL_LLM_PROVIDER"))
_mcp_logger.info("ENV snapshot: CORBELL_EMBEDDING_MODEL=%s", os.environ.get("CORBELL_EMBEDDING_MODEL"))
_mcp_logger.info("ENV snapshot: GOOGLE_API_KEY=%s", "SET" if os.environ.get("GOOGLE_API_KEY") else "UNSET")
_mcp_logger.info("ENV snapshot: ANTHROPIC_API_KEY=%s", "SET" if os.environ.get("ANTHROPIC_API_KEY") else "UNSET")
_mcp_logger.info("ENV snapshot: VOYAGE_API_KEY=%s", "SET" if os.environ.get("VOYAGE_API_KEY") else "UNSET")

# Pre-import heavy modules at startup so first tool call doesn't block on imports
_t_import = time.time()
_mcp_logger.info("Pre-importing heavy modules...")
from pathlib import Path  # noqa: E402
from corbell.core.query.engine import codebase_retrieval  # noqa: E402
import voyageai  # noqa: E402, F401
try:
    from google import genai  # noqa: E402, F401
except ImportError:
    pass
_mcp_logger.info("Pre-import done (%.3fs)", time.time() - _t_import)

# Route engine logger to the same MCP log file
_engine_logger = logging.getLogger("corbell.core.query.engine")
_engine_logger.setLevel(logging.DEBUG)
_engine_logger.addHandler(_fh)

# Route embeddings model logger to MCP log too
_emb_logger = logging.getLogger("corbell.core.embeddings.model")
_emb_logger.setLevel(logging.DEBUG)
_emb_logger.addHandler(_fh)

# Route workspace config logger
_ws_logger = logging.getLogger("corbell.core.workspace")
_ws_logger.setLevel(logging.DEBUG)
_ws_logger.addHandler(_fh)


# ---------------------------------------------------------------------------
# Tool: context_engine_codebase_retrieval
# ---------------------------------------------------------------------------

@mcp.tool()
async def context_engine_codebase_retrieval(
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
    t0 = time.time()
    _mcp_logger.info("-" * 40)
    _mcp_logger.info("TOOL CALL: query=%r, workspace_full_path=%r", query, workspace_full_path)

    try:
        workspace_path_str = _resolve_workspace(workspace_full_path)
        _mcp_logger.info("Resolved workspace: %s (%.3fs)", workspace_path_str, time.time() - t0)
        if workspace_path_str is None:
            _mcp_logger.warning("No workspace resolved — returning error")
            return (
                "Error: workspace_full_path is required. "
                "Pass the full path to the workspace (repository) root directory."
            )

        ws_path = Path(workspace_path_str).resolve()

        if not ws_path.exists():
            _mcp_logger.error("Workspace path does not exist: %s", ws_path)
            return (
                f"Error: Workspace directory not found: {ws_path}. "
                "Ensure the path points to a valid repository root."
            )

        t1 = time.time()
        _mcp_logger.info("Starting codebase_retrieval query...")

        # Must run in a worker thread: the retrieval pipeline makes blocking
        # HTTP calls (Voyage/Google embedding APIs). If run directly on the
        # event loop, those calls deadlock because httpx's sync transport
        # tries to use the already-blocked event loop.
        def _run_pipeline():
            import threading
            _mcp_logger.info("Pipeline thread started: thread=%s", threading.current_thread().name)
            return codebase_retrieval(
                query=query,
                workspace_path=ws_path,
                top_k=50,
                use_llm=True,
                rerank=True,
            )

        result = await anyio.to_thread.run_sync(_run_pipeline, cancellable=True)

        _mcp_logger.info(
            "codebase_retrieval done (%.3fs), result length=%d chars", time.time() - t1, len(result)
        )
        _mcp_logger.info("TOTAL tool call time: %.3fs", time.time() - t0)
        return result

    except Exception as exc:
        _mcp_logger.exception("Unexpected failure in tool call")
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
    _mcp_logger.info("serve() called: transport=%s, port=%d", transport, port)
    if transport == "sse":
        mcp.settings.port = port
    mcp.run(transport=transport)


def main() -> None:
    """Entry point for `uvx codebase-retrieval-context-engine`."""
    import argparse

    _mcp_logger.info("main() entry point invoked")
    _mcp_logger.info("All env vars: %s", {k: v for k, v in os.environ.items() if k.startswith(("CORBELL_", "ANTHROPIC_", "GOOGLE_", "VOYAGE_", "OPENAI_", "AWS_", "AZURE_", "GCP_", "BEDROCK_"))})

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
