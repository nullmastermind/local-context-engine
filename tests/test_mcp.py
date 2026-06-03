"""Tests for the MCP server — tool registration, schema, and error handling."""

from __future__ import annotations

import io
import sys
from unittest.mock import patch

import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def temp_workspace(tmp_path):
    """Create a minimal workspace with new repos schema."""
    ws_dir = tmp_path / "corbell"
    ws_dir.mkdir()
    yaml_file = ws_dir / "workspace.yaml"
    yaml_file.write_text("""\
version: "1"
workspace:
  name: test-workspace
repos: []
llm:
  provider: ollama
  model: llama3
""")
    return tmp_path


# ---------------------------------------------------------------------------
# Tool Registration Tests (Task 8.9)
# ---------------------------------------------------------------------------

def test_mcp_server_registers_exactly_one_tool():
    """The server must register exactly 1 tool."""
    from corbell.core.mcp.server import mcp

    tools = mcp._tool_manager.list_tools()
    tool_names = [t.name for t in tools]
    assert len(tool_names) == 1, f"Expected 1 tool, got: {tool_names}"
    assert "context_engine_codebase_retrieval" in tool_names


def test_mcp_tool_schema():
    """The single tool must have the expected parameter schema."""
    from corbell.core.mcp.server import mcp

    tools = {t.name: t for t in mcp._tool_manager.list_tools()}
    tool = tools["context_engine_codebase_retrieval"]

    props = tool.parameters["properties"]
    assert "query" in props
    assert "workspace_full_path" in props
    assert "top_k" in props
    assert "rerank" in props

    # query is the only required parameter
    required = tool.parameters.get("required", [])
    assert "query" in required


def test_mcp_tool_empty_index_returns_error(tmp_path):
    """Tool returns error string when index is empty (no chunks in DB)."""
    ws_dir = tmp_path / "corbell"
    ws_dir.mkdir()
    yaml_file = ws_dir / "workspace.yaml"
    yaml_file.write_text("""\
version: "1"
workspace:
  name: test
repos: []
""")

    from corbell.core.mcp.server import context_engine_codebase_retrieval

    result = context_engine_codebase_retrieval(
        query="test query",
        workspace_full_path=str(yaml_file),
    )

    assert isinstance(result, str)
    assert "No index found" in result or "corbell index build" in result


def test_mcp_tool_missing_workspace_returns_error(tmp_path):
    """Tool returns error string when workspace.yaml doesn't exist."""
    from corbell.core.mcp.server import context_engine_codebase_retrieval

    result = context_engine_codebase_retrieval(
        query="test query",
        workspace_full_path=str(tmp_path / "nonexistent" / "workspace.yaml"),
    )

    assert isinstance(result, str)
    assert "Error" in result


def test_mcp_tool_returns_string_not_exception():
    """Tool always returns a string, never raises exceptions."""
    from corbell.core.mcp.server import context_engine_codebase_retrieval

    # Should not raise, even with completely invalid input
    result = context_engine_codebase_retrieval(
        query="anything",
        workspace_full_path="/completely/invalid/path/workspace.yaml",
    )
    assert isinstance(result, str)


# ---------------------------------------------------------------------------
# Workspace Auto-detection Tests
# ---------------------------------------------------------------------------

def test_resolve_workspace_explicit_path(tmp_path):
    """Explicit path is used directly."""
    from corbell.core.mcp.server import _resolve_workspace

    explicit = str(tmp_path / "workspace.yaml")
    result = _resolve_workspace(explicit)
    assert result == explicit


def test_resolve_workspace_env_var(tmp_path, monkeypatch):
    """CORBELL_WORKSPACE env var is used when no explicit path given."""
    from corbell.core.mcp.server import _resolve_workspace

    monkeypatch.setenv("CORBELL_WORKSPACE", str(tmp_path))
    result = _resolve_workspace("")
    assert result == str(tmp_path)


def test_resolve_workspace_returns_none_when_nothing_found(tmp_path, monkeypatch):
    """Returns None when no path can be found."""
    from corbell.core.mcp.server import _resolve_workspace

    monkeypatch.delenv("CORBELL_WORKSPACE", raising=False)

    # Run from an isolated dir that has no workspace.yaml
    with patch("corbell.core.workspace.find_workspace_root", return_value=None):
        result = _resolve_workspace("")
    assert result is None


# ---------------------------------------------------------------------------
# FilteredStdin Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_filtered_stdin_drops_empty_lines():
    """_FilteredStdin silently drops empty/whitespace lines."""
    from corbell.core.mcp.server import _FilteredStdin

    test_input = '\n\n  \n{"valid":"json"}\n\n\t\n{"second":"line"}\n'
    fake_stdin = io.StringIO(test_input)

    with patch.object(sys, "stdin", fake_stdin):
        filtered = _FilteredStdin()
        results = []
        async for line in filtered:
            results.append(line)

    assert len(results) == 2
    assert '{"valid":"json"}\n' in results
    assert '{"second":"line"}\n' in results


@pytest.mark.asyncio
async def test_filtered_stdin_handles_eof():
    """_FilteredStdin raises StopAsyncIteration on EOF."""
    from corbell.core.mcp.server import _FilteredStdin

    with patch("sys.stdin") as mock_stdin:
        mock_stdin.readline = lambda: ""  # Immediate EOF

        filtered = _FilteredStdin()
        results = []
        async for line in filtered:
            results.append(line)

    assert results == []
