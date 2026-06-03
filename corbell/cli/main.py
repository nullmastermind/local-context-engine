"""Corbell CLI — code retrieval engine entry point."""

from __future__ import annotations

from pathlib import Path

import typer
from dotenv import load_dotenv
from rich.console import Console

from corbell.cli.commands.index import app as index_app
from corbell.cli.commands.query import app as query_app

app = typer.Typer(
    name="corbell",
    help=(
        "Corbell — Code retrieval engine for LLM context injection.\n\n"
        "Quick start:\n\n"
        "  corbell index build      Scan repo, build search index\n\n"
        "  corbell query search     Search codebase with natural language\n\n"
        "  corbell mcp serve        Start MCP server for IDE integration"
    ),
    no_args_is_help=True,
    pretty_exceptions_show_locals=False,
)
console = Console()


# ---------------------------------------------------------------------------
# Sub-apps
# ---------------------------------------------------------------------------

app.add_typer(index_app, name="index", help="Code index commands.")
app.add_typer(query_app, name="query", help="Code search commands.")


# ---------------------------------------------------------------------------
# corbell mcp serve (inline — keep MCP out of removed commands)
# ---------------------------------------------------------------------------

@app.command("mcp")
def mcp_serve(
    transport: str = typer.Option("stdio", "--transport", "-t", help="Transport: stdio or sse."),
    port: int = typer.Option(8000, "--port", "-p", help="Port for SSE transport."),
) -> None:
    """Start the MCP server for IDE integration (Cursor, Claude Desktop, etc.)."""
    from corbell.core.mcp.server import serve
    serve(transport=transport, port=port)


def main() -> None:
    load_dotenv(dotenv_path=Path.cwd() / ".env")
    app()


if __name__ == "__main__":
    main()
