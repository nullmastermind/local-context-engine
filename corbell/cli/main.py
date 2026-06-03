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
        "  corbell init             Create workspace.yaml\n\n"
        "  corbell index build      Scan repos, build search index\n\n"
        "  corbell query search     Search codebase with natural language\n\n"
        "  corbell mcp serve        Start MCP server for IDE integration"
    ),
    no_args_is_help=True,
    pretty_exceptions_show_locals=False,
)
console = Console()


# ---------------------------------------------------------------------------
# corbell init
# ---------------------------------------------------------------------------

@app.command("init")
def init(
    directory: str = typer.Option(None, "--dir", "-d", help="Target directory (default: cwd)."),
    force: bool = typer.Option(False, "--force", "-f", help="Overwrite existing workspace.yaml."),
) -> None:
    """Initialize a Corbell workspace (creates corbell/workspace.yaml)."""
    from corbell.core.workspace import init_workspace_yaml

    target = (Path(directory) if directory else Path.cwd()).resolve()
    ws_file = target / "corbell" / "workspace.yaml"

    if ws_file.exists() and not force:
        console.print(
            f"[yellow]workspace.yaml already exists at {ws_file}[/yellow]\n"
            "Use --force to overwrite."
        )
        raise typer.Exit(0)

    out = init_workspace_yaml(target)
    console.print(f"[green]Created[/green] [bold]{out}[/bold]")
    console.print("\nNext steps:")
    console.print("  1. Edit [bold]corbell/workspace.yaml[/bold] — add your repo paths")
    console.print("  2. Set [bold]ANTHROPIC_API_KEY[/bold] or [bold]OPENAI_API_KEY[/bold]")
    console.print("  3. [bold]corbell index build[/bold]")
    console.print('  4. [bold]corbell query search "your question"[/bold]')


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
