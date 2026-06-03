"""CLI: corbell query — search the codebase with natural language."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

app = typer.Typer(no_args_is_help=True, help="Code search commands.")
console = Console()


@app.command("search")
def search(
    query_text: str = typer.Argument(..., help="Natural language query."),
    top: int = typer.Option(50, "--top", "-n", help="Maximum number of chunks to return."),
    no_llm: bool = typer.Option(False, "--no-llm", help="Disable LLM enhancement and reranking."),
    no_rerank: bool = typer.Option(False, "--no-rerank", help="Disable LLM reranking."),
    workspace: Optional[str] = typer.Option(
        None, "--workspace", "-w", help="Path to workspace.yaml or its directory."
    ),
) -> None:
    """Search the indexed codebase using natural language.

    Returns relevant code snippets with file paths and line numbers,
    formatted for use as LLM context.
    """
    from corbell.core.workspace import find_workspace_root

    # Resolve workspace path
    if workspace:
        ws_path = Path(workspace)
    else:
        found = find_workspace_root()
        if not found:
            console.print("[red]Error:[/red] workspace.yaml not found. Run 'corbell init' first.")
            raise typer.Exit(1)
        ws_path = found / "workspace.yaml"

    if not ws_path.exists() and ws_path.is_dir():
        ws_path = ws_path / "workspace.yaml"

    from corbell.core.query.engine import codebase_retrieval

    use_llm = not no_llm
    do_rerank = use_llm and not no_rerank

    result = codebase_retrieval(
        query=query_text,
        workspace_path=ws_path,
        top_k=top,
        use_llm=use_llm,
        rerank=do_rerank,
    )

    if result.startswith("Error:") or result.startswith("No index"):
        console.print(f"[red]{result}[/red]")
        raise typer.Exit(1)

    # Print warnings to stderr if present
    if result.startswith("[warnings:"):
        first_newline = result.find("\n\n")
        if first_newline != -1:
            warning_line = result[:first_newline]
            code_part = result[first_newline + 2:]
            console.print(f"[yellow]{warning_line}[/yellow]")
            import sys
            print(warning_line, file=sys.stderr)
            print(code_part)
            return

    print(result)
