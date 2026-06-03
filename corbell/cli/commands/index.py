"""CLI: corbell index — build and manage the code search index."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

app = typer.Typer(no_args_is_help=True, help="Code index commands.")
console = Console()


@app.command("build")
def build(
    rebuild: bool = typer.Option(
        False, "--rebuild", help="Clear existing index and perform a full rebuild."
    ),
    workspace: Optional[str] = typer.Option(
        None, "--workspace", "-w", help="Path to workspace.yaml or its directory."
    ),
    repo: Optional[str] = typer.Option(
        None, "--repo", help="Only index a specific repo by ID."
    ),
) -> None:
    """Build (or incrementally update) the code search index.

    On first run, indexes all repos. On subsequent runs, only re-embeds
    changed files and rebuilds the graph for affected repos.

    Use --rebuild to force a full re-index from scratch.
    """
    from corbell.core.workspace import load_workspace, find_workspace_root

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

    try:
        cfg = load_workspace(ws_path)
    except FileNotFoundError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(1)

    config_dir = ws_path.parent if ws_path.is_file() else ws_path

    from corbell.core.indexing.builder import IndexBuilder
    builder = IndexBuilder()

    mode = "Full rebuild" if rebuild else "Incremental build"
    target = f" (repo: {repo})" if repo else ""
    console.print(f"[bold]{mode}{target}[/bold] starting...")

    try:
        result = builder.build(cfg, config_dir, rebuild=rebuild, repo_filter=repo)
    except ValueError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(1)

    status = result.get("status", "unknown")
    if status == "clean":
        console.print("[green]Index is up to date — no changes detected.[/green]")
    else:
        chunks = result.get("chunks_added", 0)
        repos_rebuilt = result.get("repos_rebuilt", 0)
        console.print(f"[green]Done.[/green] {chunks} chunks indexed, {repos_rebuilt} repos rebuilt.")
        if status == "incremental":
            added = result.get("files_added", 0)
            modified = result.get("files_modified", 0)
            deleted = result.get("files_deleted", 0)
            console.print(
                f"  Files: [cyan]+{added}[/cyan] added, "
                f"[yellow]~{modified}[/yellow] modified, "
                f"[red]-{deleted}[/red] deleted"
            )
