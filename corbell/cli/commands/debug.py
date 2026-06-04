"""CLI: corbell debug — launch a Gradio UI for inspecting query pipeline internals."""

from __future__ import annotations

import os
from pathlib import Path

import typer
from rich.console import Console

app = typer.Typer(no_args_is_help=False, help="Query debug UI commands.")
console = Console()


@app.callback(invoke_without_command=True)
def debug(
    ctx: typer.Context,
    workspace: str = typer.Option(
        "",
        "--workspace",
        "-w",
        help="Path to the workspace root (default: current directory).",
    ),
    port: int = typer.Option(7860, "--port", "-p", help="Port for the Gradio server."),
    share: bool = typer.Option(False, "--share", help="Create a public Gradio share link."),
) -> None:
    """Launch the Gradio debug UI for inspecting the query pipeline.

    The UI lets you run a query against a workspace and inspect:
      - Per-phase timing
      - Final formatted results
      - Pre-rerank chunk table (file, lines, score, symbol, type)
      - LLM rerank prompts and raw response
    """
    if ctx.invoked_subcommand is not None:
        return

    try:
        import gradio as gr  # type: ignore[import-untyped]
    except ImportError:
        console.print(
            "[red]Gradio is not installed. Install it with:[/red]\n"
            "  pip install 'codebase-retrieval-context-engine[debug]'"
        )
        raise typer.Exit(1)

    default_workspace = workspace or os.environ.get("CORBELL_WORKSPACE") or str(Path.cwd())

    def run_query(workspace_path: str, query: str):  # type: ignore[no-untyped-def]
        """Run the debug pipeline and return Gradio component values."""
        from corbell.core.query.engine import codebase_retrieval_debug

        if not query.strip():
            return (
                "",              # error_box
                "",              # timing_md
                "",              # final_results
                [],              # pre_rerank_table
                "",              # rerank_system
                "",              # rerank_user
                "",              # rerank_response
            )

        ws = workspace_path.strip() or default_workspace
        result = codebase_retrieval_debug(query=query, workspace_path=ws)

        # --- Error banner ---
        error_text = result.error or ""

        # --- Timing table ---
        timing = result.diagnostics.timing if result.diagnostics else {}
        if timing:
            rows = "".join(
                f"| {phase} | {elapsed:.3f}s |\n"
                for phase, elapsed in timing.items()
            )
            timing_md = (
                "| Phase | Elapsed |\n"
                "|---|---|\n"
                + rows
            )
        else:
            timing_md = "_No timing data available._"

        # --- Final results ---
        final_results = result.final_output or ""

        # --- Pre-rerank table ---
        pre_rerank_rows = []
        graph_ids = set()
        if result.diagnostics and result.diagnostics.graph_chunk_ids:
            graph_ids = result.diagnostics.graph_chunk_ids
        for chunk in result.pre_rerank_chunks:
            chunk_id = getattr(chunk, "chunk_id", "")
            parts = chunk_id.split("+") if chunk_id else []
            has_graph = any(p in graph_ids for p in parts) if graph_ids else False
            has_embedding = any(p not in graph_ids for p in parts) if graph_ids else True
            if has_graph and has_embedding and len(parts) > 1:
                source = "embedding+graph"
            elif has_graph:
                source = "graph"
            else:
                source = "embedding"
            pre_rerank_rows.append([
                getattr(chunk, "file_path", ""),
                f"{getattr(chunk, 'start_line', '')}-{getattr(chunk, 'end_line', '')}",
                f"{getattr(chunk, 'score', 0.0):.4f}",
                getattr(chunk, "symbol", "") or "",
                getattr(chunk, "chunk_type", "") or "",
                source,
                getattr(chunk, "content", "") or "",
            ])

        # --- Rerank prompts ---
        detail = result.rerank_detail
        if detail is None or not detail.system_prompt:
            rerank_system = "_LLM not configured — reranking skipped_"
            rerank_user = ""
            rerank_response = ""
        else:
            rerank_system = detail.system_prompt
            rerank_user = detail.user_prompt
            rerank_response = detail.raw_response or "_No response (LLM call failed)_"

        return (
            error_text,
            timing_md,
            final_results,
            pre_rerank_rows,
            rerank_system,
            rerank_user,
            rerank_response,
        )

    with gr.Blocks(title="Corbell Query Debugger") as demo:
        gr.Markdown("# Corbell Query Debugger")
        gr.Markdown("Inspect query pipeline internals: timing, pre-rerank chunks, and LLM rerank prompts.")

        with gr.Row():
            workspace_input = gr.Textbox(
                label="Workspace Path",
                value=default_workspace,
                placeholder="Path to repository root",
                scale=2,
            )
            query_input = gr.Textbox(
                label="Query",
                placeholder="e.g. authentication middleware",
                scale=3,
            )

        run_btn = gr.Button("Run Query", variant="primary")

        error_box = gr.Textbox(
            label="Error",
            visible=True,
            interactive=False,
            lines=2,
        )

        timing_md = gr.Markdown(label="Timing")

        with gr.Tabs():
            with gr.Tab("Final Results"):
                final_output = gr.Code(label="Formatted Output", language=None)

            with gr.Tab("Pre-Rerank Chunks"):
                pre_rerank_table = gr.Dataframe(
                    headers=["File", "Lines", "Score", "Symbol", "Type", "Source", "Content"],
                    datatype=["str", "str", "str", "str", "str", "str", "str"],
                    label="Chunks before reranking",
                    wrap=False,
                )

            with gr.Tab("LLM Rerank"):
                rerank_system_box = gr.Textbox(
                    label="System Prompt",
                    lines=6,
                    interactive=False,
                )
                rerank_user_box = gr.Textbox(
                    label="User Prompt",
                    lines=12,
                    interactive=False,
                )
                rerank_response_box = gr.Textbox(
                    label="Raw LLM Response",
                    lines=4,
                    interactive=False,
                )

        run_btn.click(
            fn=run_query,
            inputs=[workspace_input, query_input],
            outputs=[
                error_box,
                timing_md,
                final_output,
                pre_rerank_table,
                rerank_system_box,
                rerank_user_box,
                rerank_response_box,
            ],
        )

    console.print(f"[green]Starting Corbell debug UI on port {port}...[/green]")
    demo.launch(server_port=port, share=share)
