"""Background index worker — invoked as: python -m corbell.core.indexing._worker <workspace_path>

This module is the entry point for background incremental index rebuilds.
It is intentionally not exposed in the CLI help to keep the interface clean.
Spawned by the query engine when stale index is detected and last build < 1 day old.
"""

from __future__ import annotations

import sys


def main() -> None:
    """Run an incremental index build for the given workspace path."""
    if len(sys.argv) < 2:
        print("Usage: python -m corbell.core.indexing._worker <workspace_path>", file=sys.stderr)
        sys.exit(1)

    workspace_path_str = sys.argv[1]

    try:
        from pathlib import Path
        from corbell.core.workspace import build_config, db_path_for_workspace
        from corbell.core.indexing.builder import IndexBuilder

        workspace_path = Path(workspace_path_str).resolve()
        cfg = build_config(workspace_path)
        db_path = db_path_for_workspace(workspace_path)

        builder = IndexBuilder()
        result = builder.build(cfg, db_path, rebuild=False)

        print(f"Background index complete: {result}", file=sys.stderr)

    except Exception as exc:
        print(f"Background index worker error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
