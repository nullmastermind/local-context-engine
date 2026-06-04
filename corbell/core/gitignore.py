"""Gitignore-aware path matching for file discovery."""

from __future__ import annotations

from pathlib import Path
from typing import List

import pathspec

from corbell.core.constants import SKIP_DIRS


def load_gitignore(repo_path: Path) -> pathspec.PathSpec:
    """Load all .gitignore rules for a repo and return a combined matcher.

    Collects patterns from:
    - .git/info/exclude
    - Root .gitignore
    - Nested .gitignore files (with proper path anchoring)

    Returns a PathSpec that matches repo-root-relative paths.
    If no gitignore files exist, returns an empty matcher (matches nothing).
    """
    lines: List[str] = []

    # .git/info/exclude
    exclude = repo_path / ".git" / "info" / "exclude"
    if exclude.is_file():
        lines.extend(_read_patterns(exclude, rel_dir=""))

    # Root .gitignore
    root_gi = repo_path / ".gitignore"
    if root_gi.is_file():
        lines.extend(_read_patterns(root_gi, rel_dir=""))

    # Nested .gitignore files
    for gi in repo_path.rglob(".gitignore"):
        if gi == root_gi:
            continue
        rel = gi.parent.relative_to(repo_path)
        if any(part in SKIP_DIRS for part in rel.parts):
            continue
        if any(part.startswith(".") for part in rel.parts):
            continue
        lines.extend(_read_patterns(gi, rel_dir=str(rel).replace("\\", "/")))

    return pathspec.PathSpec.from_lines("gitwildmatch", lines)


def _read_patterns(gi_path: Path, rel_dir: str) -> List[str]:
    """Read a .gitignore file and transform patterns to be repo-root-relative."""
    result: List[str] = []
    try:
        content = gi_path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return result

    for raw in content.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue

        negate = ""
        if line.startswith("!"):
            negate = "!"
            line = line[1:]

        if not rel_dir:
            result.append(negate + line)
        else:
            if line.startswith("/"):
                result.append(negate + rel_dir + "/" + line[1:])
            elif "/" in line.rstrip("/"):
                result.append(negate + rel_dir + "/" + line)
            else:
                result.append(negate + rel_dir + "/**/" + line)

    return result
