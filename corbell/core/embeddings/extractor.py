"""Code chunk extractor for embedding indexing.

Extracts code chunks (functions, classes, methods) from source files.
Extracts function/class/method chunks using Python ast; generic line-split for others.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import pathspec

from corbell.core.constants import EXTENSION_LANG as _SUPPORTED, SKIP_DIRS as _SKIP_DIRS
from corbell.core.gitignore import load_gitignore


@dataclass
class EmbeddingRecord:
    """A chunk of code ready to be embedded."""

    id: str
    service_id: str
    repo: str
    file_path: str  # relative path within repo
    start_line: int
    end_line: int
    content: str
    language: str
    chunk_type: str  # function | class | method | block
    symbol: Optional[str] = None  # function/class name if known
    embedding: Optional[List[float]] = None


class CodeChunkExtractor:
    """Extract meaningful code chunks from source files in a repo.

    Produces :class:`EmbeddingRecord` instances that can be stored in any
    embedding backend.
    """

    def __init__(self, chunk_size: int = 50, overlap: int = 10):
        """Initialize the extractor.

        Args:
            chunk_size: Lines per generic block chunk.
            overlap: Overlap between consecutive generic chunks.
        """
        self.chunk_size = chunk_size
        self.overlap = overlap

    def extract_from_repo(
        self,
        repo_path: Path | str,
        service_id: str,
        max_file_bytes: int = 1024 * 1024,
        gitignore_spec: Optional[pathspec.PathSpec] = None,
    ) -> List[EmbeddingRecord]:
        """Walk a repo and extract all code chunks.

        Args:
            repo_path: Root directory of the repository.
            service_id: ID of the owning service.
            max_file_bytes: Skip files larger than this.
            gitignore_spec: Pre-loaded PathSpec for gitignore filtering.
                If None, it is loaded from the repo automatically.

        Returns:
            List of :class:`EmbeddingRecord` ready for embedding.
        """
        repo_path = Path(repo_path)
        if gitignore_spec is None:
            gitignore_spec = load_gitignore(repo_path)
        records: List[EmbeddingRecord] = []

        for fp in repo_path.rglob("*"):
            if not fp.is_file():
                continue
            if self._should_skip(fp, max_file_bytes):
                continue
            lang = _SUPPORTED.get(fp.suffix)
            if not lang:
                continue
            rel = str(fp.relative_to(repo_path))
            if gitignore_spec.match_file(rel.replace("\\", "/")):
                continue
            chunks = self._extract_file(fp, rel, lang, service_id, str(repo_path))
            records.extend(chunks)

        return records

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    def _should_skip(self, fp: Path, max_bytes: int) -> bool:
        if any(part in _SKIP_DIRS for part in fp.parts):
            return True
        try:
            if fp.stat().st_size > max_bytes:
                return True
        except OSError:
            return True
        return False

    def _extract_file(
        self, fp: Path, rel: str, lang: str, service_id: str, repo: str
    ) -> List[EmbeddingRecord]:
        try:
            content = fp.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            return []

        if lang == "python":
            return self._extract_python(content, rel, service_id, repo)
        return self._extract_generic(content, rel, lang, service_id, repo)

    def _extract_python(
        self, content: str, rel: str, service_id: str, repo: str
    ) -> List[EmbeddingRecord]:
        """Use Python's ast to extract function/class definitions."""
        records: List[EmbeddingRecord] = []
        lines = content.splitlines()

        try:
            tree = ast.parse(content)
        except SyntaxError:
            return self._extract_generic(content, rel, "python", service_id, repo)

        class _Visitor(ast.NodeVisitor):
            def __init__(self_v):
                self_v.class_stack: List[str] = []

            def _emit(self_v, node, name: str, chunk_type: str):
                line_start = node.lineno
                line_end = getattr(node, "end_lineno", node.lineno)
                chunk_content = "\n".join(lines[line_start - 1 : line_end])
                symbol = ".".join(self_v.class_stack + [name])
                rec = EmbeddingRecord(
                    id=f"{service_id}::{rel}::{symbol}",
                    service_id=service_id,
                    repo=repo,
                    file_path=rel,
                    start_line=line_start,
                    end_line=line_end,
                    content=chunk_content,
                    language="python",
                    chunk_type=chunk_type,
                    symbol=symbol,
                )
                records.append(rec)

            def visit_ClassDef(self_v, node):
                self_v._emit(node, node.name, "class")
                self_v.class_stack.append(node.name)
                self_v.generic_visit(node)
                self_v.class_stack.pop()

            def visit_FunctionDef(self_v, node):
                chunk_type = "method" if self_v.class_stack else "function"
                self_v._emit(node, node.name, chunk_type)

            visit_AsyncFunctionDef = visit_FunctionDef

        _Visitor().visit(tree)
        # Fall back to generic if nothing found
        if not records:
            return self._extract_generic(content, rel, "python", service_id, repo)
        return records

    def _extract_generic(
        self, content: str, rel: str, lang: str, service_id: str, repo: str
    ) -> List[EmbeddingRecord]:
        """Split file into overlapping line-based blocks."""
        lines = content.splitlines()
        records: List[EmbeddingRecord] = []
        step = max(1, self.chunk_size - self.overlap)

        for i in range(0, len(lines), step):
            end = min(i + self.chunk_size, len(lines))
            chunk_lines = lines[i:end]
            if not any(l.strip() for l in chunk_lines):
                continue
            chunk_content = "\n".join(chunk_lines)
            records.append(
                EmbeddingRecord(
                    id=f"{service_id}::{rel}::block_{i}",
                    service_id=service_id,
                    repo=repo,
                    file_path=rel,
                    start_line=i + 1,
                    end_line=end,
                    content=chunk_content,
                    language=lang,
                    chunk_type="block",
                )
            )

        return records
