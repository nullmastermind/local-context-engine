"""Query diagnostics for tracking and surfacing warnings during retrieval."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from corbell.core.query.graph_expander import ScoredChunk
    from corbell.core.query.reranker import RerankResult


@dataclass
class QueryDiagnostics:
    """Accumulates warning counters during a query execution.

    Counters are incremented as the pipeline runs. At the end,
    ``summary()`` returns a warning string if any threshold is exceeded.
    """

    skipped_files: int = 0       # files that no longer exist on disk
    skipped_methods: int = 0     # method nodes that couldn't be expanded
    graph_expansion_failures: int = 0  # graph lookups that failed

    # Thresholds for emitting warnings
    _FILE_THRESHOLD: int = field(default=3, init=False, repr=False)
    _METHOD_THRESHOLD: int = field(default=5, init=False, repr=False)

    # Timing: phase name -> elapsed seconds
    timing: Dict[str, float] = field(default_factory=dict)

    # Debug mode: when True, pre-rerank chunks and rerank detail are captured
    collect_debug: bool = False
    pre_rerank_chunks: Optional[List["ScoredChunk"]] = None
    rerank_detail: Optional["RerankResult"] = None

    def record_time(self, phase: str, elapsed: float) -> None:
        """Record elapsed time for a named pipeline phase."""
        self.timing[phase] = elapsed

    def summary(self) -> Optional[str]:
        """Return a warning string if any counter exceeds its threshold.

        Returns:
            Warning string suitable for display, or None if everything is fine.
        """
        parts = []
        if self.skipped_files >= self._FILE_THRESHOLD:
            parts.append(f"{self.skipped_files} files missing (index may be stale)")
        if self.skipped_methods >= self._METHOD_THRESHOLD:
            parts.append(f"{self.skipped_methods} methods skipped")
        if not parts:
            return None
        return "; ".join(parts)
