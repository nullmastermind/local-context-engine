"""Tests for graph_expander — BFS call-chain expansion."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from corbell.core.query.graph_expander import ScoredChunk, expand_via_graph
from corbell.core.query.diagnostics import QueryDiagnostics


def _make_chunk(
    chunk_id: str,
    file_path: str,
    start: int,
    end: int,
    score: float = 0.8,
    repo_id: str = "repo",
) -> ScoredChunk:
    return ScoredChunk(
        chunk_id=chunk_id,
        score=score,
        file_path=file_path,
        start_line=start,
        end_line=end,
        content="code",
        repo_id=repo_id,
    )


@pytest.fixture
def repo_path(tmp_path) -> Path:
    """Create a temporary file for testing."""
    f = tmp_path / "auth.py"
    f.write_text("\n".join([f"line {i}" for i in range(100)]))
    return tmp_path


def _make_method_node(
    method_id: str,
    file_path: str,
    line_start: int,
    line_end: int,
    service_id: str = "repo",
) -> MagicMock:
    m = MagicMock()
    m.id = method_id
    m.file_path = file_path
    m.line_start = line_start
    m.line_end = line_end
    m.service_id = service_id
    m.method_name = method_id.split("::")[-1]
    return m


# ---------------------------------------------------------------------------
# expand_via_graph with empty/no results
# ---------------------------------------------------------------------------

def test_expand_empty_input():
    """expand_via_graph returns empty list for empty input."""
    graph_store = MagicMock()
    result = expand_via_graph([], graph_store, repos=[], max_depth=2, max_chunks=30)
    assert result == []


def test_expand_no_matching_methods(tmp_path):
    """No expansion when no method nodes match the chunk's file/lines."""
    f = tmp_path / "nofile.py"
    f.write_text("code\n" * 50)

    chunk = _make_chunk("c1", str(f), 1, 10, score=0.8)

    graph_store = MagicMock()
    graph_store.get_all_services.return_value = [MagicMock(id="repo")]
    graph_store.get_methods_for_service.return_value = []  # no methods

    repo_cfg = MagicMock()
    repo_cfg.id = "repo"
    repo_cfg.resolved_path = tmp_path

    result = expand_via_graph([chunk], graph_store, [repo_cfg])
    assert result == []


# ---------------------------------------------------------------------------
# Score cascade
# ---------------------------------------------------------------------------

def test_score_cascade_callers(tmp_path):
    """Callers get parent_score * 0.6."""
    auth_py = tmp_path / "auth.py"
    auth_py.write_text("\n".join([f"line{i}" for i in range(50)]))

    caller_py = tmp_path / "caller.py"
    caller_py.write_text("\n".join([f"line{i}" for i in range(50)]))

    parent_score = 0.8
    chunk = _make_chunk("c1", str(auth_py), 1, 20, score=parent_score)

    target_method = _make_method_node("repo::auth.py::verify", str(auth_py), 1, 20)
    caller_method = _make_method_node("repo::caller.py::call_verify", str(caller_py), 1, 10)

    graph_store = MagicMock()
    graph_store.get_all_services.return_value = [MagicMock(id="repo")]
    graph_store.get_methods_for_service.return_value = [target_method]
    graph_store.get_callers_of_method.return_value = [caller_method]
    graph_store.get_dependencies.return_value = []

    repo_cfg = MagicMock()
    repo_cfg.id = "repo"
    repo_cfg.resolved_path = tmp_path

    result = expand_via_graph([chunk], graph_store, [repo_cfg], max_depth=1, max_chunks=30)

    assert len(result) == 1
    bonus = result[0]
    assert bonus.chunk_id == caller_method.id
    assert abs(bonus.score - parent_score * 0.6) < 1e-5


# ---------------------------------------------------------------------------
# Global cap
# ---------------------------------------------------------------------------

def test_global_cap_enforced(tmp_path):
    """Expansion stops at max_chunks bonus results."""
    auth_py = tmp_path / "auth.py"
    auth_py.write_text("\n".join([f"line{i}" for i in range(200)]))

    chunk = _make_chunk("c1", str(auth_py), 1, 10, score=0.9)
    target = _make_method_node("repo::auth.py::method", str(auth_py), 1, 10)

    # Create many callers
    callers = []
    for i in range(50):
        cf = tmp_path / f"caller_{i}.py"
        cf.write_text("code\n")
        callers.append(_make_method_node(f"repo::caller_{i}.py::fn", str(cf), 1, 5))

    graph_store = MagicMock()
    graph_store.get_all_services.return_value = [MagicMock(id="repo")]
    graph_store.get_methods_for_service.return_value = [target]
    graph_store.get_callers_of_method.return_value = callers
    graph_store.get_dependencies.return_value = []

    repo_cfg = MagicMock()
    repo_cfg.id = "repo"
    repo_cfg.resolved_path = tmp_path

    result = expand_via_graph([chunk], graph_store, [repo_cfg], max_depth=1, max_chunks=5)
    assert len(result) <= 5


# ---------------------------------------------------------------------------
# Score floor
# ---------------------------------------------------------------------------

def test_score_floor_stops_low_confidence(tmp_path):
    """Expansion stops when calculated score falls below 0.15."""
    auth_py = tmp_path / "auth.py"
    auth_py.write_text("code\n" * 20)
    caller_py = tmp_path / "caller.py"
    caller_py.write_text("code\n" * 20)

    # low parent score — child would be 0.2 * 0.6 = 0.12 < floor
    chunk = _make_chunk("c1", str(auth_py), 1, 10, score=0.2)
    target = _make_method_node("repo::auth.py::method", str(auth_py), 1, 10)
    caller = _make_method_node("repo::caller.py::fn", str(caller_py), 1, 5)

    graph_store = MagicMock()
    graph_store.get_all_services.return_value = [MagicMock(id="repo")]
    graph_store.get_methods_for_service.return_value = [target]
    graph_store.get_callers_of_method.return_value = [caller]
    graph_store.get_dependencies.return_value = []

    repo_cfg = MagicMock()
    repo_cfg.id = "repo"
    repo_cfg.resolved_path = tmp_path

    result = expand_via_graph([chunk], graph_store, [repo_cfg], max_depth=1, max_chunks=30)
    assert result == []


# ---------------------------------------------------------------------------
# Missing file handling
# ---------------------------------------------------------------------------

def test_missing_file_skipped(tmp_path):
    """Missing file during expansion increments diagnostics and skips that method."""
    auth_py = tmp_path / "auth.py"
    auth_py.write_text("code\n" * 20)

    chunk = _make_chunk("c1", str(auth_py), 1, 10, score=0.8)
    target = _make_method_node("repo::auth.py::method", str(auth_py), 1, 10)
    caller = _make_method_node(
        "repo::missing.py::fn",
        str(tmp_path / "nonexistent.py"),  # file doesn't exist
        1, 5,
    )

    graph_store = MagicMock()
    graph_store.get_all_services.return_value = [MagicMock(id="repo")]
    graph_store.get_methods_for_service.return_value = [target]
    graph_store.get_callers_of_method.return_value = [caller]
    graph_store.get_dependencies.return_value = []

    repo_cfg = MagicMock()
    repo_cfg.id = "repo"
    repo_cfg.resolved_path = tmp_path

    diag = QueryDiagnostics()
    result = expand_via_graph([chunk], graph_store, [repo_cfg], max_depth=1, max_chunks=30,
                               diagnostics=diag)

    assert result == []
    assert diag.skipped_files == 1


# ---------------------------------------------------------------------------
# Visited set prevents cycles
# ---------------------------------------------------------------------------

def test_visited_set_prevents_infinite_loop(tmp_path):
    """BFS visited set prevents revisiting methods (no infinite loop)."""
    a_py = tmp_path / "a.py"
    a_py.write_text("code\n" * 20)
    b_py = tmp_path / "b.py"
    b_py.write_text("code\n" * 20)

    chunk = _make_chunk("c1", str(a_py), 1, 10, score=0.8)
    method_a = _make_method_node("repo::a.py::fn_a", str(a_py), 1, 10)
    method_b = _make_method_node("repo::b.py::fn_b", str(b_py), 1, 10)

    # a calls b, b calls a (cycle)
    def callers_of(mid):
        if mid == method_a.id:
            return [method_b]
        if mid == method_b.id:
            return [method_a]
        return []

    graph_store = MagicMock()
    graph_store.get_all_services.return_value = [MagicMock(id="repo")]
    graph_store.get_methods_for_service.return_value = [method_a]
    graph_store.get_callers_of_method.side_effect = callers_of
    graph_store.get_dependencies.return_value = []

    repo_cfg = MagicMock()
    repo_cfg.id = "repo"
    repo_cfg.resolved_path = tmp_path

    # Should not hang
    result = expand_via_graph([chunk], graph_store, [repo_cfg], max_depth=5, max_chunks=30)
    # method_b is added as bonus; method_a is already visited
    assert len(result) <= 30
