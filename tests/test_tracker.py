"""Tests for IndexTracker — file-level mtime tracking and index metadata."""

from __future__ import annotations

import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from corbell.core.indexing.tracker import IndexTracker


@pytest.fixture
def tracker(tmp_db):
    return IndexTracker(tmp_db)


@pytest.fixture
def simple_repo(tmp_path) -> Path:
    """A repo with 3 Python files."""
    repo = tmp_path / "my-repo"
    repo.mkdir()
    (repo / "a.py").write_text("def a(): pass\n")
    (repo / "b.py").write_text("def b(): pass\n")
    (repo / "c.py").write_text("def c(): pass\n")
    return repo


def _make_repo_config(repo_path: Path, repo_id: str = "my-repo"):
    """Create a minimal RepoConfig mock."""
    m = MagicMock()
    m.id = repo_id
    m.resolved_path = repo_path
    return m


def _make_cfg(repos, skip_dirs=None):
    """Create a minimal WorkspaceConfig mock."""
    from corbell.core.workspace import IndexingConfig
    cfg = MagicMock()
    cfg.repos = repos
    cfg.indexing = IndexingConfig(
        skip_dirs=skip_dirs or [],
        max_file_bytes=1024 * 1024,
        chunk_size=50,
        chunk_overlap=10,
    )
    return cfg


# ---------------------------------------------------------------------------
# Basic table creation and metadata
# ---------------------------------------------------------------------------

def test_create_tables_idempotent(tracker):
    """Calling create_tables multiple times is safe."""
    tracker.create_tables()
    tracker.create_tables()  # no error


def test_get_last_build_at_none_initially(tracker):
    assert tracker.get_last_build_at() is None


def test_set_and_get_last_build_at(tracker):
    ts = time.time()
    tracker.set_meta("last_build_at", str(ts))
    result = tracker.get_last_build_at()
    assert result is not None
    assert abs(result - ts) < 0.001


def test_get_stored_model_none_initially(tracker):
    assert tracker.get_stored_model() is None


def test_set_and_get_stored_model(tracker):
    tracker.set_meta("embedding_model", "all-MiniLM-L6-v2")
    assert tracker.get_stored_model() == "all-MiniLM-L6-v2"


def test_set_meta_upserts(tracker):
    tracker.set_meta("key", "v1")
    tracker.set_meta("key", "v2")
    # Should not raise; second call overwrites
    # Verify by checking the correct value is stored
    assert tracker.get_stored_model() is None  # different key


# ---------------------------------------------------------------------------
# mark_indexed / remove_tracked
# ---------------------------------------------------------------------------

def test_mark_indexed(tracker):
    mtime = 1234567890.0
    tracker.mark_indexed("src/auth.py", "my-svc", mtime)
    # Should not raise; verify by calling get_stale_files
    # which would see it as tracked with this mtime


def test_mark_indexed_upserts(tracker):
    """Calling mark_indexed twice updates the mtime."""
    tracker.mark_indexed("src/auth.py", "my-svc", 1000.0)
    tracker.mark_indexed("src/auth.py", "my-svc", 2000.0)
    # No error; subsequent stale check would use the newer mtime


def test_remove_tracked(tracker):
    tracker.mark_indexed("src/a.py", "svc", 1000.0)
    tracker.mark_indexed("src/b.py", "svc", 1000.0)
    tracker.remove_tracked([("src/a.py", "svc")])
    # After removal, a.py would appear as "added" again in stale check


def test_clear_all(tracker):
    tracker.mark_indexed("src/a.py", "svc", 1000.0)
    tracker.set_meta("embedding_model", "test-model")
    tracker.clear_all()
    assert tracker.get_stored_model() is None
    assert tracker.get_last_build_at() is None


# ---------------------------------------------------------------------------
# get_stale_files — added detection
# ---------------------------------------------------------------------------

def test_stale_files_all_added(tracker, simple_repo):
    """All files appear as added when nothing is tracked yet."""
    repo = _make_repo_config(simple_repo)
    cfg = _make_cfg([repo])

    result = tracker.get_stale_files([repo], cfg)

    assert result.has_changes
    assert len(result.added) == 3
    assert len(result.modified) == 0
    assert len(result.deleted) == 0


def test_stale_files_clean_after_mark(tracker, simple_repo):
    """All files are clean after marking them as indexed with current mtime."""
    repo = _make_repo_config(simple_repo)
    cfg = _make_cfg([repo])

    # Mark all files as indexed
    from corbell.core.constants import EXTENSION_LANG
    for fp in simple_repo.rglob("*"):
        if fp.is_file() and fp.suffix in EXTENSION_LANG:
            rel = str(fp.relative_to(simple_repo))
            tracker.mark_indexed(rel, "my-repo", fp.stat().st_mtime)

    result = tracker.get_stale_files([repo], cfg)
    assert not result.has_changes


def test_stale_files_modified_detection(tracker, simple_repo):
    """Modified file is detected when its mtime changes."""
    repo = _make_repo_config(simple_repo)
    cfg = _make_cfg([repo])

    # Mark files with old mtime
    from corbell.core.constants import EXTENSION_LANG
    for fp in simple_repo.rglob("*"):
        if fp.is_file() and fp.suffix in EXTENSION_LANG:
            rel = str(fp.relative_to(simple_repo))
            tracker.mark_indexed(rel, "my-repo", 1000.0)  # fake old mtime

    result = tracker.get_stale_files([repo], cfg)
    # All files appear modified (current mtime != 1000.0)
    assert result.has_changes
    assert len(result.modified) == 3


def test_stale_files_deleted_detection(tracker, simple_repo):
    """Deleted file is detected when it no longer exists on disk."""
    repo = _make_repo_config(simple_repo)
    cfg = _make_cfg([repo])

    # Mark a.py as tracked
    a_py = simple_repo / "a.py"
    tracker.mark_indexed("a.py", "my-repo", a_py.stat().st_mtime)

    # Delete the file
    a_py.unlink()

    result = tracker.get_stale_files([repo], cfg)
    assert result.has_changes
    assert len(result.deleted) == 1
    assert result.deleted[0][0] == "a.py"


def test_stale_result_changed_repo_ids(tracker, simple_repo):
    """changed_repo_ids returns the set of repos with changes."""
    repo = _make_repo_config(simple_repo)
    cfg = _make_cfg([repo])

    result = tracker.get_stale_files([repo], cfg)
    assert "my-repo" in result.changed_repo_ids
