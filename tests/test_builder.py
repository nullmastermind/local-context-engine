"""Tests for IndexBuilder — full and incremental builds."""

from __future__ import annotations

import textwrap
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from corbell.core.indexing.builder import IndexBuilder
from corbell.core.indexing.tracker import IndexTracker
from corbell.core.embeddings.sqlite_store import SQLiteEmbeddingStore


@pytest.fixture
def repo(tmp_path) -> Path:
    """Create a simple Python repo for testing."""
    r = tmp_path / "my-repo"
    r.mkdir()
    (r / "module_a.py").write_text(textwrap.dedent("""\
        def func_a():
            return "hello"
    """))
    (r / "module_b.py").write_text(textwrap.dedent("""\
        def func_b():
            return "world"
    """))
    return r


@pytest.fixture
def workspace_config(repo, monkeypatch, tmp_path):
    """Build a WorkspaceConfig for the test repo, with embedding model set to 'voyage-test'."""
    # Clear CORBELL_* env vars to get clean defaults
    for var in (
        "CORBELL_TOP_K", "CORBELL_CHUNK_SIZE", "CORBELL_CHUNK_OVERLAP",
        "CORBELL_EXPAND_CALL_DEPTH", "CORBELL_EXPAND_MAX_CHUNKS",
        "CORBELL_RERANK", "CORBELL_EMBEDDING_MODEL", "CORBELL_MAX_FILE_BYTES",
        "CORBELL_SKIP_DIRS", "CORBELL_LLM_MODEL",
    ):
        monkeypatch.delenv(var, raising=False)

    monkeypatch.setenv("CORBELL_EMBEDDING_MODEL", "voyage-test")
    monkeypatch.setenv("CORBELL_TOP_K", "10")

    from corbell.core.workspace import build_config
    return build_config(repo)


@pytest.fixture
def db_path(repo, tmp_path, monkeypatch) -> Path:
    """Return a temporary db path for the test repo."""
    fake_home = tmp_path / "fake-home"
    fake_home.mkdir()
    monkeypatch.setattr(Path, "home", staticmethod(lambda: fake_home))

    from corbell.core.workspace import db_path_for_workspace
    return db_path_for_workspace(repo)


def _make_mock_model():
    """Return a mock embedding model that returns unit vectors."""
    m = MagicMock()
    m.encode.side_effect = lambda texts: [[1.0 / len(texts)] * 384 for _ in texts]
    return m


# ---------------------------------------------------------------------------
# Full build
# ---------------------------------------------------------------------------

def test_full_build_indexes_all_files(workspace_config, db_path):
    """Full build indexes all files in the repo."""
    mock_model = _make_mock_model()

    with patch("corbell.core.indexing.builder.VoyageEmbeddingModel", return_value=mock_model):
        builder = IndexBuilder()
        result = builder.build(workspace_config, db_path, rebuild=True)

    assert result["status"] == "full_build"
    assert result["chunks_added"] > 0

    # Check that embeddings were stored
    emb_store = SQLiteEmbeddingStore(db_path)
    assert emb_store.count() > 0


def test_full_build_stores_metadata(workspace_config, db_path):
    """Full build updates index_meta with model and timestamp."""
    mock_model = _make_mock_model()

    before = time.time()
    with patch("corbell.core.indexing.builder.VoyageEmbeddingModel", return_value=mock_model):
        builder = IndexBuilder()
        builder.build(workspace_config, db_path, rebuild=True)

    tracker = IndexTracker(db_path)
    stored_model = tracker.get_stored_model()
    last_build = tracker.get_last_build_at()

    assert stored_model == "voyage-test"
    assert last_build is not None
    assert last_build >= before


# ---------------------------------------------------------------------------
# Incremental build
# ---------------------------------------------------------------------------

def test_incremental_only_reindexes_changed_files(workspace_config, db_path, repo):
    """Incremental build only processes files that changed."""
    mock_model = _make_mock_model()

    # First: full build
    with patch("corbell.core.indexing.builder.VoyageEmbeddingModel", return_value=mock_model):
        builder = IndexBuilder()
        builder.build(workspace_config, db_path, rebuild=True)

    initial_encode_calls = mock_model.encode.call_count

    # Modify one file
    (repo / "module_a.py").write_text(textwrap.dedent("""\
        def func_a():
            return "hello updated"
        def func_a2():
            return "new function"
    """))

    # Second: incremental
    with patch("corbell.core.indexing.builder.VoyageEmbeddingModel", return_value=mock_model):
        builder2 = IndexBuilder()
        result = builder2.build(workspace_config, db_path, rebuild=False)

    assert result["status"] == "incremental"
    assert result["files_modified"] == 1
    # encode was called again for the changed file
    assert mock_model.encode.call_count > initial_encode_calls


def test_incremental_detects_deleted_files(workspace_config, db_path, repo):
    """Incremental build removes chunks for deleted files."""
    mock_model = _make_mock_model()

    # Full build first
    with patch("corbell.core.indexing.builder.VoyageEmbeddingModel", return_value=mock_model):
        builder = IndexBuilder()
        builder.build(workspace_config, db_path, rebuild=True)

    count_before = SQLiteEmbeddingStore(db_path).count()

    # Delete a file
    (repo / "module_b.py").unlink()

    # Incremental rebuild
    with patch("corbell.core.indexing.builder.VoyageEmbeddingModel", return_value=mock_model):
        builder2 = IndexBuilder()
        result = builder2.build(workspace_config, db_path, rebuild=False)

    assert result["files_deleted"] == 1
    count_after = SQLiteEmbeddingStore(db_path).count()
    assert count_after < count_before


# ---------------------------------------------------------------------------
# Model safety check
# ---------------------------------------------------------------------------

def test_model_mismatch_raises_error(workspace_config, db_path, monkeypatch):
    """Incremental build fails if the stored model doesn't match the current config."""
    mock_model = _make_mock_model()

    # Full build with "voyage-test"
    with patch("corbell.core.indexing.builder.VoyageEmbeddingModel", return_value=mock_model):
        builder = IndexBuilder()
        builder.build(workspace_config, db_path, rebuild=True)

    # Simulate config change to different model
    workspace_config.storage.model = "voyage-different"

    with pytest.raises(ValueError, match="Model changed"):
        with patch("corbell.core.indexing.builder.VoyageEmbeddingModel", return_value=mock_model):
            builder2 = IndexBuilder()
            builder2.build(workspace_config, db_path, rebuild=False)


# ---------------------------------------------------------------------------
# repo_filter
# ---------------------------------------------------------------------------

def test_repo_filter_unknown_raises(workspace_config, db_path):
    """Passing an unknown repo_filter raises ValueError."""
    mock_model = _make_mock_model()

    with pytest.raises(ValueError, match="not found in workspace config"):
        with patch("corbell.core.indexing.builder.VoyageEmbeddingModel", return_value=mock_model):
            builder = IndexBuilder()
            builder.build(workspace_config, db_path, rebuild=True, repo_filter="nonexistent")
