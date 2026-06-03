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
def workspace_yaml(tmp_path, repo) -> Path:
    ws_dir = tmp_path / "corbell"
    ws_dir.mkdir()
    ws = ws_dir / "workspace.yaml"
    ws.write_text(f"""\
version: "1"
workspace:
  name: test
repos:
  - id: my-repo
    path: {repo}
    language: python
storage:
  path: .corbell/test.db
  model: test-model
query:
  top_k: 10
indexing:
  chunk_size: 50
  chunk_overlap: 10
  max_file_bytes: 1048576
llm:
  provider: anthropic
""")
    return ws


def _make_mock_model():
    """Return a mock embedding model that returns unit vectors."""
    m = MagicMock()
    m.encode.side_effect = lambda texts: [[1.0 / len(texts)] * 384 for _ in texts]
    return m


# ---------------------------------------------------------------------------
# Full build
# ---------------------------------------------------------------------------

def test_full_build_indexes_all_files(workspace_yaml, tmp_path):
    """Full build indexes all files in the repo."""
    from corbell.core.workspace import load_workspace

    cfg = load_workspace(workspace_yaml)
    config_dir = workspace_yaml.parent

    db_path = cfg.db_path(config_dir)
    mock_model = _make_mock_model()

    with patch("corbell.core.indexing.builder.SentenceTransformerModel", return_value=mock_model):
        builder = IndexBuilder()
        result = builder.build(cfg, config_dir, rebuild=True)

    assert result["status"] == "full_build"
    assert result["chunks_added"] > 0

    # Check that embeddings were stored
    emb_store = SQLiteEmbeddingStore(db_path)
    assert emb_store.count() > 0


def test_full_build_stores_metadata(workspace_yaml):
    """Full build updates index_meta with model and timestamp."""
    from corbell.core.workspace import load_workspace

    cfg = load_workspace(workspace_yaml)
    config_dir = workspace_yaml.parent
    db_path = cfg.db_path(config_dir)

    mock_model = _make_mock_model()

    before = time.time()
    with patch("corbell.core.indexing.builder.SentenceTransformerModel", return_value=mock_model):
        builder = IndexBuilder()
        builder.build(cfg, config_dir, rebuild=True)

    tracker = IndexTracker(db_path)
    stored_model = tracker.get_stored_model()
    last_build = tracker.get_last_build_at()

    assert stored_model == "test-model"
    assert last_build is not None
    assert last_build >= before


# ---------------------------------------------------------------------------
# Incremental build
# ---------------------------------------------------------------------------

def test_incremental_only_reindexes_changed_files(workspace_yaml, repo):
    """Incremental build only processes files that changed."""
    from corbell.core.workspace import load_workspace

    cfg = load_workspace(workspace_yaml)
    config_dir = workspace_yaml.parent
    cfg.db_path(config_dir)  # ensure .corbell directory is created

    mock_model = _make_mock_model()

    # First: full build
    with patch("corbell.core.indexing.builder.SentenceTransformerModel", return_value=mock_model):
        builder = IndexBuilder()
        builder.build(cfg, config_dir, rebuild=True)

    initial_encode_calls = mock_model.encode.call_count

    # Modify one file
    (repo / "module_a.py").write_text(textwrap.dedent("""\
        def func_a():
            return "hello updated"
        def func_a2():
            return "new function"
    """))

    # Second: incremental
    with patch("corbell.core.indexing.builder.SentenceTransformerModel", return_value=mock_model):
        builder2 = IndexBuilder()
        result = builder2.build(cfg, config_dir, rebuild=False)

    assert result["status"] == "incremental"
    assert result["files_modified"] == 1
    # encode was called again for the changed file
    assert mock_model.encode.call_count > initial_encode_calls


def test_incremental_detects_deleted_files(workspace_yaml, repo):
    """Incremental build removes chunks for deleted files."""
    from corbell.core.workspace import load_workspace

    cfg = load_workspace(workspace_yaml)
    config_dir = workspace_yaml.parent
    db_path = cfg.db_path(config_dir)
    mock_model = _make_mock_model()

    # Full build first
    with patch("corbell.core.indexing.builder.SentenceTransformerModel", return_value=mock_model):
        builder = IndexBuilder()
        builder.build(cfg, config_dir, rebuild=True)

    count_before = SQLiteEmbeddingStore(db_path).count()

    # Delete a file
    (repo / "module_b.py").unlink()

    # Incremental rebuild
    with patch("corbell.core.indexing.builder.SentenceTransformerModel", return_value=mock_model):
        builder2 = IndexBuilder()
        result = builder2.build(cfg, config_dir, rebuild=False)

    assert result["files_deleted"] == 1
    count_after = SQLiteEmbeddingStore(db_path).count()
    assert count_after < count_before


# ---------------------------------------------------------------------------
# Model safety check
# ---------------------------------------------------------------------------

def test_model_mismatch_raises_error(workspace_yaml, repo):
    """Incremental build fails if the stored model doesn't match the current config."""
    from corbell.core.workspace import load_workspace

    cfg = load_workspace(workspace_yaml)
    config_dir = workspace_yaml.parent
    mock_model = _make_mock_model()

    # Full build with model "test-model"
    with patch("corbell.core.indexing.builder.SentenceTransformerModel", return_value=mock_model):
        builder = IndexBuilder()
        builder.build(cfg, config_dir, rebuild=True)

    # Simulate config change to different model
    cfg2 = load_workspace(workspace_yaml)
    cfg2.storage.model = "different-model"

    with pytest.raises(ValueError, match="Model changed"):
        with patch("corbell.core.indexing.builder.SentenceTransformerModel", return_value=mock_model):
            builder2 = IndexBuilder()
            builder2.build(cfg2, config_dir, rebuild=False)


# ---------------------------------------------------------------------------
# repo_filter
# ---------------------------------------------------------------------------

def test_repo_filter_unknown_raises(workspace_yaml):
    """Passing an unknown repo_filter raises ValueError."""
    from corbell.core.workspace import load_workspace

    cfg = load_workspace(workspace_yaml)
    config_dir = workspace_yaml.parent
    mock_model = _make_mock_model()

    with pytest.raises(ValueError, match="not found in workspace.yaml"):
        with patch("corbell.core.indexing.builder.SentenceTransformerModel", return_value=mock_model):
            builder = IndexBuilder()
            builder.build(cfg, config_dir, rebuild=True, repo_filter="nonexistent")
