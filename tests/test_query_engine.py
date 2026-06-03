"""Integration tests for the full query engine pipeline."""

from __future__ import annotations

import textwrap
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from corbell.core.embeddings.sqlite_store import SQLiteEmbeddingStore
from corbell.core.embeddings.extractor import EmbeddingRecord


@pytest.fixture
def workspace_with_indexed_repo(tmp_path, monkeypatch):
    """Create a workspace directory with a repo and pre-indexed embeddings."""
    # Use a fake home to isolate DB storage
    fake_home = tmp_path / "fake-home"
    fake_home.mkdir()
    monkeypatch.setattr(Path, "home", staticmethod(lambda: fake_home))

    # Clear CORBELL_* env vars
    for var in (
        "CORBELL_TOP_K", "CORBELL_CHUNK_SIZE", "CORBELL_CHUNK_OVERLAP",
        "CORBELL_EXPAND_CALL_DEPTH", "CORBELL_EXPAND_MAX_CHUNKS",
        "CORBELL_RERANK", "CORBELL_EMBEDDING_MODEL", "CORBELL_MAX_FILE_BYTES",
        "CORBELL_SKIP_DIRS", "CORBELL_LLM_MODEL",
    ):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("CORBELL_EMBEDDING_MODEL", "test-model")

    workspace = tmp_path / "my-repo"
    workspace.mkdir()
    (workspace / "auth.py").write_text(textwrap.dedent("""\
        def authenticate(user, password):
            if user == "admin":
                return True
            return False

        def get_token(user):
            return f"token-{user}"
    """))

    from corbell.core.workspace import db_path_for_workspace
    db_path = db_path_for_workspace(workspace)

    # Pre-index with fake embeddings
    emb_store = SQLiteEmbeddingStore(db_path)

    def _make_vec(seed: int) -> list:
        rng = np.random.RandomState(seed)
        v = rng.randn(384).astype(np.float32)
        return (v / np.linalg.norm(v)).tolist()

    record = EmbeddingRecord(
        id="my-repo::auth.py::authenticate",
        service_id="my-repo",
        repo=str(workspace),
        file_path="auth.py",
        start_line=1,
        end_line=5,
        content="def authenticate(user, password):\n    if user == 'admin':\n        return True\n    return False",
        language="python",
        chunk_type="function",
        symbol="authenticate",
        embedding=_make_vec(42),
    )
    emb_store.upsert(record)

    record2 = EmbeddingRecord(
        id="my-repo::auth.py::get_token",
        service_id="my-repo",
        repo=str(workspace),
        file_path="auth.py",
        start_line=7,
        end_line=8,
        content="def get_token(user):\n    return f'token-{user}'",
        language="python",
        chunk_type="function",
        symbol="get_token",
        embedding=_make_vec(99),
    )
    emb_store.upsert(record2)

    # Mark files as indexed
    from corbell.core.indexing.tracker import IndexTracker
    tracker = IndexTracker(db_path)
    tracker.set_meta("embedding_model", "test-model")
    tracker.set_meta("last_build_at", str(time.time()))
    auth_mtime = (workspace / "auth.py").stat().st_mtime
    tracker.mark_indexed("auth.py", "my-repo", auth_mtime)

    return workspace, db_path


# ---------------------------------------------------------------------------
# Basic pipeline test
# ---------------------------------------------------------------------------

def test_codebase_retrieval_returns_formatted_output(workspace_with_indexed_repo):
    """Full pipeline returns formatted code with file path and line numbers."""
    workspace, db_path = workspace_with_indexed_repo

    # Mock the embedding model to return a vector similar to the authenticate chunk
    rng = np.random.RandomState(42)
    query_vec = rng.randn(384).astype(np.float32)
    query_vec = (query_vec / np.linalg.norm(query_vec)).tolist()

    mock_model = MagicMock()
    mock_model.encode.return_value = [query_vec]

    with patch("corbell.core.query.engine.SentenceTransformerModel", return_value=mock_model):
        from corbell.core.query.engine import codebase_retrieval
        result = codebase_retrieval(
            query="authentication",
            workspace_path=workspace,
            top_k=10,
            use_llm=False,
            rerank=False,
        )

    assert isinstance(result, str)
    assert len(result) > 0
    # Should NOT be an error
    assert not result.startswith("Error:")
    assert not result.startswith("No index")
    # Should contain the file path
    assert "auth.py" in result
    # Should contain the #L notation
    assert "#L" in result


def test_codebase_retrieval_empty_index(tmp_path, monkeypatch):
    """Returns error string when index is empty."""
    fake_home = tmp_path / "fake-home"
    fake_home.mkdir()
    monkeypatch.setattr(Path, "home", staticmethod(lambda: fake_home))

    workspace = tmp_path / "empty-repo"
    workspace.mkdir()

    from corbell.core.query.engine import codebase_retrieval
    result = codebase_retrieval("anything", workspace)

    assert "No index found" in result or result.startswith("Error:")


def test_codebase_retrieval_missing_workspace(tmp_path):
    """Returns error string when workspace directory doesn't exist."""
    from corbell.core.query.engine import codebase_retrieval
    result = codebase_retrieval("query", tmp_path / "nonexistent-repo")

    assert result.startswith("Error:") or "not found" in result.lower()


def test_codebase_retrieval_no_llm(workspace_with_indexed_repo):
    """Pipeline works without LLM (use_llm=False)."""
    workspace, db_path = workspace_with_indexed_repo

    mock_model = MagicMock()
    rng = np.random.RandomState(10)
    q = rng.randn(384).astype(np.float32)
    mock_model.encode.return_value = [(q / np.linalg.norm(q)).tolist()]

    with patch("corbell.core.query.engine.SentenceTransformerModel", return_value=mock_model):
        from corbell.core.query.engine import codebase_retrieval
        result = codebase_retrieval(str(workspace), workspace, use_llm=False, rerank=False)

    assert isinstance(result, str)
    assert not result.startswith("Error:")
