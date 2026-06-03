"""Integration tests for the full query engine pipeline."""

from __future__ import annotations

import textwrap
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from corbell.core.embeddings.sqlite_store import SQLiteEmbeddingStore
from corbell.core.embeddings.extractor import EmbeddingRecord


@pytest.fixture
def workspace_with_indexed_repo(tmp_path):
    """Create a workspace with a repo and pre-indexed embeddings."""
    repo = tmp_path / "my-repo"
    repo.mkdir()
    (repo / "auth.py").write_text(textwrap.dedent("""\
        def authenticate(user, password):
            if user == "admin":
                return True
            return False

        def get_token(user):
            return f"token-{user}"
    """))

    ws_dir = tmp_path / "corbell"
    ws_dir.mkdir()
    ws_yaml = ws_dir / "workspace.yaml"
    ws_yaml.write_text(f"""\
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
  rerank: false
indexing:
  chunk_size: 50
  chunk_overlap: 10
""")

    from corbell.core.workspace import load_workspace
    cfg = load_workspace(ws_yaml)
    db_path = cfg.db_path(ws_dir)

    # Pre-index with fake embeddings
    emb_store = SQLiteEmbeddingStore(db_path)

    def _make_vec(seed: int) -> list:
        rng = np.random.RandomState(seed)
        v = rng.randn(384).astype(np.float32)
        return (v / np.linalg.norm(v)).tolist()

    record = EmbeddingRecord(
        id="my-repo::auth.py::authenticate",
        service_id="my-repo",
        repo=str(repo),
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
        repo=str(repo),
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
    import time
    tracker = IndexTracker(db_path)
    tracker.set_meta("embedding_model", "test-model")
    tracker.set_meta("last_build_at", str(time.time()))
    auth_mtime = (repo / "auth.py").stat().st_mtime
    tracker.mark_indexed("auth.py", "my-repo", auth_mtime)

    return ws_yaml, repo, db_path


# ---------------------------------------------------------------------------
# Basic pipeline test
# ---------------------------------------------------------------------------

def test_codebase_retrieval_returns_formatted_output(workspace_with_indexed_repo):
    """Full pipeline returns formatted code with file path and line numbers."""
    ws_yaml, repo, db_path = workspace_with_indexed_repo

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
            workspace_path=ws_yaml,
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


def test_codebase_retrieval_empty_index(tmp_path):
    """Returns error string when index is empty."""
    ws_dir = tmp_path / "corbell"
    ws_dir.mkdir()
    ws = ws_dir / "workspace.yaml"
    ws.write_text("""\
version: "1"
workspace:
  name: test
repos: []
storage:
  path: .corbell/test.db
  model: test-model
""")

    from corbell.core.query.engine import codebase_retrieval
    result = codebase_retrieval("anything", ws)

    assert "No index found" in result or result.startswith("Error:")


def test_codebase_retrieval_missing_workspace(tmp_path):
    """Returns error string when workspace.yaml doesn't exist."""
    from corbell.core.query.engine import codebase_retrieval
    result = codebase_retrieval("query", tmp_path / "nonexistent.yaml")

    assert result.startswith("Error:") or "not found" in result.lower()


def test_codebase_retrieval_no_llm(workspace_with_indexed_repo):
    """Pipeline works without LLM (use_llm=False)."""
    ws_yaml, repo, db_path = workspace_with_indexed_repo

    mock_model = MagicMock()
    rng = np.random.RandomState(10)
    q = rng.randn(384).astype(np.float32)
    mock_model.encode.return_value = [(q / np.linalg.norm(q)).tolist()]

    with patch("corbell.core.query.engine.SentenceTransformerModel", return_value=mock_model):
        from corbell.core.query.engine import codebase_retrieval
        result = codebase_retrieval("token", ws_yaml, use_llm=False, rerank=False)

    assert isinstance(result, str)
    assert not result.startswith("Error:")
