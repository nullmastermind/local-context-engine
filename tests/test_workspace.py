"""Tests for core/workspace.py — env-var driven config, path helpers."""

from __future__ import annotations

from pathlib import Path

from corbell.core.workspace import (
    build_config,
    db_path_for_workspace,
    sanitize_path,
)


# ---------------------------------------------------------------------------
# sanitize_path
# ---------------------------------------------------------------------------

def test_sanitize_path_linux_style(tmp_path):
    """Linux absolute path is sanitized correctly."""
    # Construct a fake Linux-style path string and test normalization
    p = Path("/home/user/projects/my-app")
    result = sanitize_path(p)
    # Leading slash becomes a leading dash which is then stripped
    assert "home" in result
    assert "user" in result
    assert "projects" in result
    assert "my-app" in result
    assert not result.startswith("-")


def test_sanitize_path_trailing_separator(tmp_path):
    """Trailing separator is stripped before sanitization."""
    p = tmp_path / "my-app"
    p.mkdir()
    result_no_sep = sanitize_path(p)
    # Even if the path string has trailing slash, resolved() removes it
    result_with_sep = sanitize_path(Path(str(p) + "/"))
    assert result_no_sep == result_with_sep


def test_sanitize_path_no_leading_dash(tmp_path):
    """Result never starts with a dash."""
    result = sanitize_path(tmp_path)
    assert not result.startswith("-")


def test_sanitize_path_replaces_separators(tmp_path):
    """Path separators are replaced with dashes."""
    result = sanitize_path(tmp_path)
    assert "/" not in result
    assert "\\" not in result
    assert ":" not in result


def test_sanitize_path_distinct_paths_produce_distinct_names(tmp_path):
    """Two different paths produce different sanitized names."""
    p1 = tmp_path / "repo-a"
    p1.mkdir()
    p2 = tmp_path / "repo-b"
    p2.mkdir()
    assert sanitize_path(p1) != sanitize_path(p2)


def test_sanitize_path_consistent_for_same_path(tmp_path):
    """Same physical directory always produces the same sanitized name."""
    p = tmp_path / "my-project"
    p.mkdir()
    assert sanitize_path(p) == sanitize_path(p)


# ---------------------------------------------------------------------------
# db_path_for_workspace
# ---------------------------------------------------------------------------

def test_db_path_for_workspace_location(tmp_path, monkeypatch):
    """DB is placed under ~/.vibervn/context-engine/{sanitized}/workspace.db."""
    fake_home = tmp_path / "fake-home"
    fake_home.mkdir()
    monkeypatch.setattr(Path, "home", staticmethod(lambda: fake_home))

    workspace = tmp_path / "my-project"
    workspace.mkdir()

    db = db_path_for_workspace(workspace)
    assert db.name == "workspace.db"
    assert db.parent.parent.name == "context-engine"
    assert db.parent.parent.parent.name == ".vibervn"
    # The workspace-named dir should contain the sanitized path segment
    assert sanitize_path(workspace) == db.parent.name


def test_db_path_creates_parent_dirs(tmp_path, monkeypatch):
    """Parent directories are created automatically."""
    fake_home = tmp_path / "fake-home"
    fake_home.mkdir()
    monkeypatch.setattr(Path, "home", staticmethod(lambda: fake_home))

    workspace = tmp_path / "new-project"
    workspace.mkdir()

    db = db_path_for_workspace(workspace)
    assert db.parent.exists()


def test_db_path_idempotent(tmp_path, monkeypatch):
    """Calling db_path_for_workspace twice returns the same path."""
    fake_home = tmp_path / "fake-home"
    fake_home.mkdir()
    monkeypatch.setattr(Path, "home", staticmethod(lambda: fake_home))

    workspace = tmp_path / "my-project"
    workspace.mkdir()

    assert db_path_for_workspace(workspace) == db_path_for_workspace(workspace)


# ---------------------------------------------------------------------------
# build_config — defaults
# ---------------------------------------------------------------------------

def _clear_corbell_env(monkeypatch):
    """Remove all CORBELL_* env vars that might affect config."""
    for var in (
        "CORBELL_TOP_K", "CORBELL_CHUNK_SIZE", "CORBELL_CHUNK_OVERLAP",
        "CORBELL_EXPAND_CALL_DEPTH", "CORBELL_EXPAND_MAX_CHUNKS",
        "CORBELL_RERANK", "CORBELL_EMBEDDING_MODEL", "CORBELL_MAX_FILE_BYTES",
        "CORBELL_SKIP_DIRS", "CORBELL_LLM_MODEL",
    ):
        monkeypatch.delenv(var, raising=False)


def test_build_config_defaults(tmp_path, monkeypatch):
    """build_config() uses spec-defined defaults when no env vars set."""
    _clear_corbell_env(monkeypatch)
    workspace = tmp_path / "my-project"
    workspace.mkdir()

    cfg = build_config(workspace)

    assert cfg.query.top_k == 50
    assert cfg.indexing.chunk_size == 50
    assert cfg.indexing.chunk_overlap == 10
    assert cfg.query.expand_call_depth == 2
    assert cfg.query.expand_max_chunks == 30
    assert cfg.query.rerank is True
    assert cfg.storage.model == "all-MiniLM-L6-v2"
    assert cfg.indexing.max_file_bytes == 1048576
    assert cfg.indexing.skip_dirs == []


def test_build_config_single_repo(tmp_path, monkeypatch):
    """build_config() creates exactly one repo with workspace_path as root."""
    _clear_corbell_env(monkeypatch)
    workspace = tmp_path / "my-project"
    workspace.mkdir()

    cfg = build_config(workspace)

    assert len(cfg.repos) == 1
    repo = cfg.repos[0]
    assert repo.id == "my-project"
    assert repo.resolved_path == workspace


def test_build_config_repo_id_is_basename(tmp_path, monkeypatch):
    """Repo ID is the workspace directory basename."""
    _clear_corbell_env(monkeypatch)
    workspace = tmp_path / "awesome-service"
    workspace.mkdir()

    cfg = build_config(workspace)
    assert cfg.repos[0].id == "awesome-service"


# ---------------------------------------------------------------------------
# build_config — env var overrides
# ---------------------------------------------------------------------------

def test_build_config_top_k_override(tmp_path, monkeypatch):
    """CORBELL_TOP_K overrides default."""
    _clear_corbell_env(monkeypatch)
    monkeypatch.setenv("CORBELL_TOP_K", "20")
    workspace = tmp_path / "proj"
    workspace.mkdir()

    cfg = build_config(workspace)
    assert cfg.query.top_k == 20


def test_build_config_chunk_size_override(tmp_path, monkeypatch):
    """CORBELL_CHUNK_SIZE overrides default."""
    _clear_corbell_env(monkeypatch)
    monkeypatch.setenv("CORBELL_CHUNK_SIZE", "100")
    workspace = tmp_path / "proj"
    workspace.mkdir()

    cfg = build_config(workspace)
    assert cfg.indexing.chunk_size == 100


def test_build_config_rerank_false(tmp_path, monkeypatch):
    """CORBELL_RERANK=false disables reranking."""
    _clear_corbell_env(monkeypatch)
    monkeypatch.setenv("CORBELL_RERANK", "false")
    workspace = tmp_path / "proj"
    workspace.mkdir()

    cfg = build_config(workspace)
    assert cfg.query.rerank is False


def test_build_config_skip_dirs(tmp_path, monkeypatch):
    """CORBELL_SKIP_DIRS is parsed as comma-separated list."""
    _clear_corbell_env(monkeypatch)
    monkeypatch.setenv("CORBELL_SKIP_DIRS", "node_modules,dist,.git")
    workspace = tmp_path / "proj"
    workspace.mkdir()

    cfg = build_config(workspace)
    assert cfg.indexing.skip_dirs == ["node_modules", "dist", ".git"]


def test_build_config_embedding_model_override(tmp_path, monkeypatch):
    """CORBELL_EMBEDDING_MODEL overrides default model."""
    _clear_corbell_env(monkeypatch)
    monkeypatch.setenv("CORBELL_EMBEDDING_MODEL", "my-custom-model")
    workspace = tmp_path / "proj"
    workspace.mkdir()

    cfg = build_config(workspace)
    assert cfg.storage.resolved_model() == "my-custom-model"


def test_build_config_all_env_vars(tmp_path, monkeypatch):
    """All CORBELL_* env vars from spec are supported."""
    _clear_corbell_env(monkeypatch)
    monkeypatch.setenv("CORBELL_TOP_K", "30")
    monkeypatch.setenv("CORBELL_CHUNK_SIZE", "75")
    monkeypatch.setenv("CORBELL_CHUNK_OVERLAP", "15")
    monkeypatch.setenv("CORBELL_EXPAND_CALL_DEPTH", "3")
    monkeypatch.setenv("CORBELL_EXPAND_MAX_CHUNKS", "20")
    monkeypatch.setenv("CORBELL_RERANK", "false")
    monkeypatch.setenv("CORBELL_EMBEDDING_MODEL", "custom-model")
    monkeypatch.setenv("CORBELL_MAX_FILE_BYTES", "512000")
    monkeypatch.setenv("CORBELL_SKIP_DIRS", "dist,build")

    workspace = tmp_path / "proj"
    workspace.mkdir()

    cfg = build_config(workspace)
    assert cfg.query.top_k == 30
    assert cfg.indexing.chunk_size == 75
    assert cfg.indexing.chunk_overlap == 15
    assert cfg.query.expand_call_depth == 3
    assert cfg.query.expand_max_chunks == 20
    assert cfg.query.rerank is False
    assert cfg.storage.model == "custom-model"
    assert cfg.indexing.max_file_bytes == 512000
    assert cfg.indexing.skip_dirs == ["dist", "build"]


# ---------------------------------------------------------------------------
# LLMConfig env var resolution (still valid after refactor)
# ---------------------------------------------------------------------------

def test_llm_resolved_api_key_anthropic(monkeypatch):
    """Anthropic API key resolved from env var."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-123")
    from corbell.core.workspace import LLMConfig
    cfg = LLMConfig(provider="anthropic")
    assert cfg.resolved_api_key() == "sk-test-123"


def test_llm_resolved_api_key_google(monkeypatch):
    """Google API key resolved from env var."""
    monkeypatch.setenv("GOOGLE_API_KEY", "sk-google-test")
    from corbell.core.workspace import LLMConfig
    cfg = LLMConfig(provider="google")
    assert cfg.resolved_api_key() == "sk-google-test"
