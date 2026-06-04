"""Tests for core/workspace.py — env-var driven config, path helpers."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from corbell.core.workspace import (
    build_config,
    db_path_for_workspace,
    detect_git_branch,
    resolve_embedding_dimension,
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
    """DB is placed under ~/.vibervn/context-engine/{sanitized}/{namespace}/workspace.db."""
    fake_home = tmp_path / "fake-home"
    fake_home.mkdir()
    monkeypatch.setattr(Path, "home", staticmethod(lambda: fake_home))

    workspace = tmp_path / "my-project"
    workspace.mkdir()

    with patch("corbell.core.workspace.detect_git_branch", return_value="main"):
        db = db_path_for_workspace(workspace)

    assert db.name == "workspace.db"
    # namespace dir: all-MiniLM-L6-v2--384--main
    assert db.parent.name == "all-MiniLM-L6-v2--384--main"
    assert db.parent.parent.name == sanitize_path(workspace)
    assert db.parent.parent.parent.name == "context-engine"
    assert db.parent.parent.parent.parent.name == ".vibervn"


def test_db_path_creates_parent_dirs(tmp_path, monkeypatch):
    """Parent directories are created automatically."""
    fake_home = tmp_path / "fake-home"
    fake_home.mkdir()
    monkeypatch.setattr(Path, "home", staticmethod(lambda: fake_home))

    workspace = tmp_path / "new-project"
    workspace.mkdir()

    with patch("corbell.core.workspace.detect_git_branch", return_value="main"):
        db = db_path_for_workspace(workspace)
    assert db.parent.exists()


def test_db_path_idempotent(tmp_path, monkeypatch):
    """Calling db_path_for_workspace twice returns the same path."""
    fake_home = tmp_path / "fake-home"
    fake_home.mkdir()
    monkeypatch.setattr(Path, "home", staticmethod(lambda: fake_home))

    workspace = tmp_path / "my-project"
    workspace.mkdir()

    with patch("corbell.core.workspace.detect_git_branch", return_value="main"):
        path1 = db_path_for_workspace(workspace)
        path2 = db_path_for_workspace(workspace)
    assert path1 == path2


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


# ---------------------------------------------------------------------------
# resolve_embedding_dimension
# ---------------------------------------------------------------------------

def test_resolve_embedding_dimension_defaults():
    """Cloud models return their expected dimensions."""
    assert resolve_embedding_dimension("voyage-code-3") == 1024
    assert resolve_embedding_dimension("voyage-4-lite") == 1024
    assert resolve_embedding_dimension("gemini-embedding-001") == 768


def test_resolve_embedding_dimension_voyage():
    """voyage-* models return 1024."""
    assert resolve_embedding_dimension("voyage-code-2") == 1024
    assert resolve_embedding_dimension("voyage-large-2") == 1024


def test_resolve_embedding_dimension_gemini():
    """gemini-* models return 768."""
    assert resolve_embedding_dimension("gemini-embedding-001") == 768


def test_resolve_embedding_dimension_unknown_fallback():
    """Unknown model names fall back to 1024 (Voyage default)."""
    assert resolve_embedding_dimension("some-unknown-model") == 1024


def test_resolve_embedding_dimension_env_override(monkeypatch):
    """CORBELL_EMBEDDING_DIM env var overrides the lookup."""
    monkeypatch.setenv("CORBELL_EMBEDDING_DIM", "512")
    assert resolve_embedding_dimension("voyage-code-3") == 512


# ---------------------------------------------------------------------------
# detect_git_branch
# ---------------------------------------------------------------------------

def test_detect_git_branch_detached(tmp_path):
    """Detached HEAD returns 'detached-<short-sha>'."""
    def mock_run(args, **kwargs):
        class Result:
            pass

        r = Result()
        if "--abbrev-ref" in args:
            r.returncode = 0
            r.stdout = "HEAD\n"
        else:
            r.returncode = 0
            r.stdout = "abc1234\n"
        return r

    with patch("corbell.core.workspace.subprocess.run", side_effect=mock_run):
        branch = detect_git_branch(tmp_path)

    assert branch == "detached-abc1234"


def test_detect_git_branch_no_git(tmp_path):
    """FileNotFoundError from subprocess (git not installed) returns '_no_git'."""
    with patch("corbell.core.workspace.subprocess.run", side_effect=FileNotFoundError):
        branch = detect_git_branch(tmp_path)

    assert branch == "_no_git"


def test_detect_git_branch_normal(tmp_path):
    """Normal branch name is returned as-is."""
    def mock_run(args, **kwargs):
        class Result:
            returncode = 0
            stdout = "feature/my-branch\n"

        return Result()

    with patch("corbell.core.workspace.subprocess.run", side_effect=mock_run):
        branch = detect_git_branch(tmp_path)

    assert branch == "feature/my-branch"


# ---------------------------------------------------------------------------
# db_path namespacing — model, dimension, branch
# ---------------------------------------------------------------------------

def test_db_path_different_model(tmp_path, monkeypatch):
    """Different model param produces a different DB path."""
    fake_home = tmp_path / "fake-home"
    fake_home.mkdir()
    monkeypatch.setattr(Path, "home", staticmethod(lambda: fake_home))

    workspace = tmp_path / "proj"
    workspace.mkdir()

    with patch("corbell.core.workspace.detect_git_branch", return_value="main"):
        path_a = db_path_for_workspace(workspace, model="all-MiniLM-L6-v2")
        path_b = db_path_for_workspace(workspace, model="all-mpnet-base-v2")

    assert path_a != path_b
    assert "all-MiniLM-L6-v2" in str(path_a)
    assert "all-mpnet-base-v2" in str(path_b)


def test_db_path_different_branch(tmp_path, monkeypatch):
    """Different git branch produces a different DB path."""
    fake_home = tmp_path / "fake-home"
    fake_home.mkdir()
    monkeypatch.setattr(Path, "home", staticmethod(lambda: fake_home))

    workspace = tmp_path / "proj"
    workspace.mkdir()

    with patch("corbell.core.workspace.detect_git_branch", return_value="main"):
        path_main = db_path_for_workspace(workspace)

    with patch("corbell.core.workspace.detect_git_branch", return_value="feature-x"):
        path_feature = db_path_for_workspace(workspace)

    assert path_main != path_feature
    assert "main" in str(path_main)
    assert "feature-x" in str(path_feature)


def test_db_path_different_dimension(tmp_path, monkeypatch):
    """CORBELL_EMBEDDING_DIM env var changes the namespace."""
    fake_home = tmp_path / "fake-home"
    fake_home.mkdir()
    monkeypatch.setattr(Path, "home", staticmethod(lambda: fake_home))

    workspace = tmp_path / "proj"
    workspace.mkdir()

    with patch("corbell.core.workspace.detect_git_branch", return_value="main"):
        path_default = db_path_for_workspace(workspace, model="all-MiniLM-L6-v2")

    monkeypatch.setenv("CORBELL_EMBEDDING_DIM", "512")
    with patch("corbell.core.workspace.detect_git_branch", return_value="main"):
        path_custom_dim = db_path_for_workspace(workspace, model="all-MiniLM-L6-v2")

    assert path_default != path_custom_dim
    assert "--512--" in str(path_custom_dim)
    assert "--384--" in str(path_default)


# ---------------------------------------------------------------------------
# _seed_from_sibling
# ---------------------------------------------------------------------------

def test_seed_from_sibling(tmp_path, monkeypatch):
    """A new branch namespace is seeded from the most-recent sibling DB."""
    fake_home = tmp_path / "fake-home"
    fake_home.mkdir()
    monkeypatch.setattr(Path, "home", staticmethod(lambda: fake_home))

    workspace = tmp_path / "proj"
    workspace.mkdir()

    # Build a sibling DB for "main"
    with patch("corbell.core.workspace.detect_git_branch", return_value="main"):
        main_db = db_path_for_workspace(workspace, model="all-MiniLM-L6-v2")

    # Write some content so copy makes sense
    main_db.write_bytes(b"SQLITE3 fake content for main")

    # Now request a new branch — it should be seeded from main
    with patch("corbell.core.workspace.detect_git_branch", return_value="new-feature"):
        feature_db = db_path_for_workspace(workspace, model="all-MiniLM-L6-v2")

    assert feature_db.exists()
    assert feature_db.read_bytes() == b"SQLITE3 fake content for main"
    # Paths are different
    assert main_db != feature_db


def test_seed_atomic_no_partial(tmp_path, monkeypatch):
    """If copy fails mid-write, no workspace.db is left behind."""
    from corbell.core.workspace import _seed_from_sibling

    base_dir = tmp_path / "base"
    base_dir.mkdir()

    # Create a sibling DB
    sibling_dir = base_dir / "all-MiniLM-L6-v2--384--main"
    sibling_dir.mkdir()
    sibling_db = sibling_dir / "workspace.db"
    sibling_db.write_bytes(b"content")

    target_namespace = "all-MiniLM-L6-v2--384--feature"

    # Patch shutil.copy2 to raise an exception mid-copy
    with patch("corbell.core.workspace.shutil.copy2", side_effect=OSError("disk full")):
        _seed_from_sibling(base_dir, target_namespace, "all-MiniLM-L6-v2--384")

    target_db = base_dir / target_namespace / "workspace.db"
    assert not target_db.exists()
