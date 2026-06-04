"""Workspace configuration for Corbell — env-var driven, no YAML required."""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel, Field


class RepoConfig(BaseModel):
    """A single repository definition."""

    id: str
    path: str
    language: Optional[str] = None
    resolved_path: Optional[Path] = Field(default=None, exclude=True)

    model_config = {"extra": "ignore"}


class StorageConfig(BaseModel):
    """Storage sub-config (single SQLite file for both graph and embeddings)."""

    model: str = "voyage-code-3"

    model_config = {"extra": "ignore"}

    def resolved_model(self) -> str:
        """Return the effective embedding model name.

        Resolution order:
        1. ``CORBELL_EMBEDDING_MODEL`` env var (if set)
        2. ``model`` field default
        """
        return os.environ.get("CORBELL_EMBEDDING_MODEL") or self.model


class QueryConfig(BaseModel):
    """Query pipeline configuration."""

    top_k: int = 50
    expand_call_depth: int = 2
    expand_max_chunks: int = 30
    rerank: bool = True

    model_config = {"extra": "ignore"}


class IndexingConfig(BaseModel):
    """Indexing pipeline configuration."""

    skip_dirs: List[str] = Field(default_factory=list)
    max_file_bytes: int = 1024 * 1024  # 1 MB
    chunk_size: int = 50
    chunk_overlap: int = 10

    model_config = {"extra": "ignore"}


class LLMConfig(BaseModel):
    """LLM provider configuration.

    Local providers: openai, anthropic, ollama, google.
    Cloud providers: aws (Bedrock), azure (Azure OpenAI), gcp (Vertex AI).

    API key resolved via env vars:
    ANTHROPIC_API_KEY, OPENAI_API_KEY, AZURE_OPENAI_API_KEY, CORBELL_LLM_API_KEY

    Model can be overridden via env vars (checked in order):
    1. Provider-specific: ANTHROPIC_MODEL, OPENAI_MODEL, GOOGLE_MODEL, etc.
    2. Generic: CORBELL_LLM_MODEL
    3. ``model`` field default
    """

    provider: str = "anthropic"
    model: str = "claude-sonnet-4-5"
    api_key: Optional[str] = None

    # AWS Bedrock
    aws_region: Optional[str] = None

    # Azure OpenAI
    azure_endpoint: Optional[str] = None
    azure_deployment: Optional[str] = None
    azure_api_version: Optional[str] = None

    # GCP Vertex AI
    gcp_project: Optional[str] = None
    gcp_region: Optional[str] = None

    model_config = {"extra": "ignore"}

    def resolved_model(self) -> str:
        """Return the effective LLM model name.

        Resolution order:
        1. Provider-specific env var (e.g. ``ANTHROPIC_MODEL``, ``GOOGLE_MODEL``)
        2. ``CORBELL_LLM_MODEL`` env var
        3. ``model`` field default
        """
        provider_env_map = {
            "anthropic": "ANTHROPIC_MODEL",
            "openai": "OPENAI_MODEL",
            "google": "GOOGLE_MODEL",
            "ollama": "OLLAMA_MODEL",
            "aws": "AWS_MODEL",
            "azure": "AZURE_MODEL",
            "gcp": "GCP_MODEL",
        }
        provider_var = provider_env_map.get(self.provider.lower())
        if provider_var:
            val = os.environ.get(provider_var)
            if val:
                return val
        return os.environ.get("CORBELL_LLM_MODEL") or self.model

    def resolved_api_key(self) -> Optional[str]:
        """Return the API key from env vars."""
        # Cloud providers use their own credential chains (no API key needed)
        if self.provider in ("aws", "gcp"):
            return None
        # Fall back to well-known env vars
        env_map = {
            "openai": "OPENAI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
            "azure": "AZURE_OPENAI_API_KEY",
            "google": "GOOGLE_API_KEY",
            "ollama": None,
        }
        env_var = env_map.get(self.provider.lower(), "CORBELL_LLM_API_KEY")
        if env_var:
            return os.environ.get(env_var) or os.environ.get("CORBELL_LLM_API_KEY")
        return None


class WorkspaceConfig(BaseModel):
    """Root workspace configuration model (populated from env vars)."""

    version: str = "1"
    repos: List[RepoConfig] = Field(default_factory=list)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    query: QueryConfig = Field(default_factory=QueryConfig)
    indexing: IndexingConfig = Field(default_factory=IndexingConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)

    model_config = {"extra": "ignore"}


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------


def sanitize_path(workspace_path: Path) -> str:
    """Sanitize a workspace path for use as a filesystem directory name.

    Steps:
    1. Resolve to absolute path.
    2. Strip trailing separators.
    3. Replace ``/``, ``\\``, ``:`` with ``-``.
    4. Strip leading ``-`` characters.

    Examples:
        /home/user/projects/my-app  →  home-user-projects-my-app
        D:\\projects\\Python\\local-context-engine  →  D--projects-Python-local-context-engine
    """
    resolved = str(workspace_path.resolve())
    # Strip trailing path separators
    resolved = resolved.rstrip("/\\")
    # Replace path separators and Windows drive colon with dash
    sanitized = resolved.replace("\\", "-").replace("/", "-").replace(":", "-")
    # Strip leading dashes (e.g. from a leading / after replacement on Linux)
    sanitized = sanitized.lstrip("-")
    return sanitized


def resolve_embedding_dimension(model_name: str) -> int:
    """Return the embedding vector dimension for *model_name*.

    Resolution order:
    1. ``CORBELL_EMBEDDING_DIM`` env var (if set)
    2. Prefix-based rule for voyage-* and gemini-* models
    3. Exact lookup in ``KNOWN_DIMS``
    4. Default fallback of 384

    Never loads an actual model — pure lookup only.
    """
    dim_env = os.environ.get("CORBELL_EMBEDDING_DIM", "").strip()
    if dim_env:
        return int(dim_env)
    if model_name.startswith("voyage-"):
        return 1024
    if model_name.startswith("gemini-"):
        return 768
    return 1024


def detect_git_branch(workspace_path: Path) -> str:
    """Detect the current git branch for *workspace_path*.

    Returns the branch name, ``"detached-<short-sha>"`` for detached HEAD,
    or ``"_no_git"`` when git is unavailable or the directory is not a repo.

    Reads .git/HEAD directly to avoid subprocess overhead and timeout issues
    on Windows. Falls back to subprocess only for worktrees (.git is a file).
    """
    git_dir = workspace_path / ".git"

    # Standard repo: .git is a directory with HEAD file
    if git_dir.is_dir():
        head_file = git_dir / "HEAD"
        if head_file.exists():
            try:
                content = head_file.read_text(encoding="utf-8").strip()
                if content.startswith("ref: refs/heads/"):
                    return content[len("ref: refs/heads/"):]
                if content.startswith("ref: "):
                    return content[len("ref: "):]
                # Detached HEAD — content is a full SHA
                if len(content) >= 7:
                    return f"detached-{content[:7]}"
            except OSError:
                pass
        return "_no_git"

    # Worktree or submodule: .git is a file pointing elsewhere
    if git_dir.is_file():
        try:
            pointer = git_dir.read_text(encoding="utf-8").strip()
            if pointer.startswith("gitdir: "):
                real_git_dir = Path(pointer[len("gitdir: "):])
                if not real_git_dir.is_absolute():
                    real_git_dir = (workspace_path / real_git_dir).resolve()
                head_file = real_git_dir / "HEAD"
                if head_file.exists():
                    content = head_file.read_text(encoding="utf-8").strip()
                    if content.startswith("ref: refs/heads/"):
                        return content[len("ref: refs/heads/"):]
                    if content.startswith("ref: "):
                        return content[len("ref: "):]
                    if len(content) >= 7:
                        return f"detached-{content[:7]}"
        except OSError:
            pass

    return "_no_git"


def _seed_from_sibling(base_dir: Path, target_namespace: str, model_dim_prefix: str) -> None:
    """Copy the most-recently-modified sibling DB into *target_namespace* if it doesn't exist yet.

    Looks for sibling directories under *base_dir* that share the same
    ``model--dimension`` prefix but differ in branch name.  The most recent DB
    is copied atomically (temp file + os.replace) so a crash mid-copy never
    leaves a partial database file.

    Args:
        base_dir: Parent directory that contains per-namespace subdirectories.
        target_namespace: The namespace directory name to seed (``model--dim--branch``).
        model_dim_prefix: The ``model--dimension`` prefix used to identify siblings
            (e.g. ``"all-MiniLM-L6-v2--384"``).  Passed explicitly to avoid
            ambiguous parsing when branch names contain ``--``.
    """
    target_dir = base_dir / target_namespace
    target_db = target_dir / "workspace.db"
    if target_db.exists():
        return

    # Find sibling dirs with same model+dim but different branch
    candidates = []
    if base_dir.exists():
        for d in base_dir.iterdir():
            if not d.is_dir() or d.name == target_namespace:
                continue
            if not d.name.startswith(model_dim_prefix + "--"):
                continue
            db_file = d / "workspace.db"
            if db_file.exists():
                try:
                    candidates.append((db_file.stat().st_mtime, db_file))
                except OSError:
                    continue

    if not candidates:
        return

    candidates.sort(reverse=True)
    best_db = candidates[0][1]

    target_dir.mkdir(parents=True, exist_ok=True)

    # Atomic: write to temp file in same directory, then rename
    fd, tmp_path = tempfile.mkstemp(dir=str(target_dir), suffix=".db.tmp")
    try:
        os.close(fd)
        shutil.copy2(str(best_db), tmp_path)
        os.replace(tmp_path, str(target_db))
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def db_path_for_workspace(workspace_path: Path, model: Optional[str] = None) -> Path:
    """Return the SQLite DB path for a workspace, namespaced by model, dimension, and git branch.

    Stored at ``~/.vibervn/context-engine/{sanitized}/{model}--{dim}--{branch}/workspace.db``.
    Creates parent directories automatically.

    The namespace isolates index data so switching embedding models, changing
    vector dimensions, or checking out a different branch never corrupts an
    existing index.  When a new branch namespace is first used, the most recent
    sibling DB (same model+dim, different branch) is copied as a warm seed so
    incremental indexing can pick up where it left off.

    Args:
        workspace_path: Path to the workspace root directory.
        model: Embedding model name.  Falls back to ``CORBELL_EMBEDDING_MODEL``
               env var, then ``"voyage-code-3"``.
    """
    model_name = model or os.environ.get("CORBELL_EMBEDDING_MODEL") or "voyage-code-3"
    dimension = resolve_embedding_dimension(model_name)
    branch = detect_git_branch(workspace_path)

    sanitized_model = model_name.replace("/", "_").replace("\\", "_")
    sanitized_branch = branch.replace("/", "_").replace("\\", "_")
    model_dim_prefix = f"{sanitized_model}--{dimension}"
    namespace = f"{model_dim_prefix}--{sanitized_branch}"

    name = sanitize_path(workspace_path)
    base_dir = Path.home() / ".vibervn" / "context-engine" / name

    # Seed from sibling branch if this is a new namespace
    if not (base_dir / namespace / "workspace.db").exists():
        _seed_from_sibling(base_dir, namespace, model_dim_prefix)

    db_dir = base_dir / namespace
    db_dir.mkdir(parents=True, exist_ok=True)
    return db_dir / "workspace.db"


def _detect_language(path: Path) -> str:
    """Detect the most likely language of a project directory based on key files."""
    if (path / "package.json").exists() or (path / "tsconfig.json").exists():
        return "typescript"
    if (
        (path / "requirements.txt").exists()
        or (path / "pyproject.toml").exists()
        or (path / "Pipfile").exists()
        or (path / "setup.py").exists()
    ):
        return "python"
    if (path / "go.mod").exists():
        return "go"
    if (path / "pom.xml").exists() or (path / "build.gradle").exists():
        return "java"
    if (path / "Cargo.toml").exists():
        return "rust"
    return "python"


def build_config(workspace_path: Path) -> WorkspaceConfig:
    """Build a WorkspaceConfig from environment variables and a workspace path.

    Reads all ``CORBELL_*`` env vars with sensible defaults, then constructs
    a single RepoConfig from the workspace_path (id = basename, path = workspace_path).

    Args:
        workspace_path: Absolute path to the workspace (repository) root directory.

    Returns:
        Fully populated WorkspaceConfig ready for use by the indexer and query engine.
    """
    workspace_path = workspace_path.resolve()

    # Parse env vars
    top_k = int(os.environ.get("CORBELL_TOP_K", "50"))
    chunk_size = int(os.environ.get("CORBELL_CHUNK_SIZE", "50"))
    chunk_overlap = int(os.environ.get("CORBELL_CHUNK_OVERLAP", "10"))
    expand_call_depth = int(os.environ.get("CORBELL_EXPAND_CALL_DEPTH", "2"))
    expand_max_chunks = int(os.environ.get("CORBELL_EXPAND_MAX_CHUNKS", "30"))
    rerank_str = os.environ.get("CORBELL_RERANK", "true").lower()
    rerank = rerank_str not in ("false", "0", "no")
    embedding_model = os.environ.get("CORBELL_EMBEDDING_MODEL", "voyage-code-3")
    max_file_bytes = int(os.environ.get("CORBELL_MAX_FILE_BYTES", str(1024 * 1024)))
    skip_dirs_str = os.environ.get("CORBELL_SKIP_DIRS", "")
    skip_dirs = [d.strip() for d in skip_dirs_str.split(",") if d.strip()] if skip_dirs_str else []
    llm_model = os.environ.get("CORBELL_LLM_MODEL", "claude-sonnet-4-5")
    llm_provider = os.environ.get("CORBELL_LLM_PROVIDER", "anthropic")

    # Single repo: workspace root IS the repo
    repo_id = workspace_path.name
    language = _detect_language(workspace_path)
    repo = RepoConfig(
        id=repo_id,
        path=str(workspace_path),
        language=language,
        resolved_path=workspace_path,
    )

    return WorkspaceConfig(
        repos=[repo],
        storage=StorageConfig(model=embedding_model),
        query=QueryConfig(
            top_k=top_k,
            expand_call_depth=expand_call_depth,
            expand_max_chunks=expand_max_chunks,
            rerank=rerank,
        ),
        indexing=IndexingConfig(
            skip_dirs=skip_dirs,
            max_file_bytes=max_file_bytes,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        ),
        llm=LLMConfig(provider=llm_provider, model=llm_model),
    )
