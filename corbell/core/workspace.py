"""Workspace configuration for Corbell — env-var driven, no YAML required."""

from __future__ import annotations

import os
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

    model: str = "all-MiniLM-L6-v2"

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


def db_path_for_workspace(workspace_path: Path) -> Path:
    """Return the SQLite DB path for a workspace.

    Stored at ``~/.vibervn/context-engine/{sanitized}/workspace.db``.
    Creates parent directories automatically.
    """
    name = sanitize_path(workspace_path)
    db_dir = Path.home() / ".vibervn" / "context-engine" / name
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
    embedding_model = os.environ.get("CORBELL_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    max_file_bytes = int(os.environ.get("CORBELL_MAX_FILE_BYTES", str(1024 * 1024)))
    skip_dirs_str = os.environ.get("CORBELL_SKIP_DIRS", "")
    skip_dirs = [d.strip() for d in skip_dirs_str.split(",") if d.strip()] if skip_dirs_str else []
    llm_model = os.environ.get("CORBELL_LLM_MODEL", "claude-sonnet-4-5")

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
        llm=LLMConfig(model=llm_model),
    )
