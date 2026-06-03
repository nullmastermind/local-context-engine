"""Workspace configuration loader for Corbell."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from pydantic import BaseModel, Field


class RepoConfig(BaseModel):
    """A single repository definition in workspace.yaml."""

    id: str
    path: str
    language: Optional[str] = None
    resolved_path: Optional[Path] = Field(default=None, exclude=True)

    model_config = {"extra": "ignore"}


class StorageConfig(BaseModel):
    """Storage sub-config (single SQLite file for both graph and embeddings).

    The embedding model can be overridden via the ``CORBELL_EMBEDDING_MODEL``
    environment variable without touching workspace.yaml.
    """

    path: str = ".corbell/workspace.db"
    model: str = "all-MiniLM-L6-v2"

    model_config = {"extra": "ignore"}

    def resolved_model(self) -> str:
        """Return the effective embedding model name.

        Resolution order:
        1. ``CORBELL_EMBEDDING_MODEL`` env var (if set)
        2. ``model`` field from workspace.yaml
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

    API key can be provided here or via env vars:
    ANTHROPIC_API_KEY, OPENAI_API_KEY, AZURE_OPENAI_API_KEY, CORBELL_LLM_API_KEY

    Model can be overridden via env vars (checked in order):
    1. Provider-specific: ANTHROPIC_MODEL, OPENAI_MODEL, GOOGLE_MODEL, etc.
    2. Generic: CORBELL_LLM_MODEL
    3. ``model`` field from workspace.yaml
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
        3. ``model`` field from workspace.yaml
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
        """Return the API key, resolving env var placeholders if needed."""
        key = self.api_key or ""
        if key.startswith("${") and key.endswith("}"):
            var = key[2:-1]
            return os.environ.get(var)
        if key:
            return key
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


class WorkspaceInfo(BaseModel):
    """Top-level workspace metadata."""

    name: str = "my-platform"
    root: str = ".."

    model_config = {"extra": "ignore"}


class WorkspaceConfig(BaseModel):
    """Root workspace configuration model (parsed from workspace.yaml)."""

    version: str = "1"
    workspace: WorkspaceInfo = Field(default_factory=WorkspaceInfo)
    repos: List[RepoConfig] = Field(default_factory=list)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    query: QueryConfig = Field(default_factory=QueryConfig)
    indexing: IndexingConfig = Field(default_factory=IndexingConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)

    # Internal: path this config was loaded from
    _config_path: Optional[Path] = None

    model_config = {"extra": "ignore"}

    def resolve_paths(self, config_dir: Path) -> "WorkspaceConfig":
        """Resolve relative repo paths to absolute paths under config_dir."""
        for repo in self.repos:
            raw = repo.path
            if raw.startswith("${"):
                var = raw[2:-1]
                raw = os.environ.get(var, raw)
            p = Path(raw)
            if not p.is_absolute():
                p = (config_dir / p).resolve()
            repo.resolved_path = p
        return self

    def db_path(self, config_dir: Path) -> Path:
        """Return absolute path to the SQLite DB file."""
        raw = self.storage.path
        p = Path(raw)
        if not p.is_absolute():
            p = (config_dir / p).resolve()
        p.parent.mkdir(parents=True, exist_ok=True)
        return p


def _expand_env(value: Any) -> Any:
    """Recursively expand ${VAR} references in dict/list/str values.

    - ``${VAR}``  → value of env var VAR, or None if not set
                    (a warning is emitted when the var is missing)
    - Any other string → used as-is (literal value)
    """
    if isinstance(value, str):
        if value.startswith("${") and value.endswith("}"):
            var = value[2:-1]
            resolved = os.environ.get(var)
            if resolved is None:
                import warnings
                warnings.warn(
                    f"Environment variable '{var}' is referenced in workspace.yaml but is not set.",
                    UserWarning,
                    stacklevel=2,
                )
            return resolved
        return value
    if isinstance(value, dict):
        return {k: _expand_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env(i) for i in value]
    return value


def load_workspace(path: Path | str) -> "WorkspaceConfig":
    """Load and parse a workspace.yaml file.

    Args:
        path: Path to ``workspace.yaml`` or the directory containing it.

    Returns:
        Parsed and path-resolved :class:`WorkspaceConfig`.

    Raises:
        FileNotFoundError: If the workspace file does not exist.
        ValueError: If the file is not valid YAML or fails schema validation.
    """
    path = Path(path)
    if path.is_dir():
        path = path / "workspace.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Workspace file not found: {path}")

    with path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}

    raw = _expand_env(raw)
    config = WorkspaceConfig.model_validate(raw)
    config._config_path = path
    config.resolve_paths(path.parent)
    return config


def find_workspace_root(start: Path | str | None = None) -> Optional[Path]:
    """Walk up directories looking for workspace.yaml.

    Args:
        start: Directory to start searching from (default: cwd).

    Returns:
        Path to the **directory** containing ``workspace.yaml`` (or
        ``corbell/workspace.yaml`` or ``corbell-data/workspace.yaml``), or
        ``None`` if not found.
    """
    current = Path(start or Path.cwd()).resolve()
    for candidate in [current, *current.parents]:
        for sub in ("corbell", "corbell-data", ""):
            ws = candidate / sub / "workspace.yaml" if sub else candidate / "workspace.yaml"
            if ws.exists():
                return ws.parent
    return None


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


def _detect_repos(target_dir: Path) -> List[Dict[str, Any]]:
    """Detect repos in the target directory (single repo or monorepo subdirectories)."""
    repos = []

    def is_repo_dir(d: Path) -> bool:
        indicators = [
            ".git", "package.json", "requirements.txt", "pyproject.toml",
            "go.mod", "pom.xml", "Cargo.toml",
        ]
        return any((d / i).exists() for i in indicators)

    if is_repo_dir(target_dir):
        sub_repos = []
        for child in target_dir.iterdir():
            if (
                child.is_dir()
                and not child.name.startswith(".")
                and child.name not in ("node_modules", "venv", ".venv", "dist", "build")
            ):
                if is_repo_dir(child):
                    sub_repos.append(child)

        if len(sub_repos) > 0:
            for child in sub_repos:
                repos.append({
                    "id": child.name,
                    "path": f"../{child.name}",
                    "language": _detect_language(child),
                })
        else:
            repos.append({
                "id": target_dir.name,
                "path": "..",
                "language": _detect_language(target_dir),
            })
    else:
        for child in target_dir.iterdir():
            if (
                child.is_dir()
                and not child.name.startswith(".")
                and child.name not in ("node_modules", "venv", ".venv", "dist", "build")
            ):
                if is_repo_dir(child):
                    repos.append({
                        "id": child.name,
                        "path": f"../{child.name}",
                        "language": _detect_language(child),
                    })

    if not repos:
        repos.append({
            "id": "my-repo",
            "path": "../my-repo",
            "language": "python",
        })

    return repos


def init_workspace_yaml(target_dir: Path) -> Path:
    """Write a starter workspace.yaml into target_dir/corbell/workspace.yaml.

    Args:
        target_dir: Root directory for the new workspace.

    Returns:
        Path to the written file.
    """
    out_dir = target_dir / "corbell"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "workspace.yaml"
    repos_detected = _detect_repos(target_dir)
    repos_yaml = ""
    for repo in repos_detected:
        repos_yaml += f"  - id: {repo['id']}\n"
        repos_yaml += f"    path: {repo['path']}\n"
        if repo.get("language"):
            repos_yaml += f"    language: {repo['language']}\n"

    template = """\
version: "1"

workspace:
  name: "my-platform"
  root: ".."

repos:
{repos_block}

storage:
  path: .corbell/workspace.db
  model: all-MiniLM-L6-v2

query:
  top_k: 50
  expand_call_depth: 2
  expand_max_chunks: 30
  rerank: true

indexing:
  skip_dirs: []
  max_file_bytes: 1048576
  chunk_size: 50
  chunk_overlap: 10

llm:
  # ---- Option 1: Anthropic (recommended) ----
  provider: anthropic
  model: claude-sonnet-4-5
  api_key: ${{ANTHROPIC_API_KEY}}

  # ---- Option 2: OpenAI ----
  # provider: openai
  # model: gpt-4o
  # api_key: ${{OPENAI_API_KEY}}
"""
    out.write_text(template.replace("{repos_block}", repos_yaml), encoding="utf-8")
    return out
