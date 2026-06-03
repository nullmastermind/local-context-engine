"""Shared test fixtures for Corbell code retrieval engine."""

from __future__ import annotations

import hashlib
import textwrap
from pathlib import Path
from typing import List
from unittest.mock import MagicMock

import numpy as np
import pytest


@pytest.fixture
def tmp_db(tmp_path) -> Path:
    """Return a temporary SQLite database path."""
    return tmp_path / "test.db"


@pytest.fixture
def sample_repo(tmp_path) -> Path:
    """Create a minimal sample Python repo for testing."""
    repo = tmp_path / "sample-service"
    repo.mkdir()

    (repo / "__init__.py").write_text("")
    (repo / "auth_client.py").write_text(textwrap.dedent("""\
        import requests
        import redis

        class AuthClient:
            def get_token(self, user_id: str) -> str:
                cache = redis.Redis(host="localhost")
                cached = cache.get(f"token:{user_id}")
                if cached:
                    return cached.decode()
                resp = requests.get(f"http://auth-service/token/{user_id}")
                return resp.json()["token"]

            def validate_token(self, token: str) -> bool:
                resp = requests.post("http://auth-service/validate", json={"token": token})
                return resp.json().get("valid", False)
    """))

    (repo / "db.py").write_text(textwrap.dedent("""\
        import psycopg2
        from psycopg2 import sql

        class Database:
            def __init__(self, dsn: str):
                self.conn = psycopg2.connect(dsn)

            def get_user(self, user_id: str) -> dict:
                cursor = self.conn.cursor()
                cursor.execute("SELECT * FROM users WHERE id = %s", (user_id,))
                return cursor.fetchone()
    """))

    (repo / "orchestrator.py").write_text(textwrap.dedent("""\
        from auth_client import AuthClient
        from db import Database

        class Orchestrator:
            def __init__(self):
                self.auth = AuthClient()
                self.db = Database("postgresql://localhost/mydb")

            def process(self, user_id: str, payload: dict) -> dict:
                token = self.auth.get_token(user_id)
                user = self.db.get_user(user_id)
                return {"user": user, "token": token, "payload": payload}
    """))

    return repo


@pytest.fixture
def sample_workspace_yaml(tmp_path, sample_repo) -> Path:
    """Write a valid workspace.yaml into tmp_path/corbell/ using new repos schema."""
    ws_dir = tmp_path / "corbell"
    ws_dir.mkdir()
    yaml_content = f"""\
version: "1"
workspace:
  name: test-platform
  root: ..

repos:
  - id: sample-service
    path: {sample_repo}
    language: python

storage:
  path: .corbell/test.db
  model: all-MiniLM-L6-v2

query:
  top_k: 50
  expand_call_depth: 2
  expand_max_chunks: 30
  rerank: false

indexing:
  skip_dirs: []
  max_file_bytes: 1048576
  chunk_size: 50
  chunk_overlap: 10

llm:
  provider: anthropic
  model: claude-sonnet-4-5
"""
    ws_file = ws_dir / "workspace.yaml"
    ws_file.write_text(yaml_content)
    return ws_file


@pytest.fixture
def mock_llm():
    """Return a mock LLMClient that returns deterministic responses."""
    m = MagicMock()
    m.is_configured = True
    m.call.return_value = '["chunk_id_1", "chunk_id_2"]'
    return m


@pytest.fixture
def mock_embedding_model():
    """Return a mock embedding model that produces deterministic hash-based vectors.

    Each text gets a 384-dim vector derived from its MD5 hash, normalized to
    unit length. This ensures different texts get different vectors, and the
    same text always gets the same vector.
    """
    dim = 384

    def _encode(texts: List[str]) -> List[List[float]]:
        result = []
        for text in texts:
            # Hash the text to get a deterministic seed
            h = hashlib.md5(text.encode()).digest()
            # Expand the 16-byte hash to 384 floats using repetition
            seed = int.from_bytes(h, "big") % (2**32)
            rng = np.random.RandomState(seed)
            vec = rng.randn(dim).astype(np.float32)
            # Normalize to unit length
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            result.append(vec.tolist())
        return result

    m = MagicMock()
    m.encode.side_effect = _encode
    return m
