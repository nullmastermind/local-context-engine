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
def workspace_config(sample_repo, monkeypatch):
    """Return a WorkspaceConfig built from sample_repo with env var overrides cleared."""
    # Clear any CORBELL_* env vars that might bleed from the environment
    for var in (
        "CORBELL_TOP_K", "CORBELL_CHUNK_SIZE", "CORBELL_CHUNK_OVERLAP",
        "CORBELL_EXPAND_CALL_DEPTH", "CORBELL_EXPAND_MAX_CHUNKS",
        "CORBELL_RERANK", "CORBELL_EMBEDDING_MODEL", "CORBELL_MAX_FILE_BYTES",
        "CORBELL_SKIP_DIRS", "CORBELL_LLM_MODEL",
    ):
        monkeypatch.delenv(var, raising=False)

    from corbell.core.workspace import build_config
    return build_config(sample_repo)


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
