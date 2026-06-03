"""Tests for code chunk extractor and embedding store."""

import sys
from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest

from corbell.core.embeddings.extractor import CodeChunkExtractor, EmbeddingRecord
from corbell.core.embeddings.sqlite_store import SQLiteEmbeddingStore


# ─── Extractor tests ────────────────────────────────────────────────────────

def test_extract_python_functions(sample_repo):
    extractor = CodeChunkExtractor()
    records = extractor.extract_from_repo(sample_repo, "sample-service")
    assert len(records) > 0
    func_types = [r.chunk_type for r in records]
    assert any(t in ("function", "method", "class") for t in func_types)


def test_extract_python_symbols_named(sample_repo):
    extractor = CodeChunkExtractor()
    records = extractor.extract_from_repo(sample_repo, "sample-service")
    symbols = [r.symbol for r in records if r.symbol]
    assert any("get_token" in s or "AuthClient" in s for s in symbols)


def test_extract_generic_markdown(tmp_path):
    f = tmp_path / "DESIGN.md"
    f.write_text("# Title\n\n## Section\n\nsome content\n" * 30)
    extractor = CodeChunkExtractor(chunk_size=20, overlap=5)
    records = extractor.extract_from_repo(tmp_path, "docs")
    assert len(records) > 0
    assert records[0].language == "markdown"


def test_skip_large_files(tmp_path):
    large = tmp_path / "big.py"
    large.write_bytes(b"x" * (2 * 1024 * 1024))  # 2MB
    extractor = CodeChunkExtractor()
    records = extractor.extract_from_repo(tmp_path, "svc", max_file_bytes=1024 * 1024)
    assert len(records) == 0


# ─── SQLiteEmbeddingStore tests ──────────────────────────────────────────────

@pytest.fixture
def emb_store(tmp_db):
    return SQLiteEmbeddingStore(tmp_db)


def _make_record(i: int, svc: str = "svc") -> EmbeddingRecord:
    return EmbeddingRecord(
        id=f"{svc}::f{i}.py::func_{i}",
        service_id=svc,
        repo="/r",
        file_path=f"f{i}.py",
        start_line=1,
        end_line=10,
        content=f"def func_{i}(): pass",
        language="python",
        chunk_type="function",
        symbol=f"func_{i}",
        embedding=[float(i) * 0.1] * 384,
    )


def test_upsert_and_count(emb_store):
    for i in range(5):
        emb_store.upsert(_make_record(i))
    assert emb_store.count() == 5


def test_upsert_batch(emb_store):
    records = [_make_record(i) for i in range(10)]
    emb_store.upsert_batch(records)
    assert emb_store.count() == 10


def test_query_returns_results(emb_store):
    records = [_make_record(i) for i in range(5)]
    emb_store.upsert_batch(records)

    # Query with embedding close to record 2
    qvec = [0.2] * 384
    results = emb_store.query(qvec, top_k=3)
    assert len(results) <= 3
    assert all(isinstance(r, EmbeddingRecord) for r in results)


def test_query_service_filter(emb_store):
    for i in range(3):
        emb_store.upsert(_make_record(i, "svc-a"))
    for i in range(3):
        emb_store.upsert(_make_record(i, "svc-b"))

    qvec = [0.1] * 384
    results = emb_store.query(qvec, service_ids=["svc-a"], top_k=10)
    assert all(r.service_id == "svc-a" for r in results)


def test_clear_all(emb_store):
    emb_store.upsert_batch([_make_record(i) for i in range(5)])
    emb_store.clear()
    assert emb_store.count() == 0


def test_clear_service(emb_store):
    for i in range(3):
        emb_store.upsert(_make_record(i, "svc-a"))
    for i in range(3):
        emb_store.upsert(_make_record(i, "svc-b"))
    emb_store.clear(service_id="svc-a")
    assert emb_store.count() == 3
    assert emb_store.count("svc-b") == 3


# ─── New method tests (Task 8.10) ──────────────────────────────────────────

def test_delete_by_file(emb_store):
    """delete_by_file removes only chunks for the specified file and repo."""
    for i in range(3):
        emb_store.upsert(_make_record(i))  # all use f{i}.py
    # f0.py, f1.py, f2.py — delete f1.py
    deleted = emb_store.delete_by_file("f1.py", "svc")
    assert deleted == 1
    assert emb_store.count() == 2


def test_delete_by_file_wrong_repo(emb_store):
    """delete_by_file with wrong repo_id deletes nothing."""
    emb_store.upsert(_make_record(0))
    deleted = emb_store.delete_by_file("f0.py", "other-svc")
    assert deleted == 0
    assert emb_store.count() == 1


def test_get_all_vectors(emb_store):
    """get_all_vectors returns (id, blob) tuples for all records with embeddings."""
    records = [_make_record(i) for i in range(5)]
    emb_store.upsert_batch(records)

    all_vecs = emb_store.get_all_vectors()
    assert len(all_vecs) == 5
    for chunk_id, blob in all_vecs:
        assert isinstance(chunk_id, str)
        assert isinstance(blob, bytes)
        assert len(blob) > 0


def test_get_all_vectors_empty(emb_store):
    """get_all_vectors returns empty list when no embeddings stored."""
    result = emb_store.get_all_vectors()
    assert result == []


def test_get_chunks_by_ids(emb_store):
    """get_chunks_by_ids fetches the correct records."""
    records = [_make_record(i) for i in range(5)]
    emb_store.upsert_batch(records)

    ids_to_fetch = [records[1].id, records[3].id]
    fetched = emb_store.get_chunks_by_ids(ids_to_fetch)
    assert len(fetched) == 2
    fetched_ids = {r.id for r in fetched}
    assert fetched_ids == set(ids_to_fetch)


def test_get_chunks_by_ids_empty(emb_store):
    """get_chunks_by_ids returns empty list for empty input."""
    result = emb_store.get_chunks_by_ids([])
    assert result == []


def test_get_chunks_by_ids_missing(emb_store):
    """get_chunks_by_ids returns only existing records."""
    emb_store.upsert(_make_record(0))
    result = emb_store.get_chunks_by_ids(["nonexistent::id"])
    assert result == []


# ─── GoogleEmbeddingModel tests ──────────────────────────────────────────────

def _make_google_mocks(num_texts: int = 2):
    """Build minimal google.genai / types mocks for embedding tests."""
    embeddings = []
    for i in range(num_texts):
        emb = MagicMock()
        emb.values = [float(i) * 0.1] * 768
        embeddings.append(emb)

    result_mock = MagicMock()
    result_mock.embeddings = embeddings

    models_mock = MagicMock()
    models_mock.embed_content.return_value = result_mock

    client_instance = MagicMock()
    client_instance.models = models_mock

    genai_mod = MagicMock()
    genai_mod.Client.return_value = client_instance

    types_mod = MagicMock()
    types_mod.EmbedContentConfig = MagicMock(return_value=MagicMock())

    return genai_mod, types_mod, models_mock


def test_google_embedding_model_encode():
    """encode() returns one vector per input text."""
    from corbell.core.embeddings.model import GoogleEmbeddingModel

    texts = ["hello", "world"]
    genai_mod, types_mod, models_mock = _make_google_mocks(num_texts=len(texts))

    google_pkg = ModuleType("google")
    google_pkg.genai = genai_mod
    genai_pkg = ModuleType("google.genai")
    genai_pkg.types = types_mod

    with patch.dict(sys.modules, {
        "google": google_pkg,
        "google.genai": genai_pkg,
        "google.genai.types": types_mod,
    }):
        model = GoogleEmbeddingModel(api_key="test-key")
        result = model.encode(texts)

    assert len(result) == len(texts)


def test_google_embedding_model_dimension():
    """dimension property returns 768."""
    from corbell.core.embeddings.model import GoogleEmbeddingModel
    model = GoogleEmbeddingModel()
    assert model.dimension == 768


def test_google_embedding_model_default_name():
    """Default model name is gemini-embedding-001."""
    from corbell.core.embeddings.model import GoogleEmbeddingModel
    model = GoogleEmbeddingModel()
    assert model.model_name == "gemini-embedding-001"


def test_google_embedding_model_task_type_forwarded():
    """task_type is forwarded to embed_content config."""
    from corbell.core.embeddings.model import GoogleEmbeddingModel

    texts = ["query text"]
    genai_mod, types_mod, models_mock = _make_google_mocks(num_texts=len(texts))

    google_pkg = ModuleType("google")
    google_pkg.genai = genai_mod
    genai_pkg = ModuleType("google.genai")
    genai_pkg.types = types_mod

    with patch.dict(sys.modules, {
        "google": google_pkg,
        "google.genai": genai_pkg,
        "google.genai.types": types_mod,
    }):
        model = GoogleEmbeddingModel(api_key="test-key")
        model.encode(texts, task_type="RETRIEVAL_QUERY")

    # Verify EmbedContentConfig was called with the right task_type
    types_mod.EmbedContentConfig.assert_called_once_with(
        task_type="RETRIEVAL_QUERY",
        output_dimensionality=768,
    )


def test_google_embedding_model_api_key_from_env(monkeypatch):
    """api_key is resolved from GOOGLE_API_KEY env var."""
    from corbell.core.embeddings.model import GoogleEmbeddingModel

    monkeypatch.setenv("GOOGLE_API_KEY", "env-google-key")
    model = GoogleEmbeddingModel()
    assert model._api_key == "env-google-key"


def test_google_embedding_model_import_error():
    """ImportError raised when google-genai is not installed."""
    from corbell.core.embeddings.model import GoogleEmbeddingModel

    model = GoogleEmbeddingModel(api_key="test-key")

    with patch.dict(sys.modules, {"google": None, "google.genai": None}):
        with pytest.raises(ImportError, match="pip install corbell\\[google\\]"):
            model.encode(["text"])


# ─── GoogleEmbeddingModel multi-key tests ────────────────────────────────────

def _make_google_mocks_for_keys(num_texts: int = 1):
    """Return (genai_mod, types_mod) with a fresh models mock per Client() call."""
    embeddings = []
    for i in range(num_texts):
        emb = MagicMock()
        emb.values = [float(i) * 0.1] * 768
        embeddings.append(emb)
    result_mock = MagicMock()
    result_mock.embeddings = embeddings

    models_mock = MagicMock()
    models_mock.embed_content.return_value = result_mock

    client_instance = MagicMock()
    client_instance.models = models_mock

    genai_mod = MagicMock()
    genai_mod.Client.return_value = client_instance

    types_mod = MagicMock()
    types_mod.EmbedContentConfig = MagicMock(return_value=MagicMock())

    return genai_mod, types_mod, models_mock, client_instance


def test_google_embedding_multikey_parses_csv(monkeypatch):
    """Constructor parses comma-separated keys into _api_keys list."""
    from corbell.core.embeddings.model import GoogleEmbeddingModel

    model = GoogleEmbeddingModel(api_key="key1, key2 , key3")
    assert model._api_keys == ["key1", "key2", "key3"]


def test_google_embedding_multikey_parses_env(monkeypatch):
    """Constructor parses CSV from GOOGLE_API_KEY env var."""
    from corbell.core.embeddings.model import GoogleEmbeddingModel

    monkeypatch.setenv("GOOGLE_API_KEY", "envkey1,envkey2")
    model = GoogleEmbeddingModel()
    assert model._api_keys == ["envkey1", "envkey2"]


def test_google_embedding_multikey_roundrobin():
    """_key_index advances after a successful encode call."""
    from corbell.core.embeddings.model import GoogleEmbeddingModel

    genai_mod, types_mod, models_mock, _ = _make_google_mocks_for_keys()

    google_pkg = ModuleType("google")
    google_pkg.genai = genai_mod
    genai_pkg = ModuleType("google.genai")
    genai_pkg.types = types_mod

    with patch.dict(sys.modules, {
        "google": google_pkg, "google.genai": genai_pkg, "google.genai.types": types_mod,
    }):
        model = GoogleEmbeddingModel(api_key="key0,key1,key2")
        assert model._key_index == 0
        model.encode(["text"])
        assert model._key_index == 1
        model.encode(["text"])
        assert model._key_index == 2
        model.encode(["text"])
        assert model._key_index == 0  # wraps


def test_google_embedding_multikey_failover():
    """On key error for key[0], retries with key[1] and succeeds."""
    from corbell.core.embeddings.model import GoogleEmbeddingModel

    emb = MagicMock()
    emb.values = [0.1] * 768
    result_mock = MagicMock()
    result_mock.embeddings = [emb]

    # key error for code 401
    key_err = Exception("bad key")
    key_err.code = 401
    key_err.message = "UNAUTHENTICATED"

    call_count = {"n": 0}

    def fake_embed_content(**kwargs):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise key_err
        return result_mock

    models_mock = MagicMock()
    models_mock.embed_content.side_effect = fake_embed_content

    client_instance = MagicMock()
    client_instance.models = models_mock

    genai_mod = MagicMock()
    genai_mod.Client.return_value = client_instance

    types_mod = MagicMock()
    types_mod.EmbedContentConfig = MagicMock(return_value=MagicMock())

    google_pkg = ModuleType("google")
    google_pkg.genai = genai_mod
    genai_pkg = ModuleType("google.genai")
    genai_pkg.types = types_mod

    with patch.dict(sys.modules, {
        "google": google_pkg, "google.genai": genai_pkg, "google.genai.types": types_mod,
    }):
        model = GoogleEmbeddingModel(api_key="bad-key,good-key")
        result = model.encode(["text"])

    assert len(result) == 1
    assert model._key_index == 0  # key[1] succeeded → next is key[0] (wraps from idx=1)


def test_google_embedding_multikey_all_fail():
    """RuntimeError raised when all keys fail with key errors."""
    from corbell.core.embeddings.model import GoogleEmbeddingModel

    key_err = Exception("invalid key")
    key_err.code = 403
    key_err.message = "PERMISSION_DENIED"

    models_mock = MagicMock()
    models_mock.embed_content.side_effect = key_err

    client_instance = MagicMock()
    client_instance.models = models_mock

    genai_mod = MagicMock()
    genai_mod.Client.return_value = client_instance

    types_mod = MagicMock()
    types_mod.EmbedContentConfig = MagicMock(return_value=MagicMock())

    google_pkg = ModuleType("google")
    google_pkg.genai = genai_mod
    genai_pkg = ModuleType("google.genai")
    genai_pkg.types = types_mod

    with patch.dict(sys.modules, {
        "google": google_pkg, "google.genai": genai_pkg, "google.genai.types": types_mod,
    }):
        model = GoogleEmbeddingModel(api_key="key1,key2")
        with pytest.raises(RuntimeError, match="All 2 Google API key"):
            model.encode(["text"])

    # index unchanged from start (0)
    assert model._key_index == 0


def test_google_embedding_nonkey_400_propagates():
    """A 400 without 'api key' in message is re-raised immediately, not rotated."""
    from corbell.core.embeddings.model import GoogleEmbeddingModel

    bad_req = Exception("invalid contents field")
    bad_req.code = 400
    bad_req.message = "INVALID_ARGUMENT: invalid contents field"

    models_mock = MagicMock()
    models_mock.embed_content.side_effect = bad_req

    client_instance = MagicMock()
    client_instance.models = models_mock

    genai_mod = MagicMock()
    genai_mod.Client.return_value = client_instance

    types_mod = MagicMock()
    types_mod.EmbedContentConfig = MagicMock(return_value=MagicMock())

    google_pkg = ModuleType("google")
    google_pkg.genai = genai_mod
    genai_pkg = ModuleType("google.genai")
    genai_pkg.types = types_mod

    with patch.dict(sys.modules, {
        "google": google_pkg, "google.genai": genai_pkg, "google.genai.types": types_mod,
    }):
        model = GoogleEmbeddingModel(api_key="key1,key2")
        with pytest.raises(Exception, match="invalid contents field"):
            model.encode(["text"])

    # Only one attempt — embed_content called once
    assert models_mock.embed_content.call_count == 1


# ─── VoyageEmbeddingModel tests ──────────────────────────────────────────────

def _make_voyage_mocks(num_texts: int = 2):
    """Build minimal voyageai Client mock for embedding tests."""
    embeddings = [[float(i) * 0.1] * 1024 for i in range(num_texts)]

    result_mock = MagicMock()
    result_mock.embeddings = embeddings

    client_instance = MagicMock()
    client_instance.embed.return_value = result_mock

    voyage_mod = MagicMock()
    voyage_mod.Client.return_value = client_instance

    return voyage_mod, client_instance


def test_voyage_embedding_model_dimension():
    """dimension property returns 1024 by default."""
    from corbell.core.embeddings.model import VoyageEmbeddingModel

    model = VoyageEmbeddingModel(api_key="test-key")
    assert model.dimension == 1024


def test_voyage_embedding_model_dimension_env_override(monkeypatch):
    """dimension property reads CORBELL_EMBEDDING_DIM env var."""
    from corbell.core.embeddings.model import VoyageEmbeddingModel

    monkeypatch.setenv("CORBELL_EMBEDDING_DIM", "512")
    model = VoyageEmbeddingModel(api_key="test-key")
    assert model.dimension == 512


def test_voyage_embedding_model_default_name():
    """Default model name is voyage-code-3."""
    from corbell.core.embeddings.model import VoyageEmbeddingModel

    model = VoyageEmbeddingModel(api_key="test-key")
    assert model.model_name == "voyage-code-3"


def test_voyage_embedding_model_encode():
    """encode() returns one vector per input text."""
    from corbell.core.embeddings.model import VoyageEmbeddingModel

    texts = ["hello", "world"]
    voyage_mod, client_instance = _make_voyage_mocks(num_texts=len(texts))

    with patch.dict(sys.modules, {"voyageai": voyage_mod}):
        model = VoyageEmbeddingModel(api_key="test-key")
        result = model.encode(texts)

    assert len(result) == len(texts)
    assert len(result[0]) == 1024


def test_voyage_embedding_model_input_type_forwarded():
    """input_type is forwarded to Client.embed."""
    from corbell.core.embeddings.model import VoyageEmbeddingModel

    texts = ["query text"]
    voyage_mod, client_instance = _make_voyage_mocks(num_texts=len(texts))

    with patch.dict(sys.modules, {"voyageai": voyage_mod}):
        model = VoyageEmbeddingModel(api_key="test-key")
        model.encode(texts, input_type="query")

    client_instance.embed.assert_called_once()
    call_kwargs = client_instance.embed.call_args
    assert call_kwargs.kwargs.get("input_type") == "query" or call_kwargs.args[1] == "query" \
        or "query" in str(call_kwargs)


def test_voyage_embedding_model_input_type_forwarded_kwargs():
    """input_type='query' is forwarded correctly via kwargs."""
    from corbell.core.embeddings.model import VoyageEmbeddingModel

    texts = ["query text"]
    voyage_mod, client_instance = _make_voyage_mocks(num_texts=len(texts))

    with patch.dict(sys.modules, {"voyageai": voyage_mod}):
        model = VoyageEmbeddingModel(api_key="test-key")
        model.encode(texts, input_type="query")

    _, kwargs = client_instance.embed.call_args
    assert kwargs.get("input_type") == "query"


def test_voyage_embedding_model_api_key_from_env(monkeypatch):
    """api_key is resolved from VOYAGE_API_KEY env var."""
    from corbell.core.embeddings.model import VoyageEmbeddingModel

    monkeypatch.setenv("VOYAGE_API_KEY", "env-voyage-key")
    model = VoyageEmbeddingModel()
    assert model._api_key == "env-voyage-key"


def test_voyage_embedding_model_import_error():
    """ImportError raised when voyageai is not installed."""
    from corbell.core.embeddings.model import VoyageEmbeddingModel

    model = VoyageEmbeddingModel(api_key="test-key")

    with patch.dict(sys.modules, {"voyageai": None}):
        with pytest.raises(ImportError, match="pip install corbell\\[voyage\\]"):
            model.encode(["text"])


def test_voyage_embedding_model_no_key_raises():
    """ValueError raised when no API key is provided."""
    from corbell.core.embeddings.model import VoyageEmbeddingModel
    import os

    # ensure env var is not set
    env_backup = os.environ.pop("VOYAGE_API_KEY", None)
    try:
        with pytest.raises(ValueError, match="VOYAGE_API_KEY"):
            VoyageEmbeddingModel()
    finally:
        if env_backup is not None:
            os.environ["VOYAGE_API_KEY"] = env_backup


# ─── VoyageEmbeddingModel multi-key tests ───────────────────────────────────

def test_voyage_embedding_multikey_parses_csv():
    """Constructor parses comma-separated keys into _api_keys list."""
    from corbell.core.embeddings.model import VoyageEmbeddingModel

    model = VoyageEmbeddingModel(api_key="key1, key2 , key3")
    assert model._api_keys == ["key1", "key2", "key3"]


def test_voyage_embedding_multikey_parses_env(monkeypatch):
    """Constructor parses CSV from VOYAGE_API_KEY env var."""
    from corbell.core.embeddings.model import VoyageEmbeddingModel

    monkeypatch.setenv("VOYAGE_API_KEY", "pa-key1,pa-key2")
    model = VoyageEmbeddingModel()
    assert model._api_keys == ["pa-key1", "pa-key2"]


def test_voyage_embedding_multikey_roundrobin():
    """_key_index advances after a successful encode call."""
    from corbell.core.embeddings.model import VoyageEmbeddingModel

    voyage_mod, _ = _make_voyage_mocks(num_texts=1)

    with patch.dict(sys.modules, {"voyageai": voyage_mod}):
        model = VoyageEmbeddingModel(api_key="key0,key1,key2")
        assert model._key_index == 0
        model.encode(["text"])
        assert model._key_index == 1
        model.encode(["text"])
        assert model._key_index == 2
        model.encode(["text"])
        assert model._key_index == 0  # wraps


def test_voyage_embedding_multikey_failover():
    """On rate limit for key[0], retries with key[1] and succeeds."""
    from corbell.core.embeddings.model import VoyageEmbeddingModel

    embeddings = [[0.1] * 1024]
    result_mock = MagicMock()
    result_mock.embeddings = embeddings

    rate_err = Exception("rate limited")
    rate_err.status_code = 429

    call_count = {"n": 0}

    def fake_embed(*args, **kwargs):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise rate_err
        return result_mock

    client_instance = MagicMock()
    client_instance.embed.side_effect = fake_embed

    voyage_mod = MagicMock()
    voyage_mod.Client.return_value = client_instance

    with patch.dict(sys.modules, {"voyageai": voyage_mod}):
        model = VoyageEmbeddingModel(api_key="slow-key,fast-key")
        result = model.encode(["text"])

    assert len(result) == 1
    assert model._key_index == 0  # key[1] succeeded → next is key[0] (wraps from idx=1)


def test_voyage_embedding_retry_on_429():
    """All keys rate-limited triggers exponential backoff and eventual success."""
    from corbell.core.embeddings.model import VoyageEmbeddingModel

    embeddings = [[0.1] * 1024]
    result_mock = MagicMock()
    result_mock.embeddings = embeddings

    rate_err = Exception("quota exhausted")
    rate_err.status_code = 429

    call_count = {"n": 0}

    def fake_embed(*args, **kwargs):
        call_count["n"] += 1
        if call_count["n"] < 3:  # fail twice then succeed
            raise rate_err
        return result_mock

    client_instance = MagicMock()
    client_instance.embed.side_effect = fake_embed

    voyage_mod = MagicMock()
    voyage_mod.Client.return_value = client_instance

    with patch.dict(sys.modules, {"voyageai": voyage_mod}):
        with patch("time.sleep"):  # avoid actual sleeps in tests
            model = VoyageEmbeddingModel(api_key="only-key")
            result = model.encode(["text"])

    assert len(result) == 1
