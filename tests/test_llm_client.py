"""Tests for LLMClient — Google AI provider."""

from __future__ import annotations

import sys
from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest

from corbell.core.llm_client import LLMClient


# ---------------------------------------------------------------------------
# Helpers — build a minimal google.genai mock hierarchy
# ---------------------------------------------------------------------------

def _make_genai_mock(response_text: str = "hello from gemini",
                     input_tokens: int = 10,
                     output_tokens: int = 20):
    """Return a (genai_module_mock, types_module_mock) pair."""
    # types mock
    types_mod = MagicMock()
    types_mod.GenerateContentConfig = MagicMock(return_value=MagicMock())

    # usage metadata
    usage = MagicMock()
    usage.prompt_token_count = input_tokens
    usage.candidates_token_count = output_tokens

    # response
    response = MagicMock()
    response.text = response_text
    response.usage_metadata = usage

    # client.models.generate_content
    models_mock = MagicMock()
    models_mock.generate_content.return_value = response

    # genai.Client(...)
    client_instance = MagicMock()
    client_instance.models = models_mock

    genai_mod = MagicMock()
    genai_mod.Client.return_value = client_instance

    return genai_mod, types_mod, response


# ---------------------------------------------------------------------------
# Scenario: Default model is gemini-2.5-flash  (task 2.1)
# ---------------------------------------------------------------------------

def test_google_default_model():
    client = LLMClient(provider="google")
    assert client.model == "gemini-2.5-flash"


# ---------------------------------------------------------------------------
# Scenario: is_configured True when API key present  (task 2.2 / spec §3)
# ---------------------------------------------------------------------------

def test_google_is_configured_true():
    client = LLMClient(provider="google", api_key="AIza-test-key")
    assert client.is_configured is True


# ---------------------------------------------------------------------------
# Scenario: is_configured False when no API key  (spec §4)
# ---------------------------------------------------------------------------

def test_google_is_configured_false(monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("CORBELL_LLM_API_KEY", raising=False)
    client = LLMClient(provider="google")
    assert client.is_configured is False


# ---------------------------------------------------------------------------
# Scenario: API key resolved from GOOGLE_API_KEY env var  (spec §5)
# ---------------------------------------------------------------------------

def test_google_resolve_key_from_env(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "env-test-key")
    client = LLMClient(provider="google")
    assert client._api_key == "env-test-key"


# ---------------------------------------------------------------------------
# Scenario: call() dispatches to _call_google_ai and returns response.text  (spec §1)
# ---------------------------------------------------------------------------

def test_google_call_returns_text():
    genai_mod, types_mod, response = _make_genai_mock("result text")

    google_pkg = ModuleType("google")
    google_pkg.genai = genai_mod
    genai_pkg = ModuleType("google.genai")
    genai_pkg.types = types_mod

    with patch.dict(sys.modules, {
        "google": google_pkg,
        "google.genai": genai_pkg,
        "google.genai.types": types_mod,
    }):
        client = LLMClient(provider="google", api_key="AIza-test")
        result = client._call_google_ai("system", "user", 100, 0.1, "test")

    assert result == "result text"


# ---------------------------------------------------------------------------
# Scenario: Missing google-genai package raises ImportError  (spec §6)
# ---------------------------------------------------------------------------

def test_google_import_error():
    client = LLMClient(provider="google", api_key="AIza-test")

    with patch.dict(sys.modules, {"google": None, "google.genai": None}):
        with pytest.raises(ImportError, match="pip install corbell\\[google\\]"):
            client._call_google_ai("system", "user", 100, 0.1)


# ---------------------------------------------------------------------------
# Scenario: Token usage tracked  (spec §7)
# ---------------------------------------------------------------------------

def test_google_token_tracking(monkeypatch):
    genai_mod, types_mod, response = _make_genai_mock(input_tokens=42, output_tokens=17)

    tracker = MagicMock()

    google_pkg = ModuleType("google")
    google_pkg.genai = genai_mod
    genai_types_mod = ModuleType("google.genai")
    genai_types_mod.types = types_mod

    with patch.dict(sys.modules, {
        "google": google_pkg,
        "google.genai": genai_types_mod,
        "google.genai.types": types_mod,
    }):
        client = LLMClient(provider="google", api_key="AIza-test", token_tracker=tracker)
        client._call_google_ai("system", "user", 100, 0.1, "my_request")

    tracker.record.assert_called_once_with("my_request", "gemini-2.5-flash", 42, 17)


# ---------------------------------------------------------------------------
# Scenario: provider_display includes "Google AI" and model name  (spec §9)
# ---------------------------------------------------------------------------

def test_google_provider_display():
    client = LLMClient(provider="google", model="gemini-2.5-flash", api_key="key")
    display = client.provider_display
    assert "Google AI" in display
    assert "gemini-2.5-flash" in display


# ---------------------------------------------------------------------------
# Scenario: Fallback response on missing credentials  (spec §8)
# ---------------------------------------------------------------------------

def test_google_fallback_on_no_credentials(monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("CORBELL_LLM_API_KEY", raising=False)
    client = LLMClient(provider="google")
    result = client.call("system prompt", "user prompt")
    # Should return the fallback string, not raise
    assert isinstance(result, str)
    assert len(result) > 0
    # Fallback string mentions Google AI
    assert "Google AI" in result
