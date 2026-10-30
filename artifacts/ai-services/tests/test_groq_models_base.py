"""Groq /models base normalization (live-frontend batch, item 3).

GROQ_BASE_URL is host-only (https://api.groq.com) because the Groq SDK
appends /openai/v1 itself. Raw use for {base}/models would request
https://api.groq.com/models and 404 (live catalog degrade, broken
Test-connection, boot-check warning spam). groq_models_base() must add
/openai/v1 only when the base carries no /v1 path segment.
"""

from config.settings import groq_models_base


def test_host_only_base_gets_openai_v1():
    assert groq_models_base("https://api.groq.com") == "https://api.groq.com/openai/v1"


def test_full_base_passes_through():
    assert groq_models_base("https://api.groq.com/openai/v1") == "https://api.groq.com/openai/v1"


def test_trailing_slash_trimmed_not_doubled():
    assert groq_models_base("https://api.groq.com/openai/v1/") == "https://api.groq.com/openai/v1"


def test_non_groq_v1_proxy_passes_through():
    assert groq_models_base("https://proxy.example.com/v1") == "https://proxy.example.com/v1"


def test_none_falls_back_to_full_default():
    assert groq_models_base(None) == "https://api.groq.com/openai/v1"
