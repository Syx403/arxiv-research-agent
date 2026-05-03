from __future__ import annotations

from pathlib import Path

import pytest

from src.llm.registry import REGISTRY, get_route


def test_registry_default_keys_only() -> None:
    assert set(REGISTRY) == {"main", "fast", "embed-small", "rerank"}
    assert REGISTRY["main"].provider == "deepseek_native"
    assert REGISTRY["main"].vendor_model == "deepseek-v4-pro"
    assert REGISTRY["fast"].provider == "deepseek_native"
    assert REGISTRY["fast"].vendor_model == "deepseek-v4-flash"
    assert REGISTRY["embed-small"].vendor_model == "text-embedding-3-small"
    assert REGISTRY["rerank"].vendor_model == "rerank-v4.0-pro"


def test_unknown_model_raises_key_error() -> None:
    with pytest.raises(KeyError):
        get_route("haiku-4.5")


def test_capability_mismatch_raises_key_error() -> None:
    with pytest.raises(KeyError):
        get_route("main", capability="embedding")


def test_fallback_haiku_is_comment_only() -> None:
    text = Path("src/llm/registry.py").read_text()
    assert '# "fallback-haiku": ProviderRoute("openrouter", "anthropic/claude-haiku-4.5", "chat")' in text
    assert "fallback-haiku" not in REGISTRY
