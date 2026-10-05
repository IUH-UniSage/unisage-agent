"""`app.api.deps::get_graph_models` — plan.md Task 5/"Cutover khỏi cấu hình `.env` tĩnh":
builds the CHAT model from the model registry snapshot, and never falls back to any static
credential - not even when `MODEL_REGISTRY_ENABLED` is off or no snapshot was ever loaded.
No failover here (Task 5's explicit scope) - just the single highest-priority ACTIVE CHAT
credential.
"""

from __future__ import annotations

from typing import Any

import pytest
from pydantic_ai.models.openai import OpenAIChatModel

import app.core.registry.model_registry as model_registry
from app.api.deps import get_graph_models
from app.core.config import settings
from app.core.errors.error_codes import ErrorCode
from app.core.errors.llm_failure import LLMCallException
from app.core.llm.rate_limited_model import RateLimitedModel
from app.core.registry.model_registry import ModelRegistryError, parse_snapshot
from app.rag.retrieval.service import RetrievalService


@pytest.fixture(autouse=True)
def _reset_cached_snapshot() -> Any:
    model_registry._current_snapshot = None
    yield
    model_registry._current_snapshot = None


def _snapshot_payload(credentials: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "version": 7,
        "generatedAt": "2026-09-25T03:00:00Z",
        "purposes": {"CHAT": credentials, "EMBEDDING": [], "EXTRACTION": []},
        "embeddingIndexIdentity": None,
    }


def _chat_credential(**overrides: object) -> dict[str, Any]:
    base: dict[str, Any] = {
        "id": "11111111-1111-1111-1111-111111111111",
        "revision": 3,
        "sourceType": "CLOUD_API",
        "provider": "openai",
        "modelName": "gpt-4o-mini",
        "apiBaseUrl": "https://api.openai.com/v1",
        "apiKey": "sk-real-secret-value",
        "priority": 1,
        "maxRpm": 500,
    }
    base.update(overrides)
    return base


def test_registry_enabled_builds_model_from_highest_priority_chat_credential(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "MODEL_REGISTRY_ENABLED", True)
    snapshot = parse_snapshot(
        _snapshot_payload(
            [
                _chat_credential(id="low-priority", priority=2, modelName="gpt-4o"),
                _chat_credential(id="high-priority", priority=1, modelName="gpt-4o-mini"),
            ]
        )
    )
    monkeypatch.setattr(model_registry, "_current_snapshot", snapshot)

    graph_models = get_graph_models()

    assert isinstance(graph_models.classification, RateLimitedModel)
    assert isinstance(graph_models.classification.wrapped, OpenAIChatModel)
    # priority=1 (the lower number) must win over priority=2 - no failover, just top priority.
    assert graph_models.classification.model_name == "gpt-4o-mini"


def test_registry_enabled_shares_one_model_across_the_three_nodes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "MODEL_REGISTRY_ENABLED", True)
    snapshot = parse_snapshot(_snapshot_payload([_chat_credential()]))
    monkeypatch.setattr(model_registry, "_current_snapshot", snapshot)

    graph_models = get_graph_models()

    assert graph_models.classification is graph_models.query_transformation
    assert graph_models.classification is graph_models.generation
    assert isinstance(graph_models.retrieval, RetrievalService)


def test_registry_enabled_with_no_chat_credential_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "MODEL_REGISTRY_ENABLED", True)
    monkeypatch.setattr(model_registry, "_current_snapshot", parse_snapshot(_snapshot_payload([])))

    with pytest.raises(LLMCallException) as exc_info:
        get_graph_models()

    # Surfaces to the client as a specific "CHAT model not configured" error, not a generic 500.
    assert exc_info.value.error_code is ErrorCode.LLM_NOT_CONFIGURED
    assert isinstance(exc_info.value.__cause__, ModelRegistryError)


def test_no_snapshot_loaded_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """Same refusal even if `init_model_registry()` never ran at all (should not happen in
    practice - it gates startup - but `get_graph_models()` must not paper over it either)."""

    monkeypatch.setattr(settings, "MODEL_REGISTRY_ENABLED", True)
    monkeypatch.setattr(model_registry, "_current_snapshot", None)

    with pytest.raises(LLMCallException) as exc_info:
        get_graph_models()

    # Surfaces to the client as a specific "CHAT model not configured" error, not a generic 500.
    assert exc_info.value.error_code is ErrorCode.LLM_NOT_CONFIGURED
    assert isinstance(exc_info.value.__cause__, ModelRegistryError)


def test_registry_flag_off_still_raises_no_legacy_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """Flipping `MODEL_REGISTRY_ENABLED` off no longer resurrects any static-credential path -
    that branch was deleted in the cutover (plan.md "Cutover khỏi cấu hình `.env` tĩnh"). All the
    flag controls now is whether `init_model_registry()` loads a snapshot at startup; it has no
    effect on `get_graph_models()` itself, which only ever looks at the current snapshot."""

    monkeypatch.setattr(settings, "MODEL_REGISTRY_ENABLED", False)
    monkeypatch.setattr(model_registry, "_current_snapshot", None)

    with pytest.raises(LLMCallException) as exc_info:
        get_graph_models()

    # Surfaces to the client as a specific "CHAT model not configured" error, not a generic 500.
    assert exc_info.value.error_code is ErrorCode.LLM_NOT_CONFIGURED
    assert isinstance(exc_info.value.__cause__, ModelRegistryError)
