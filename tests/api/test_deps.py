"""`app.api.deps::get_graph_models` — plan.md Task 5: build the CHAT model from the model
registry snapshot when `MODEL_REGISTRY_ENABLED=true`, keep the legacy `.env`-based
`OpenAIChatModel` untouched when `false`. No failover here (Task 5's explicit scope) - just
the single highest-priority ACTIVE CHAT credential.
"""

from __future__ import annotations

from typing import Any

import pytest
from pydantic_ai.models.openai import OpenAIChatModel

import app.core.model_registry as model_registry
from app.api.deps import get_graph_models
from app.core.config import settings
from app.core.model_registry import parse_snapshot
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
        "modelName": settings.OPENAI_MODEL,
        "apiBaseUrl": "https://api.openai.com/v1",
        "apiKey": "sk-real-secret-value",
        "priority": 1,
        "maxRpm": 500,
    }
    base.update(overrides)
    return base


def test_registry_disabled_builds_legacy_openai_model_from_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "MODEL_REGISTRY_ENABLED", False)

    graph_models = get_graph_models()

    assert isinstance(graph_models.classification, OpenAIChatModel)
    assert graph_models.classification.model_name == settings.OPENAI_MODEL
    assert graph_models.classification is graph_models.query_transformation
    assert graph_models.classification is graph_models.generation
    assert isinstance(graph_models.retrieval, RetrievalService)


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

    assert isinstance(graph_models.classification, OpenAIChatModel)
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


def test_registry_enabled_with_single_credential_matches_legacy_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No-regression check (Task 5 acceptance): with exactly one CHAT credential mirroring
    today's single-`.env`-key setup, the registry path must produce the same shape of graph
    models as the legacy `.env` path - same Model class, same model name, same
    one-model-for-three-nodes sharing, same retrieval service type."""

    monkeypatch.setattr(settings, "MODEL_REGISTRY_ENABLED", False)
    legacy = get_graph_models()

    monkeypatch.setattr(settings, "MODEL_REGISTRY_ENABLED", True)
    snapshot = parse_snapshot(_snapshot_payload([_chat_credential(modelName=settings.OPENAI_MODEL)]))
    monkeypatch.setattr(model_registry, "_current_snapshot", snapshot)
    registry_based = get_graph_models()

    assert type(legacy.classification) is type(registry_based.classification)
    assert legacy.classification.model_name == registry_based.classification.model_name
    assert registry_based.classification is registry_based.query_transformation
    assert registry_based.classification is registry_based.generation
    assert type(legacy.retrieval) is type(registry_based.retrieval)
