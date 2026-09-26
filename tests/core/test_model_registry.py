"""`app.core.model_registry` — parses `GET /internal/model-registry/snapshot` into an in-memory
frozen dataclass and gates startup (plan.md "Internal API contract" endpoint #1, "Cutover khỏi
cấu hình `.env` tĩnh"). Java's real response shape is exercised against `httpx.MockTransport`,
never a live Java instance.
"""

from typing import Any

import httpx
import pytest

import app.core.model_registry as model_registry
from app.core.model_registry import (
    ModelRegistryError,
    get_current_snapshot,
    init_model_registry,
    parse_snapshot,
)
from app.integrations.backend_java_client import BackendJavaClient

_SNAPSHOT_PAYLOAD: dict[str, Any] = {
    "version": 42,
    "generatedAt": "2026-09-25T03:00:00Z",
    "purposes": {
        "CHAT": [
            {
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
        ],
        "EMBEDDING": [
            {
                "id": "22222222-2222-2222-2222-222222222222",
                "revision": 1,
                "sourceType": "CLOUD_API",
                "provider": "openai",
                "modelName": "text-embedding-3-small",
                "apiBaseUrl": "https://api.openai.com/v1",
                "apiKey": "sk-embed-secret",
                "priority": 1,
                "maxRpm": 200,
            }
        ],
        "EXTRACTION": [],
    },
    "embeddingIndexIdentity": {
        "provider": "openai",
        "modelName": "text-embedding-3-small",
        "modelSourceRef": None,
        "apiBaseUrl": "https://api.openai.com/v1",
        "dimension": 2,
        # Flat - matches Java's `InternalEmbeddingIndexIdentityResponse.fingerprint` (a single
        # Float[], not a nested array): 3 probes x dimension 2.
        "fingerprint": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
    },
}


@pytest.fixture(autouse=True)
def _reset_cached_snapshot() -> Any:
    """The module keeps a process-local cache — reset it around every test so tests don't
    leak state into each other."""

    model_registry._current_snapshot = None
    yield
    model_registry._current_snapshot = None


def _client_with(handler: Any) -> BackendJavaClient:
    return BackendJavaClient(base_url="http://java.test", transport=httpx.MockTransport(handler))


# ── parse_snapshot: pure parsing, no I/O ────────────────────────────────────


def test_parse_snapshot_groups_credentials_by_purpose() -> None:
    snapshot = parse_snapshot(_SNAPSHOT_PAYLOAD)

    assert snapshot.version == 42
    chat = snapshot.credentials_for("CHAT")
    assert len(chat) == 1
    assert chat[0].id == "11111111-1111-1111-1111-111111111111"
    assert chat[0].revision == 3
    assert chat[0].provider == "openai"
    assert chat[0].model_name == "gpt-4o-mini"
    assert chat[0].api_base_url == "https://api.openai.com/v1"
    assert chat[0].api_key == "sk-real-secret-value"
    assert chat[0].priority == 1
    assert chat[0].max_rpm == 500
    assert snapshot.credentials_for("EXTRACTION") == ()


def test_parse_snapshot_missing_purpose_returns_empty_tuple() -> None:
    payload = {**_SNAPSHOT_PAYLOAD, "purposes": {"CHAT": _SNAPSHOT_PAYLOAD["purposes"]["CHAT"]}}
    snapshot = parse_snapshot(payload)

    assert snapshot.credentials_for("EMBEDDING") == ()
    assert snapshot.credentials_for("SOME_UNKNOWN_PURPOSE") == ()


def test_parse_snapshot_embedding_index_identity_present() -> None:
    snapshot = parse_snapshot(_SNAPSHOT_PAYLOAD)

    identity = snapshot.embedding_index_identity
    assert identity is not None
    assert identity.provider == "openai"
    assert identity.api_base_url == "https://api.openai.com/v1"
    assert identity.dimension == 2
    assert identity.fingerprint == ((0.1, 0.2), (0.3, 0.4), (0.5, 0.6))


def test_parse_snapshot_embedding_index_identity_null_when_absent() -> None:
    payload = {**_SNAPSHOT_PAYLOAD, "embeddingIndexIdentity": None}
    snapshot = parse_snapshot(payload)

    assert snapshot.embedding_index_identity is None


# ── redaction: never leak the key via repr()/str() ─────────────────────────


def test_credential_repr_never_contains_api_key() -> None:
    snapshot = parse_snapshot(_SNAPSHOT_PAYLOAD)
    chat_credential = snapshot.credentials_for("CHAT")[0]

    assert "sk-real-secret-value" not in repr(chat_credential)
    assert "sk-real-secret-value" not in str(chat_credential)


def test_snapshot_repr_never_contains_any_api_key() -> None:
    snapshot = parse_snapshot(_SNAPSHOT_PAYLOAD)

    assert "sk-real-secret-value" not in repr(snapshot)
    assert "sk-embed-secret" not in repr(snapshot)
    assert "sk-real-secret-value" not in str(snapshot)


# ── init_model_registry: startup gate ───────────────────────────────────────


@pytest.mark.asyncio
async def test_init_model_registry_noop_when_flag_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(model_registry.settings, "MODEL_REGISTRY_ENABLED", False)

    def handler(_request: httpx.Request) -> httpx.Response:
        raise AssertionError("must not call Java when MODEL_REGISTRY_ENABLED=false")

    result = await init_model_registry(client=_client_with(handler))

    assert result is None
    assert get_current_snapshot() is None


@pytest.mark.asyncio
async def test_init_model_registry_loads_and_caches_snapshot(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(model_registry.settings, "MODEL_REGISTRY_ENABLED", True)

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_SNAPSHOT_PAYLOAD)

    result = await init_model_registry(client=_client_with(handler))

    assert result is not None
    assert result.version == 42
    assert get_current_snapshot() is result


@pytest.mark.asyncio
async def test_init_model_registry_raises_when_no_active_chat_credential(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(model_registry.settings, "MODEL_REGISTRY_ENABLED", True)
    payload = {**_SNAPSHOT_PAYLOAD, "purposes": {"CHAT": [], "EMBEDDING": [], "EXTRACTION": []}}

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=payload)

    with pytest.raises(ModelRegistryError):
        await init_model_registry(client=_client_with(handler))

    # A failed load must not leave a stale/partial snapshot cached.
    assert get_current_snapshot() is None


@pytest.mark.asyncio
async def test_init_model_registry_raises_when_chat_purpose_missing_entirely(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(model_registry.settings, "MODEL_REGISTRY_ENABLED", True)
    payload = {**_SNAPSHOT_PAYLOAD, "purposes": {"EMBEDDING": [], "EXTRACTION": []}}

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=payload)

    with pytest.raises(ModelRegistryError):
        await init_model_registry(client=_client_with(handler))
