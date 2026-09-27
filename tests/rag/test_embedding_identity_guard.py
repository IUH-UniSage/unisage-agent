"""`app.core.embedding_identity.ensure_embedding_identity` — plan.md "Embedding identity guard",
todo.md Task 13b.

Every case that refuses (mismatch, or vectors-but-no-identity) is proven with a call-count
assertion on the probe callable, not just "an exception was raised" — the acceptance criterion is
that the provider is never even called for those cases, and a raised exception alone doesn't rule
out a provider call having already happened first.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.core.embedding_identity import (
    EmbeddingIdentityMismatchError,
    ensure_embedding_identity,
    identity_key,
    reset_verified_cache_for_tests,
)
from app.core.llm_error_classifier import EmbeddingProviderError
from app.core.model_registry import CredentialConfig, EmbeddingIndexIdentity, ModelRegistrySnapshot
from app.integrations.backend_java_client import BackendJavaHTTPError

_DIMENSION = 3
_VECTOR_A = (1.0, 0.0, 0.0)
_VECTOR_B = (0.0, 1.0, 0.0)
_VECTOR_C = (0.0, 0.0, 1.0)
_MATCHING_FINGERPRINT = (_VECTOR_A, _VECTOR_B, _VECTOR_C)
# Orthogonal to every probe vector above - cosine similarity 0.0, never matches.
_MISMATCHED_FINGERPRINT = ((0.0, 1.0, 0.0), (1.0, 0.0, 0.0), (1.0, 0.0, 0.0))


@pytest.fixture(autouse=True)
def _clear_cache() -> None:
    reset_verified_cache_for_tests()
    yield
    reset_verified_cache_for_tests()


def _credential(**overrides: Any) -> CredentialConfig:
    values: dict[str, Any] = {
        "id": "embed-cred",
        "revision": 1,
        "source_type": "CLOUD_API",
        "provider": "openai",
        "model_name": "text-embedding-3-small",
        "api_base_url": "https://api.openai.com/v1",
        "priority": 1,
        "max_rpm": 500,
        "api_key": "sk-test",
    }
    values.update(overrides)
    return CredentialConfig(**values)


def _identity(**overrides: Any) -> EmbeddingIndexIdentity:
    values: dict[str, Any] = {
        "provider": "openai",
        "model_name": "text-embedding-3-small",
        "model_source_ref": None,
        "api_base_url": "https://api.openai.com/v1",
        "dimension": _DIMENSION,
        "fingerprint": _MATCHING_FINGERPRINT,
    }
    values.update(overrides)
    return EmbeddingIndexIdentity(**values)


def _snapshot(
    identity: EmbeddingIndexIdentity | None, *, version: int = 1
) -> ModelRegistrySnapshot:
    return ModelRegistrySnapshot(
        version=version, generated_at=None, purposes={}, embedding_index_identity=identity
    )


def _probe_returning(vectors: tuple[tuple[float, ...], ...]) -> MagicMock:
    return MagicMock(return_value=[list(v) for v in vectors])


def _qdrant_client_stub(*, has_points: bool, dimension: int | None) -> MagicMock:
    client = MagicMock()
    client._fake_has_points = has_points
    client._fake_dimension = dimension
    return client


@pytest.fixture(autouse=True)
def _patch_qdrant_helpers(monkeypatch: pytest.MonkeyPatch) -> None:
    from app.rag.vectorstore import qdrant_store

    monkeypatch.setattr(
        qdrant_store, "collection_has_points", lambda client: client._fake_has_points
    )
    monkeypatch.setattr(
        qdrant_store, "get_collection_dimension", lambda client: client._fake_dimension
    )


def test_collection_has_vectors_but_no_identity_refuses_without_calling_provider() -> None:
    probe = _probe_returning(_MATCHING_FINGERPRINT)
    client = _qdrant_client_stub(has_points=True, dimension=_DIMENSION)

    with pytest.raises(EmbeddingIdentityMismatchError):
        ensure_embedding_identity(
            _credential(),
            snapshot=_snapshot(None),
            embed_probe=probe,
            qdrant_client=client,
        )

    probe.assert_not_called()


def test_provider_model_mismatch_refuses_without_calling_provider() -> None:
    probe = _probe_returning(_MATCHING_FINGERPRINT)
    client = _qdrant_client_stub(has_points=True, dimension=_DIMENSION)
    registered = _identity(model_name="text-embedding-3-large")

    with pytest.raises(EmbeddingIdentityMismatchError):
        ensure_embedding_identity(
            _credential(),
            snapshot=_snapshot(registered),
            embed_probe=probe,
            qdrant_client=client,
        )

    probe.assert_not_called()


def test_dimension_mismatch_refuses_without_calling_provider() -> None:
    probe = _probe_returning(_MATCHING_FINGERPRINT)
    client = _qdrant_client_stub(has_points=True, dimension=1536)
    registered = _identity(dimension=_DIMENSION)

    with pytest.raises(EmbeddingIdentityMismatchError):
        ensure_embedding_identity(
            _credential(),
            snapshot=_snapshot(registered),
            embed_probe=probe,
            qdrant_client=client,
        )

    probe.assert_not_called()


def test_matching_fingerprint_promotes_and_returns_identity_key() -> None:
    probe = _probe_returning(_MATCHING_FINGERPRINT)
    client = _qdrant_client_stub(has_points=True, dimension=_DIMENSION)
    registered = _identity()

    key = ensure_embedding_identity(
        _credential(), snapshot=_snapshot(registered), embed_probe=probe, qdrant_client=client
    )

    assert probe.call_count == 1
    assert key == identity_key(
        "openai", "text-embedding-3-small", None, "https://api.openai.com/v1", _DIMENSION
    )


def test_mismatched_fingerprint_refuses_after_measuring() -> None:
    """Unlike the earlier cases, a fingerprint check has to call the provider to measure it -
    the refusal happens only after comparing, not instead of measuring."""

    probe = _probe_returning(_MISMATCHED_FINGERPRINT)
    client = _qdrant_client_stub(has_points=True, dimension=_DIMENSION)
    registered = _identity()

    with pytest.raises(EmbeddingIdentityMismatchError):
        ensure_embedding_identity(
            _credential(), snapshot=_snapshot(registered), embed_probe=probe, qdrant_client=client
        )

    assert probe.call_count == 1


def test_verified_result_is_cached_across_calls_for_same_credential_and_snapshot() -> None:
    probe = _probe_returning(_MATCHING_FINGERPRINT)
    client = _qdrant_client_stub(has_points=True, dimension=_DIMENSION)
    registered = _identity()
    credential = _credential()
    snapshot = _snapshot(registered)

    first = ensure_embedding_identity(
        credential, snapshot=snapshot, embed_probe=probe, qdrant_client=client
    )
    second = ensure_embedding_identity(
        credential, snapshot=snapshot, embed_probe=probe, qdrant_client=client
    )

    assert first == second
    assert probe.call_count == 1


def test_empty_collection_bootstraps_identity_on_201() -> None:
    probe = _probe_returning(_MATCHING_FINGERPRINT)
    client = _qdrant_client_stub(has_points=False, dimension=None)
    backend_client = MagicMock()
    backend_client.put_embedding_index_identity = AsyncMock(return_value={"established": True})

    key = ensure_embedding_identity(
        _credential(),
        snapshot=_snapshot(None),
        embed_probe=probe,
        qdrant_client=client,
        backend_client=backend_client,
    )

    assert probe.call_count == 1
    backend_client.put_embedding_index_identity.assert_awaited_once()
    put_kwargs = backend_client.put_embedding_index_identity.call_args.kwargs
    assert put_kwargs["established_by"] == "first-upsert"
    assert put_kwargs["dimension"] == _DIMENSION
    assert key == identity_key(
        "openai", "text-embedding-3-small", None, "https://api.openai.com/v1", _DIMENSION
    )


def test_empty_collection_lost_race_reconciles_on_matching_winner() -> None:
    probe = _probe_returning(_MATCHING_FINGERPRINT)
    client = _qdrant_client_stub(has_points=False, dimension=None)
    backend_client = MagicMock()
    backend_client.put_embedding_index_identity = AsyncMock(
        side_effect=BackendJavaHTTPError(
            "PUT", "/x", 409, {"error": "EMBEDDING_INDEX_IDENTITY_EXISTS"}
        )
    )
    backend_client.get_embedding_index_identity = AsyncMock(
        return_value={
            "provider": "openai",
            "modelName": "text-embedding-3-small",
            "modelSourceRef": None,
            "apiBaseUrl": "https://api.openai.com/v1",
            "dimension": _DIMENSION,
            "fingerprint": [v for vec in _MATCHING_FINGERPRINT for v in vec],
        }
    )

    key = ensure_embedding_identity(
        _credential(),
        snapshot=_snapshot(None),
        embed_probe=probe,
        qdrant_client=client,
        backend_client=backend_client,
    )

    assert key == identity_key(
        "openai", "text-embedding-3-small", None, "https://api.openai.com/v1", _DIMENSION
    )


def test_empty_collection_lost_race_refuses_on_mismatched_winner() -> None:
    probe = _probe_returning(_MATCHING_FINGERPRINT)
    client = _qdrant_client_stub(has_points=False, dimension=None)
    backend_client = MagicMock()
    backend_client.put_embedding_index_identity = AsyncMock(
        side_effect=BackendJavaHTTPError(
            "PUT", "/x", 409, {"error": "EMBEDDING_INDEX_IDENTITY_EXISTS"}
        )
    )
    backend_client.get_embedding_index_identity = AsyncMock(
        return_value={
            "provider": "openai",
            "modelName": "text-embedding-3-large",
            "modelSourceRef": None,
            "apiBaseUrl": "https://api.openai.com/v1",
            "dimension": _DIMENSION,
            "fingerprint": [v for vec in _MATCHING_FINGERPRINT for v in vec],
        }
    )

    with pytest.raises(EmbeddingIdentityMismatchError):
        ensure_embedding_identity(
            _credential(),
            snapshot=_snapshot(None),
            embed_probe=probe,
            qdrant_client=client,
            backend_client=backend_client,
        )


def test_probe_failure_is_reclassified_as_embedding_provider_error() -> None:
    client = _qdrant_client_stub(has_points=True, dimension=_DIMENSION)
    registered = _identity()

    def _failing_probe(_texts: list[str]) -> list[list[float]]:
        raise RuntimeError("connection reset")

    with pytest.raises(EmbeddingProviderError):
        ensure_embedding_identity(
            _credential(),
            snapshot=_snapshot(registered),
            embed_probe=_failing_probe,
            qdrant_client=client,
        )
