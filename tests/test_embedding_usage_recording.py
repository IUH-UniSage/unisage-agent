"""Usage-recording tests for Embedding: one `embed()` call is one
`purpose=EMBEDDING` business request. Exercises `OpenAIEmbedder` with a
registry-resolved credential already populated in `_resolved` (bypassing
`_resolve_from_registry`'s identity-guard chain, which has its own dedicated
tests elsewhere) - `embed()`'s own logic for building/closing a UsageRecorder
around `_call_provider` is what's under test here."""

import json
from typing import Any
from unittest.mock import MagicMock

import pytest

import app.core.usage.usage_outbox as usage_outbox_module
from app.core.registry.model_registry import CredentialConfig
from app.rag.embeddings.openai_embedder import OpenAIEmbedder


def _credential() -> CredentialConfig:
    return CredentialConfig(
        id="embed-cred",
        revision=1,
        source_type="CLOUD_API",
        provider="openai",
        model_name="text-embedding-3-small",
        api_base_url="https://api.openai.com/v1",
        priority=1,
        max_rpm=None,
        api_key="sk-test",
    )


def _mock_client(vectors: list[list[float]], *, prompt_tokens: int = 42) -> MagicMock:
    client = MagicMock()
    client.embeddings.create.return_value = MagicMock(
        data=[MagicMock(embedding=vector) for vector in vectors],
        usage=MagicMock(prompt_tokens=prompt_tokens),
    )
    return client


@pytest.fixture
def captured_outbox(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    captured: list[dict[str, Any]] = []

    async def _fake_enqueue(payload: dict[str, Any], *, redis_client: Any = None) -> None:
        del redis_client
        captured.append(json.loads(json.dumps(payload)))

    monkeypatch.setattr(usage_outbox_module, "enqueue_usage_payload", _fake_enqueue)
    return captured


def _resolved_embedder(client: MagicMock) -> OpenAIEmbedder:
    embedder = OpenAIEmbedder()
    embedder._resolved["model"] = "text-embedding-3-small"
    embedder._resolved["client"] = client
    embedder._resolved["credential"] = _credential()
    return embedder


def test_embed_records_one_line_with_purpose_embedding_and_no_message_ids(
    captured_outbox: list[dict[str, Any]],
) -> None:
    client = _mock_client([[0.1, 0.2]], prompt_tokens=42)
    embedder = _resolved_embedder(client)

    vectors = embedder.embed(["some chunk text"])

    assert vectors == [[0.1, 0.2]]
    assert len(captured_outbox) == 1
    payload = captured_outbox[0]
    assert payload["purpose"] == "EMBEDDING"
    assert payload["conversationId"] is None
    assert payload["userMessageId"] is None
    assert payload["assistantMessageId"] is None
    assert payload["status"] == "SUCCESS"

    lines = payload["lines"]
    assert len(lines) == 1
    assert lines[0]["nodeName"] == "embed_batch"
    assert lines[0]["inputTokens"] == 42
    assert lines[0]["chatModelId"] == "embed-cred"
    assert lines[0]["provider"] == "openai"


def test_embed_failure_records_error_line_and_still_raises(
    captured_outbox: list[dict[str, Any]],
) -> None:
    from app.core.errors.llm_error_classifier import EmbeddingProviderError

    client = MagicMock()
    client.embeddings.create.side_effect = RuntimeError("provider down")
    embedder = _resolved_embedder(client)

    with pytest.raises(EmbeddingProviderError):
        embedder.embed(["some chunk text"])

    assert len(captured_outbox) == 1
    payload = captured_outbox[0]
    assert payload["status"] == "ERROR"
    assert payload["lines"][0]["status"] == "ERROR"
    assert payload["lines"][0]["errorCode"] == "RuntimeError"


def test_embed_with_injected_client_and_no_credential_records_nothing(
    captured_outbox: list[dict[str, Any]],
) -> None:
    """A test double with model/client injected directly (no registry credential
    resolved) must not create a usage log at all - matches the identity-probe
    call inside `_resolve_from_registry`, which never passes a recorder either."""

    client = _mock_client([[0.1, 0.2]])
    embedder = OpenAIEmbedder(model="text-embedding-3-small", client=client)

    embedder.embed(["some chunk text"])

    assert captured_outbox == []


def test_embed_empty_texts_records_nothing(captured_outbox: list[dict[str, Any]]) -> None:
    client = _mock_client([])
    embedder = _resolved_embedder(client)

    assert embedder.embed([]) == []
    assert captured_outbox == []
