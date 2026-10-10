"""Every provider call site actually goes through `PinnedNetworkBackend`/
`PinnedNetworkBackendSync` at request time - todo.md Task 0.6's "Test cho từng
đường gọi" item.

`tests/core/test_no_raw_provider_clients.py` already proves, by AST scan, that
none of these modules *construct* a raw `OpenAI(...)`/`httpx.Client(...)`
outside the factory. This file is the structural test that same acceptance
criterion explicitly says isn't enough on its own: it drives one real (network-
attempting) call through each of the three call sites named in plan.md/todo.md
- `OpenAIEmbedder`, `MultiRepresentationEnricher`, `get_graph_models()` - and
counts `connect_tcp` invocations on the pinned backend classes themselves,
so a future regression that swaps in some other transport still passes the
AST scan (if it's clever about it) but fails here because nothing was ever
recorded.

`connect_tcp` is made to raise immediately after recording the call (no real
socket ever opens) - these are the same call sites already covered by mocked-
LLM tests elsewhere; this file only cares whether the attempted connection
went through the pinned backend, not what a real provider would answer.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import MagicMock

import pytest

import app.core.registry.embedding_identity as embedding_identity
import app.core.registry.model_registry as model_registry
from app.api.deps import get_graph_models
from app.core.registry.model_registry import parse_snapshot
from app.core.security.ssrf_guard import PinnedNetworkBackend, PinnedNetworkBackendSync
from app.rag.embeddings.openai_embedder import OpenAIEmbedder
from app.rag.enrichment.multi_representation import MultiRepresentationEnricher
from app.rag.vectorstore import qdrant_store
from app.schemas.ingestion import Chunk, RegionType


def _credential(purpose_id: str, model_name: str) -> dict[str, Any]:
    return {
        "id": purpose_id,
        "revision": 1,
        "sourceType": "CLOUD_API",
        "provider": "openai",
        "modelName": model_name,
        "apiBaseUrl": "https://api.openai.com/v1",
        "apiKey": "sk-test-not-real",
        "priority": 1,
        "maxRpm": 500,
    }


_SNAPSHOT_PAYLOAD: dict[str, Any] = {
    "version": 1,
    "generatedAt": "2026-09-25T03:00:00Z",
    "purposes": {
        "CHAT": [_credential("chat-cred", "gpt-4o-mini")],
        "EMBEDDING": [_credential("embed-cred", "text-embedding-3-small")],
        "EXTRACTION": [_credential("extract-cred", "gpt-4o-mini")],
    },
    "embeddingIndexIdentity": None,
}


@pytest.fixture(autouse=True)
def _registry_snapshot_with_every_purpose() -> Any:
    """Every provider call site this file drives a real request through
    (`OpenAIEmbedder`, `MultiRepresentationEnricher`, `get_graph_models()`) now resolves its
    model/key/base URL from the registry snapshot, not `.env` - so this file needs one loaded,
    all pointed at `api.openai.com` to match the `connect_tcp` host assertions below."""

    model_registry._current_snapshot = parse_snapshot(_SNAPSHOT_PAYLOAD)
    yield
    model_registry._current_snapshot = None


@pytest.fixture(autouse=True)
def _empty_qdrant_collection_no_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    """`OpenAIEmbedder` now runs the embedding identity guard
    (`app.core.registry.embedding_identity.ensure_embedding_identity`) before it ever calls the
    provider - which otherwise talks to whatever real Qdrant collection happens to be configured
    in this environment. This file only cares about the provider call site reaching the pinned
    network backend, not about identity-guard behavior (that's
    `tests/rag/test_embedding_identity_guard.py`), so it fakes an empty collection with no
    identity registered yet: the guard's bootstrap
    path then measures a fingerprint by calling `embed_probe` - the real provider call - which is
    exactly the call this file's `connect_tcp` spies are watching for."""

    monkeypatch.setattr(qdrant_store, "get_client", lambda: MagicMock())
    monkeypatch.setattr(qdrant_store, "collection_has_points", lambda client: False)
    monkeypatch.setattr(qdrant_store, "get_collection_dimension", lambda client: None)
    # The guard's "already verified" cache is process-global and keyed by
    # (credential.id, revision, snapshot.version) - this file reuses the same ids/version
    # across runs, so a prior test (in this file or elsewhere in the suite) having already
    # passed the guard for "embed-cred" would let this test's embed() skip straight past the
    # provider call the connect_tcp spy is watching for.
    embedding_identity.reset_verified_cache_for_tests()


@pytest.fixture
def sync_connect_tcp_spy(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, int]]:
    calls: list[tuple[str, int]] = []

    def _spy(self: PinnedNetworkBackendSync, host: str, port: int, **kwargs: object) -> object:
        calls.append((host, port))
        raise RuntimeError("blocked by test spy - no real socket opened")

    monkeypatch.setattr(PinnedNetworkBackendSync, "connect_tcp", _spy)
    return calls


@pytest.fixture
def async_connect_tcp_spy(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, int]]:
    calls: list[tuple[str, int]] = []

    async def _spy(self: PinnedNetworkBackend, host: str, port: int, **kwargs: object) -> object:
        calls.append((host, port))
        raise RuntimeError("blocked by test spy - no real socket opened")

    monkeypatch.setattr(PinnedNetworkBackend, "connect_tcp", _spy)
    return calls


def test_embedder_call_site_uses_pinned_backend_sync(
    sync_connect_tcp_spy: list[tuple[str, int]],
) -> None:
    embedder = OpenAIEmbedder()
    # openai SDK wraps the spy's RuntimeError - exact wrapper type isn't the point.
    with pytest.raises(Exception):  # noqa: B017
        embedder.embed(["hello"])

    assert sync_connect_tcp_spy, (
        "OpenAIEmbedder.embed() never reached PinnedNetworkBackendSync.connect_tcp"
    )
    assert all(host == "api.openai.com" for host, _ in sync_connect_tcp_spy)


def test_multi_representation_call_site_uses_pinned_backend_async(
    async_connect_tcp_spy: list[tuple[str, int]],
) -> None:
    """`MultiRepresentationEnricher.enrich()` now goes through `pydantic_ai.Agent`
    (`app.core.llm.provider_models.build_model()`, the same provider-agnostic factory
    CHAT uses - see that module's docstring) instead of a raw sync OpenAI SDK client, so
    this reaches the ASYNC pinned backend, same as `get_graph_models()` below - not the
    sync one `OpenAIEmbedder` still uses."""

    enricher = MultiRepresentationEnricher()
    chunk = Chunk(
        chunk_index=0,
        content="some chunk content",
        region_type=RegionType.TEXT,
        page_start=1,
        page_end=1,
    )
    with pytest.raises(Exception):  # noqa: B017
        enricher.enrich(chunk)

    assert async_connect_tcp_spy, (
        "MultiRepresentationEnricher.enrich() never reached PinnedNetworkBackend.connect_tcp"
    )
    assert all(host == "api.openai.com" for host, _ in async_connect_tcp_spy)


def test_graph_model_call_site_uses_pinned_backend_async(
    async_connect_tcp_spy: list[tuple[str, int]],
) -> None:
    graph_models = asyncio.run(get_graph_models())
    # Reach through pydantic_ai's OpenAIChatModel -> OpenAIProvider -> AsyncOpenAI down to
    # the exact httpx.AsyncClient get_graph_models() built via build_provider_http_client()
    # - not a new one - and issue one real request through it, proving that object (not
    # just "a" PinnedNetworkBackend somewhere) is what's wired to
    # app.api.deps.get_graph_models()'s classification/query_transformation/generation models.
    async_openai_client = graph_models.classification.provider.client
    httpx_client = async_openai_client._client

    async def _make_one_request() -> None:
        async with httpx_client:
            await httpx_client.get("/models")

    with pytest.raises(Exception):  # noqa: B017
        asyncio.run(_make_one_request())

    assert async_connect_tcp_spy, (
        "get_graph_models() client never reached PinnedNetworkBackend.connect_tcp"
    )
    assert all(host == "api.openai.com" for host, _ in async_connect_tcp_spy)
