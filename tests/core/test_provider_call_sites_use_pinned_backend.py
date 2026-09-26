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

import pytest

from app.api.deps import get_graph_models
from app.core.ssrf_guard import PinnedNetworkBackend, PinnedNetworkBackendSync
from app.rag.embeddings.openai_embedder import OpenAIEmbedder
from app.rag.enrichment.multi_representation import MultiRepresentationEnricher
from app.schemas.ingestion import Chunk, RegionType


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


def test_multi_representation_call_site_uses_pinned_backend_sync(
    sync_connect_tcp_spy: list[tuple[str, int]],
) -> None:
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

    assert sync_connect_tcp_spy, (
        "MultiRepresentationEnricher.enrich() never reached PinnedNetworkBackendSync.connect_tcp"
    )
    assert all(host == "api.openai.com" for host, _ in sync_connect_tcp_spy)


def test_graph_model_call_site_uses_pinned_backend_async(
    async_connect_tcp_spy: list[tuple[str, int]],
) -> None:
    graph_models = get_graph_models()
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
