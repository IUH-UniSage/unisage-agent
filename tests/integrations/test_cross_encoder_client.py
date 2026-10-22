"""CrossEncoderClient against `httpx.MockTransport`: the contract, and every
failure as `CrossEncoderUnavailableError` with a Vietnamese message."""

import json

import httpx
import pytest

from app.integrations.cross_encoder_client import (
    CrossEncoderClient,
    CrossEncoderUnavailableError,
)


def _client(handler) -> CrossEncoderClient:
    return CrossEncoderClient(
        base_url="http://ce.test", timeout_seconds=0.5, transport=httpx.MockTransport(handler)
    )


@pytest.mark.asyncio
async def test_rerank_posts_the_pairs_and_returns_scores_in_passage_order() -> None:
    seen: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        assert request.url.path == "/rerank"
        return httpx.Response(
            200,
            json={
                "scores": [0.1, 2.5],
                "ranking": [1, 0],
                "truncated": 1,
                "took_ms": 7,
                "model": "BAAI/bge-reranker-v2-m3",
            },
        )

    result = await _client(handler).rerank("q", ["p1", "p2"], top_k=1)

    assert seen == [{"query": "q", "passages": ["p1", "p2"], "top_k": 1}]
    assert result.scores == [0.1, 2.5] and result.ranking == [1, 0]
    assert result.truncated == 1 and result.took_ms == 7
    assert result.model == "BAAI/bge-reranker-v2-m3"


@pytest.mark.asyncio
async def test_no_passages_does_not_call_the_service() -> None:
    def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover
        raise AssertionError("must not be called")

    result = await _client(handler).rerank("q", [])

    assert result.scores == [] and result.ranking == []


@pytest.mark.asyncio
async def test_connection_error_names_the_url_and_the_serve_command() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    with pytest.raises(CrossEncoderUnavailableError) as info:
        await _client(handler).rerank("q", ["p"])

    assert "http://ce.test" in str(info.value) and "task reranker:serve" in str(info.value)


@pytest.mark.asyncio
async def test_timeout_names_the_setting() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow", request=request)

    with pytest.raises(CrossEncoderUnavailableError) as info:
        await _client(handler).rerank("q", ["p"])

    assert "CROSS_ENCODER_TIMEOUT_SECONDS" in str(info.value)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(500, text="boom"),
        httpx.Response(200, text="not json"),
        httpx.Response(200, json={"scores": [1.0], "ranking": [0]}),  # 1 score for 2 passages
    ],
)
async def test_bad_responses_are_unavailable_errors(response: httpx.Response) -> None:
    with pytest.raises(CrossEncoderUnavailableError):
        await _client(lambda request: response).rerank("q", ["p1", "p2"])


@pytest.mark.asyncio
async def test_health_returns_the_body() -> None:
    body = {"status": "ok", "loaded": True, "device": "cuda"}

    assert await _client(lambda r: httpx.Response(200, json=body)).health() == body
