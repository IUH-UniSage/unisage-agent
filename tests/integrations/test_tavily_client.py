"""`TavilyClient` against an `httpx.MockTransport` - no real network call."""

import asyncio
import json

import httpx
import pytest

from app.core.config import settings
from app.integrations.tavily_client import TavilyClient, WebSearchUnavailableError


@pytest.fixture(autouse=True)
def _tavily_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "TAVILY_API_KEY", "tvly-secret")
    monkeypatch.setattr(settings, "TAVILY_BASE_URL", "https://tavily.test")
    monkeypatch.setattr(settings, "TAVILY_INCLUDE_DOMAINS", ["iuh.edu.vn"])
    monkeypatch.setattr(settings, "TAVILY_SEARCH_DEPTH", "basic")


def _client(handler: httpx.MockTransport) -> TavilyClient:
    return TavilyClient(transport=handler)


@pytest.mark.asyncio
async def test_search_sends_domain_restricted_request_and_parses_results() -> None:
    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(
            200,
            json={
                "query": "lịch thi",
                "results": [
                    {
                        "title": "Lịch thi HK1",
                        "url": "https://pdt.iuh.edu.vn/lich-thi",
                        "content": "Lịch thi học kỳ 1...",
                        "score": 0.82,
                        "raw_content": None,
                    }
                ],
            },
        )

    results = await _client(httpx.MockTransport(handler)).search("lịch thi", max_results=2)

    request = captured[0]
    assert str(request.url) == "https://tavily.test/search"
    assert request.headers["Authorization"] == "Bearer tvly-secret"
    body = json.loads(request.content)
    assert body["query"] == "lịch thi"
    assert body["max_results"] == 2
    assert body["include_domains"] == ["iuh.edu.vn"]
    assert body["include_raw_content"] is False
    assert [result.url for result in results] == ["https://pdt.iuh.edu.vn/lich-thi"]
    assert results[0].score == 0.82


@pytest.mark.asyncio
async def test_missing_api_key_raises_without_calling_tavily(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "TAVILY_API_KEY", "")

    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("must not call Tavily without a key")

    with pytest.raises(WebSearchUnavailableError, match="TAVILY_API_KEY"):
        await _client(httpx.MockTransport(handler)).search("q", max_results=2)


@pytest.mark.parametrize("status", [401, 429, 500])
@pytest.mark.asyncio
async def test_http_error_status_raises(status: int) -> None:
    transport = httpx.MockTransport(lambda _request: httpx.Response(status, json={}))

    with pytest.raises(WebSearchUnavailableError, match=f"HTTP {status}") as exc_info:
        await _client(transport).search("q", max_results=2)

    assert "tvly-secret" not in str(exc_info.value)


@pytest.mark.parametrize(
    ("error", "message"),
    [
        (httpx.ReadTimeout("slow"), "timed out"),
        (httpx.ConnectError("refused"), "network error"),
    ],
)
@pytest.mark.asyncio
async def test_transport_errors_raise(error: Exception, message: str) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise error

    with pytest.raises(WebSearchUnavailableError, match=message):
        await _client(httpx.MockTransport(handler)).search("q", max_results=2)


@pytest.mark.parametrize(
    "body",
    [b"not json", b'{"no_results": []}', b'{"results": [{"title": "x"}]}'],
)
@pytest.mark.asyncio
async def test_unexpected_body_raises(body: bytes) -> None:
    transport = httpx.MockTransport(lambda _request: httpx.Response(200, content=body))

    with pytest.raises(WebSearchUnavailableError, match="unexpected response body"):
        await _client(transport).search("q", max_results=2)


@pytest.mark.asyncio
async def test_slow_tavily_is_cut_off_at_one_overall_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "TAVILY_TIMEOUT_SECONDS", 0.05)

    async def slow(request: httpx.Request) -> httpx.Response:
        await asyncio.sleep(1)
        return httpx.Response(200, json={"results": []})

    with pytest.raises(WebSearchUnavailableError, match="timed out"):
        await _client(httpx.MockTransport(slow)).search("q", max_results=2)


@pytest.mark.asyncio
async def test_a_runaway_query_is_cut_before_it_reaches_tavily() -> None:
    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(200, json={"results": []})

    await _client(httpx.MockTransport(handler)).search("x" * 1000, max_results=2)

    assert len(json.loads(captured[0].content)["query"]) == 400
