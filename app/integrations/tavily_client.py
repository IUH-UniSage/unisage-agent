"""Tavily Search API client (`POST /search`) for WebSearchNode.

Like `slack_notifier`, this talks to a fixed, operator-configured host
(`settings.TAVILY_BASE_URL`), so it builds a plain `httpx.AsyncClient`
rather than going through the SSRF-guarded provider HTTP client factory.

Every failure - missing key, network error, timeout, non-2xx, unparseable
body - is raised as `WebSearchUnavailableError`. Web search is only a last
resort before the ticket fallback, so the caller logs it and carries on
without web results; it never reaches the user as an error.
"""

from typing import Any

import httpx
from pydantic import ValidationError

from app.core.config import settings
from app.schemas.web_search import WebSearchResult


class WebSearchUnavailableError(Exception):
    """The web search provider could not answer this query."""


class TavilyClient:
    def __init__(self, *, transport: httpx.AsyncBaseTransport | None = None) -> None:
        self._transport = transport

    async def search(self, query: str, *, max_results: int) -> list[WebSearchResult]:
        if not settings.TAVILY_API_KEY:
            raise WebSearchUnavailableError("TAVILY_API_KEY is not set")

        body: dict[str, Any] = {
            "query": query,
            "search_depth": settings.TAVILY_SEARCH_DEPTH,
            "max_results": max_results,
            "include_domains": settings.TAVILY_INCLUDE_DOMAINS,
            "include_answer": False,
            "include_raw_content": False,
        }
        try:
            async with httpx.AsyncClient(
                base_url=settings.TAVILY_BASE_URL,
                timeout=settings.TAVILY_TIMEOUT_SECONDS,
                transport=self._transport,
            ) as client:
                response = await client.post(
                    "/search",
                    json=body,
                    headers={"Authorization": f"Bearer {settings.TAVILY_API_KEY}"},
                )
        except httpx.TimeoutException as exc:
            raise WebSearchUnavailableError("timed out") from exc
        except httpx.RequestError as exc:
            raise WebSearchUnavailableError(f"network error: {type(exc).__name__}") from exc

        if response.status_code >= 300:
            raise WebSearchUnavailableError(f"HTTP {response.status_code}")

        try:
            raw_results = response.json()["results"]
            return [WebSearchResult.model_validate(item) for item in raw_results]
        except (ValueError, KeyError, TypeError, ValidationError) as exc:
            raise WebSearchUnavailableError("unexpected response body") from exc
