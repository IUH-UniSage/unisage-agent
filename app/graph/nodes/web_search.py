"""WebSearchNode - last resort before TicketFallbackNode.

Runs only for the sub-queries rerank left with no chunk. Each is searched on
the university's own sites (`TAVILY_INCLUDE_DOMAINS`) in parallel; results
under `CHAT_WEB_SEARCH_MIN_SCORE` are dropped and the rest are shared out
round-robin within `CHAT_WEB_SEARCH_MAX_RESULTS_PER_TURN`: every failed
sub-query first gets its own best page (so none is left without a source and
answered from nothing), then the remaining slots go to the best-scoring pages
overall.

No LLM call happens here. A search that fails just contributes no results -
the turn goes on, and ends in the ticket fallback if nothing else was found.
"""

import asyncio
import logging
from collections.abc import Sequence
from typing import Protocol

from app.core.config import settings
from app.integrations.tavily_client import TavilyClient, WebSearchUnavailableError
from app.schemas.web_search import WebSearchResult

logger = logging.getLogger(__name__)

_TRUNCATION_MARK = "…"


class WebSearchClient(Protocol):
    async def search(self, query: str, *, max_results: int) -> list[WebSearchResult]: ...


async def search_web(
    queries: Sequence[str], *, client: WebSearchClient | None = None
) -> list[WebSearchResult]:
    """Web pages for `queries` (the failed sub-queries), best score first."""

    if not queries or not settings.CHAT_WEB_SEARCH_ENABLED:
        return []
    if not settings.TAVILY_API_KEY:
        logger.warning("Web search is enabled but TAVILY_API_KEY is empty - skipping it")
        return []

    search_client = client or TavilyClient()
    per_query = await asyncio.gather(*(_search_one(search_client, query) for query in queries))
    selected = _allocate(per_query, settings.CHAT_WEB_SEARCH_MAX_RESULTS_PER_TURN)
    return [_truncated(result) for result in selected]


async def _search_one(client: WebSearchClient, query: str) -> list[WebSearchResult]:
    try:
        results = await client.search(
            query, max_results=settings.CHAT_WEB_SEARCH_MAX_RESULTS_PER_SUB
        )
    except WebSearchUnavailableError as exc:
        logger.warning("Web search failed for a sub-query - continuing without it: %s", exc)
        return []
    if settings.APP_DEBUG:
        dump = "\n".join(f"[{result.score:.2f}] {result.url}\n{result.content}" for result in results)
        logger.info(
            "web search query=%r min_score=%s raw_results=%d:\n%s",
            query,
            settings.CHAT_WEB_SEARCH_MIN_SCORE,
            len(results),
            dump,
        )
    relevant = [result for result in results if result.score >= settings.CHAT_WEB_SEARCH_MIN_SCORE]
    return sorted(relevant, key=lambda result: result.score, reverse=True)


def _allocate(per_query: Sequence[list[WebSearchResult]], limit: int) -> list[WebSearchResult]:
    seen_urls: set[str] = set()

    # Round 1: each sub-query's own best page not already picked by an earlier one.
    heads: list[WebSearchResult] = []
    for results in per_query:
        head = next((result for result in results if result.url not in seen_urls), None)
        if head is not None:
            heads.append(head)
            seen_urls.add(head.url)
    # More failed sub-queries than slots: the best-scoring heads win.
    selected = sorted(heads, key=lambda result: result.score, reverse=True)[:limit]
    seen_urls = {result.url for result in selected}

    # Round 2: fill the remaining slots with the best pages overall.
    pool = sorted(
        (result for results in per_query for result in results),
        key=lambda result: result.score,
        reverse=True,
    )
    for result in pool:
        if len(selected) >= limit:
            break
        if result.url not in seen_urls:
            selected.append(result)
            seen_urls.add(result.url)

    return sorted(selected, key=lambda result: result.score, reverse=True)


def _truncated(result: WebSearchResult) -> WebSearchResult:
    max_chars = settings.CHAT_WEB_SEARCH_RESULT_MAX_CHARS
    if len(result.content) <= max_chars:
        return result
    content = result.content[:max_chars].rstrip() + _TRUNCATION_MARK
    return result.model_copy(update={"content": content})
