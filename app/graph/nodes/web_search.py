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
The failure is still returned (`WebSearchOutcome.failure`) so an AI admin is
told why the website wasn't used.
"""

import asyncio
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from app.core.config import settings
from app.integrations.tavily_client import (
    TavilyClient,
    WebSearchUnavailableError,
    not_configured_error,
)
from app.schemas.web_search import WebSearchResult

logger = logging.getLogger(__name__)

_TRUNCATION_MARK = "…"


class WebSearchClient(Protocol):
    async def search(self, query: str, *, max_results: int) -> list[WebSearchResult]: ...


@dataclass(frozen=True)
class WebSearchOutcome:
    """`results` go into the prompt; `failure` is the first search error of the
    turn (None when every search answered), for the AI-admin warning."""

    results: list[WebSearchResult]
    failure: WebSearchUnavailableError | None = None


async def search_web(
    queries: Sequence[str], *, client: WebSearchClient | None = None
) -> WebSearchOutcome:
    """Web pages for `queries` (the failed sub-queries, most-needed first),
    best score first. Only the first `CHAT_WEB_SEARCH_MAX_QUERIES` are searched."""

    if not queries or not settings.CHAT_WEB_SEARCH_ENABLED:
        return WebSearchOutcome(results=[])
    if not settings.TAVILY_API_KEY:
        not_configured = not_configured_error()
        logger.warning("Web search skipped: %s", not_configured)
        return WebSearchOutcome(results=[], failure=not_configured)

    search_client = client or TavilyClient()
    searched = queries[: settings.CHAT_WEB_SEARCH_MAX_QUERIES]
    outcomes = await asyncio.gather(*(_search_one(search_client, query) for query in searched))
    per_query = [results for results, _failure in outcomes]
    failure = next((failure for _results, failure in outcomes if failure is not None), None)
    selected = _allocate(per_query, settings.CHAT_WEB_SEARCH_MAX_RESULTS_PER_TURN)
    logger.info(
        "Web search: %d of %d sub-query(ies) searched, %d page(s) kept for the prompt",
        len(searched),
        len(queries),
        len(selected),
    )
    return WebSearchOutcome(results=[_truncated(result) for result in selected], failure=failure)


async def _search_one(
    client: WebSearchClient, query: str
) -> tuple[list[WebSearchResult], WebSearchUnavailableError | None]:
    try:
        results = await client.search(
            query, max_results=settings.CHAT_WEB_SEARCH_MAX_RESULTS_PER_SUB
        )
    except WebSearchUnavailableError as exc:
        logger.warning(
            "Web search failed for query=%r (%s) - continuing without it: %s",
            query,
            exc.code,
            exc,
        )
        return [], exc
    min_score = settings.CHAT_WEB_SEARCH_MIN_SCORE
    relevant = [result for result in results if result.score >= min_score]
    # Always logged: how many pages came back and how many cleared MIN_SCORE, with each
    # page's score - enough to tune the threshold from production logs.
    logger.info(
        "Web search query=%r: %d page(s) found, %d with score >= %s [%s]",
        query,
        len(results),
        len(relevant),
        min_score,
        ", ".join(f"{result.score:.2f} {result.url}" for result in results),
    )
    if settings.APP_DEBUG:
        logger.info(
            "Web search query=%r page contents:\n%s",
            query,
            "\n".join(f"[{result.score:.2f}] {result.url}\n{result.content}" for result in results),
        )
    return sorted(relevant, key=lambda result: result.score, reverse=True), None


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
