"""WebSearchNode - searches the university's own sites (`TAVILY_INCLUDE_DOMAINS`) for the
advisory tasks, and hands the web judge (`web_judge.py`) the pages worth reading.

A request is the keyword query built from the task's `WebSlots` (app/rag/web_query.py),
or - when classification wrote no slots - the question of a sub-query rerank left empty.
Requests are searched in parallel, each through the Redis result cache. Tavily's score only
drops junk (`CHAT_WEB_SEARCH_SCORE_FLOOR`): it rates a question-form query's right page 0.08
and a keyword query's off-topic page 0.90, so relevance is the judge's call. The pages left
are ordered by the topic's own sites, the newest academic year, then score, and every one
of them goes to the judge (searches shared out round-robin): no page is lost for its rank.

With no judge (no RERANK/EXTRACTION model, or it failed), `select_without_judge` keeps
the rule from before it existed: pages scoring at least `CHAT_WEB_SEARCH_MIN_SCORE`.

No LLM call happens here. A search that fails just contributes no results - the turn goes
on, and ends in the ticket fallback if nothing else was found. The failure is still
returned (`WebSearchOutcome.failure`) so an AI admin is told why the website wasn't used.
"""

import asyncio
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from app.core.config import settings
from app.core.timezone import now_ict
from app.integrations.tavily_client import (
    TavilyClient,
    WebSearchUnavailableError,
    not_configured_error,
)
from app.integrations.web_search_cache import WebSearchCache, default_cache
from app.rag.web_query import cache_ttl_seconds, order_candidates
from app.schemas.web_search import WebSearchResult

logger = logging.getLogger(__name__)

_TRUNCATION_MARK = "…"


class WebSearchClient(Protocol):
    async def search(self, query: str, *, max_results: int) -> list[WebSearchResult]: ...


@dataclass(frozen=True)
class WebSearchRequest:
    query: str
    # The task's first web topic: picks the sites tried first and the cache TTL.
    topic: str | None = None
    # The campus the question names, if any: other campuses' sites are left out.
    co_so: str | None = None


@dataclass(frozen=True)
class WebSearchOutcome:
    """`results` are the candidates for the web judge, best first; `failure` is the first
    search error of the turn (None when every search answered), for the AI-admin warning."""

    results: list[WebSearchResult]
    failure: WebSearchUnavailableError | None = None
    cache_hits: int = 0


async def search_web(
    requests: Sequence[WebSearchRequest | str],
    *,
    client: WebSearchClient | None = None,
    cache: WebSearchCache | None = None,
    use_cache: bool = True,
) -> WebSearchOutcome:
    """Candidate pages for `requests` (most-needed first). Only the first
    `CHAT_WEB_SEARCH_MAX_QUERIES` are searched."""

    if not requests or not settings.CHAT_WEB_SEARCH_ENABLED:
        return WebSearchOutcome(results=[])
    if not settings.TAVILY_API_KEY:
        not_configured = not_configured_error()
        logger.warning("Web search skipped: %s", not_configured)
        return WebSearchOutcome(results=[], failure=not_configured)

    search_client = client or TavilyClient()
    result_cache = (cache or default_cache()) if use_cache else None
    searched = [
        request if isinstance(request, WebSearchRequest) else WebSearchRequest(query=request)
        for request in requests[: settings.CHAT_WEB_SEARCH_MAX_QUERIES]
    ]
    outcomes = await asyncio.gather(
        *(_search_one(search_client, result_cache, request) for request in searched)
    )
    per_query = [results for results, _failure, _hit in outcomes]
    failure = next((failure for _results, failure, _hit in outcomes if failure is not None), None)
    candidates = _round_robin(per_query)
    cache_hits = sum(1 for _results, _failure, hit in outcomes if hit)
    logger.info(
        "Web search: %d of %d request(s) searched (%d from cache), %d candidate page(s)",
        len(searched),
        len(requests),
        cache_hits,
        len(candidates),
    )
    return WebSearchOutcome(
        results=[_truncated(result) for result in candidates],
        failure=failure,
        cache_hits=cache_hits,
    )


def select_without_judge(candidates: Sequence[WebSearchResult]) -> list[WebSearchResult]:
    """The pages used when the web judge can't run: the rule from before it existed."""

    kept = [result for result in candidates if result.score >= settings.CHAT_WEB_SEARCH_MIN_SCORE]
    kept.sort(key=lambda result: result.score, reverse=True)
    return kept[: settings.CHAT_WEB_SEARCH_MAX_RESULTS_PER_TURN]


async def _search_one(
    client: WebSearchClient, cache: WebSearchCache | None, request: WebSearchRequest
) -> tuple[list[WebSearchResult], WebSearchUnavailableError | None, bool]:
    max_results = settings.CHAT_WEB_SEARCH_MAX_RESULTS_PER_SUB
    results = await cache.get(request.query, max_results=max_results) if cache else None
    cache_hit = results is not None
    if results is None:
        try:
            results = await client.search(request.query, max_results=max_results)
        except WebSearchUnavailableError as exc:
            logger.warning(
                "Web search failed for query=%r (%s) - continuing without it: %s",
                request.query,
                exc.code,
                exc,
            )
            return [], exc, False
        if cache is not None:
            await cache.set(
                request.query,
                results,
                max_results=max_results,
                ttl_seconds=cache_ttl_seconds(
                    request.topic, today=now_ict().date(), empty=not results
                ),
            )
    candidates = order_candidates(
        results,
        topic=request.topic,
        co_so=request.co_so,
        floor=settings.CHAT_WEB_SEARCH_SCORE_FLOOR,
    )
    # Always logged: every page with its score, and which ones are shown to the judge -
    # enough to tune the domain policy from production logs.
    logger.info(
        "Web search query=%r topic=%s cache=%s: %d page(s) found, %d candidate(s) [%s]",
        request.query,
        request.topic,
        "hit" if cache_hit else ("miss" if cache else "off"),
        len(results),
        len(candidates),
        ", ".join(f"{result.score:.2f} {result.url}" for result in results),
    )
    if settings.APP_DEBUG:
        logger.info(
            "Web search query=%r page contents:\n%s",
            request.query,
            "\n".join(f"[{result.score:.2f}] {result.url}\n{result.content}" for result in results),
        )
    return candidates, None, cache_hit


def _round_robin(per_query: Sequence[list[WebSearchResult]]) -> list[WebSearchResult]:
    """Each search's best page not already taken, then each one's next, ... - a search
    whose best page another search already took still gets one of its own early."""

    seen_urls: set[str] = set()
    merged: list[WebSearchResult] = []
    remaining = [list(results) for results in per_query]
    while any(remaining):
        for results in remaining:
            while results and results[0].url in seen_urls:
                results.pop(0)
            if results:
                page = results.pop(0)
                merged.append(page)
                seen_urls.add(page.url)
    return merged


def _truncated(result: WebSearchResult) -> WebSearchResult:
    max_chars = settings.CHAT_WEB_SEARCH_RESULT_MAX_CHARS
    if len(result.content) <= max_chars:
        return result
    content = result.content[:max_chars].rstrip() + _TRUNCATION_MARK
    return result.model_copy(update={"content": content})
