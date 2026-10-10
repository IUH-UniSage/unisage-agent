"""Tavily Search API client (`POST /search`) for WebSearchNode.

Like `slack_notifier`, this talks to a fixed, operator-configured host
(`settings.TAVILY_BASE_URL`), so it builds a plain `httpx.AsyncClient`
rather than going through the SSRF-guarded provider HTTP client factory.

Every failure - missing key, network error, timeout, non-2xx, unparseable
body - is raised as `WebSearchUnavailableError`, carrying a
`WebSearchFailureReason` and a Vietnamese message naming the cause (with
Tavily's own error text when it sent one). Web search is only a last resort
before the ticket fallback, so the turn carries on without web results; the
message only ever reaches an AI admin, as a warning next to the answer (see
`run_and_persist`).

Tavily's status codes: 401 bad key, 429 rate limit, 432 key/plan credit
limit reached, 433 pay-as-you-go spending limit reached.
"""

import asyncio
from enum import StrEnum
from typing import Any

import httpx
from pydantic import ValidationError

from app.core.config import settings
from app.schemas.web_search import WebSearchResult

# Search queries are short questions (HyDE's standalone rewrite or a decomposed
# sub-query); this only guards against a runaway rewrite - Tavily rejects very long
# queries outright.
_MAX_QUERY_CHARS = 400


_MAX_DETAIL_CHARS = 300
_CREDIT_LIMIT_STATUSES = (432, 433)


class WebSearchFailureReason(StrEnum):
    NOT_CONFIGURED = "NOT_CONFIGURED"
    AUTH_FAILED = "AUTH_FAILED"
    CREDITS_EXHAUSTED = "CREDITS_EXHAUSTED"
    RATE_LIMITED = "RATE_LIMITED"
    REQUEST_REJECTED = "REQUEST_REJECTED"
    PROVIDER_ERROR = "PROVIDER_ERROR"
    TIMEOUT = "TIMEOUT"
    CONNECTION_ERROR = "CONNECTION_ERROR"
    MALFORMED_RESPONSE = "MALFORMED_RESPONSE"


class WebSearchUnavailableError(Exception):
    """The web search provider could not answer this query. `str(exc)` is the
    admin-facing Vietnamese explanation, never containing the API key."""

    def __init__(self, reason: WebSearchFailureReason, message: str) -> None:
        super().__init__(message)
        self.reason = reason

    @property
    def code(self) -> str:
        return f"WEB_SEARCH_{self.reason.value}"


def not_configured_error() -> WebSearchUnavailableError:
    return WebSearchUnavailableError(
        WebSearchFailureReason.NOT_CONFIGURED,
        "Tìm kiếm web đang bật (CHAT_WEB_SEARCH_ENABLED) nhưng chưa cấu hình TAVILY_API_KEY.",
    )


class TavilyClient:
    def __init__(self, *, transport: httpx.AsyncBaseTransport | None = None) -> None:
        self._transport = transport

    async def search(self, query: str, *, max_results: int) -> list[WebSearchResult]:
        if not settings.TAVILY_API_KEY:
            raise not_configured_error()

        body: dict[str, Any] = {
            "query": query[:_MAX_QUERY_CHARS],
            "search_depth": settings.TAVILY_SEARCH_DEPTH,
            "max_results": max_results,
            "include_domains": settings.TAVILY_INCLUDE_DOMAINS,
            "include_answer": False,
            "include_raw_content": False,
        }
        try:
            # One deadline for the whole call - httpx's own timeout applies per phase
            # (connect, read, ...), so a slow connect plus a slow read could add up past it.
            async with asyncio.timeout(settings.TAVILY_TIMEOUT_SECONDS):
                async with httpx.AsyncClient(
                    base_url=settings.TAVILY_BASE_URL, transport=self._transport
                ) as client:
                    response = await client.post(
                        "/search",
                        json=body,
                        headers={"Authorization": f"Bearer {settings.TAVILY_API_KEY}"},
                    )
        except (TimeoutError, httpx.TimeoutException) as exc:
            raise WebSearchUnavailableError(
                WebSearchFailureReason.TIMEOUT,
                f"Tavily không phản hồi trong {settings.TAVILY_TIMEOUT_SECONDS:g} giây "
                "(TAVILY_TIMEOUT_SECONDS).",
            ) from exc
        except httpx.RequestError as exc:
            raise WebSearchUnavailableError(
                WebSearchFailureReason.CONNECTION_ERROR,
                f"Không kết nối được tới Tavily ({type(exc).__name__}). "
                "Kiểm tra mạng và TAVILY_BASE_URL.",
            ) from exc

        if response.status_code >= 300:
            raise _status_error(response)

        try:
            raw_results = response.json()["results"]
            return [WebSearchResult.model_validate(item) for item in raw_results]
        except (ValueError, KeyError, TypeError, ValidationError) as exc:
            raise WebSearchUnavailableError(
                WebSearchFailureReason.MALFORMED_RESPONSE,
                "Tavily trả về dữ liệu không đọc được (sai định dạng kết quả tìm kiếm).",
            ) from exc


def _status_error(response: httpx.Response) -> WebSearchUnavailableError:
    status = response.status_code
    if status in (401, 403):
        reason = WebSearchFailureReason.AUTH_FAILED
        message = f"Tavily từ chối API key (HTTP {status}). Kiểm tra TAVILY_API_KEY."
    elif status in _CREDIT_LIMIT_STATUSES:
        reason = WebSearchFailureReason.CREDITS_EXHAUSTED
        message = (
            f"Tài khoản Tavily đã hết credit hoặc chạm giới hạn chi tiêu (HTTP {status}). "
            "Nạp thêm credit hoặc nâng gói tại app.tavily.com."
        )
    elif status == 429:
        reason = WebSearchFailureReason.RATE_LIMITED
        message = "Tavily đang giới hạn tốc độ gọi (HTTP 429), thử lại sau ít phút."
    elif status >= 500:
        reason = WebSearchFailureReason.PROVIDER_ERROR
        message = f"Tavily đang gặp sự cố (HTTP {status}), thử lại sau."
    else:
        reason = WebSearchFailureReason.REQUEST_REJECTED
        message = f"Tavily từ chối yêu cầu tìm kiếm (HTTP {status})."
    detail = _error_detail(response)
    if detail:
        message = f"{message} Chi tiết: {detail}"
    return WebSearchUnavailableError(reason, message)


def _error_detail(response: httpx.Response) -> str:
    """Tavily's own error text (`{"detail": {"error": "..."}}`), key-redacted
    and trimmed - empty when the body says nothing usable."""

    try:
        body = response.json()
    except ValueError:
        return ""
    detail = body.get("detail") if isinstance(body, dict) else None
    if isinstance(detail, dict):
        detail = detail.get("error")
    if not isinstance(detail, str):
        return ""
    if settings.TAVILY_API_KEY:
        detail = detail.replace(settings.TAVILY_API_KEY, "***")
    return detail.strip()[:_MAX_DETAIL_CHARS]
