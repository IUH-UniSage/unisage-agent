"""Client of the local cross-encoder rerank service (`reranker/`, `task reranker:serve`).

`POST {CROSS_ENCODER_URL}/rerank {query, passages, top_k?}` returns one score per passage
(docs/specs/SPEC-cross-encoder-rerank.md). Like `tavily_client`, this talks to a fixed,
operator-configured host, so it uses a plain `httpx.AsyncClient`, not the SSRF-guarded
provider factory.

Every failure - connection refused (service not started), timeout, non-2xx, a body that
is not the expected shape - is raised as `CrossEncoderUnavailableError` with a Vietnamese
message for the AI-admin warning; the graph then falls over to the LLM rerank.
"""

import asyncio
from dataclasses import dataclass

import httpx
from pydantic import BaseModel, ValidationError

from app.core.config import settings


class CrossEncoderUnavailableError(Exception):
    """The rerank service could not score this request. `str(exc)` is the admin-facing
    Vietnamese explanation."""


@dataclass(frozen=True)
class CrossEncoderScores:
    scores: list[float]  # one per passage, passage order
    ranking: list[int]  # passage indexes, best first (cut to top_k when asked)
    truncated: int  # passages the service cut at its max_length
    took_ms: int  # the service's own scoring time
    model: str


class _RerankResponse(BaseModel):
    scores: list[float]
    ranking: list[int]
    truncated: int = 0
    took_ms: int = 0
    model: str = ""


class CrossEncoderClient:
    def __init__(
        self,
        *,
        base_url: str | None = None,
        timeout_seconds: float | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._base_url = base_url or settings.CROSS_ENCODER_URL
        self._timeout = timeout_seconds or settings.CROSS_ENCODER_TIMEOUT_SECONDS
        self._transport = transport

    async def rerank(
        self, query: str, passages: list[str], *, top_k: int | None = None
    ) -> CrossEncoderScores:
        if not passages:
            return CrossEncoderScores(scores=[], ranking=[], truncated=0, took_ms=0, model="")
        body: dict[str, object] = {"query": query, "passages": passages}
        if top_k is not None:
            body["top_k"] = top_k
        try:
            # One deadline for the whole call (httpx's timeout applies per phase).
            async with asyncio.timeout(self._timeout):
                async with httpx.AsyncClient(
                    base_url=self._base_url,
                    transport=self._transport,
                    timeout=httpx.Timeout(self._timeout),
                ) as client:
                    response = await client.post("/rerank", json=body)
        except (TimeoutError, httpx.TimeoutException) as exc:
            raise CrossEncoderUnavailableError(
                f"Service cross-encoder không phản hồi trong {self._timeout:g} giây "
                "(CROSS_ENCODER_TIMEOUT_SECONDS)."
            ) from exc
        except httpx.RequestError as exc:
            raise CrossEncoderUnavailableError(
                f"Không kết nối được tới service cross-encoder tại {self._base_url} "
                f"({type(exc).__name__}). Chạy `task reranker:serve` hoặc kiểm tra "
                "CROSS_ENCODER_URL."
            ) from exc

        if response.status_code >= 300:
            raise CrossEncoderUnavailableError(
                f"Service cross-encoder trả về HTTP {response.status_code}."
            )
        try:
            parsed = _RerankResponse.model_validate(response.json())
        except (ValueError, ValidationError) as exc:
            raise CrossEncoderUnavailableError(
                "Service cross-encoder trả về dữ liệu không đọc được."
            ) from exc
        if len(parsed.scores) != len(passages):
            raise CrossEncoderUnavailableError(
                f"Service cross-encoder trả về {len(parsed.scores)} điểm cho {len(passages)} đoạn."
            )
        return CrossEncoderScores(
            scores=parsed.scores,
            ranking=parsed.ranking,
            truncated=parsed.truncated,
            took_ms=parsed.took_ms,
            model=parsed.model,
        )

    async def health(self) -> dict[str, object]:
        """`GET /health` as returned, or raises `CrossEncoderUnavailableError`."""

        try:
            async with asyncio.timeout(self._timeout):
                async with httpx.AsyncClient(
                    base_url=self._base_url,
                    transport=self._transport,
                    timeout=httpx.Timeout(self._timeout),
                ) as client:
                    response = await client.get("/health")
        except (TimeoutError, httpx.RequestError) as exc:
            raise CrossEncoderUnavailableError(
                f"Không kết nối được tới service cross-encoder tại {self._base_url}."
            ) from exc
        if response.status_code >= 300:
            raise CrossEncoderUnavailableError(
                f"Service cross-encoder trả về HTTP {response.status_code} ở /health."
            )
        body = response.json()
        return body if isinstance(body, dict) else {}
