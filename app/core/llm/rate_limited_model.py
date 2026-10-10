"""`RateLimitedModel` - a PydanticAI `WrapperModel` that enforces the credential's own limits
before every provider request: an in-flight slot from `max_concurrency`
(`app.core.registry.concurrency_limiter`, held until the response - or the whole stream - is
done) and a slot from the `max_rpm` window (`app.core.registry.rpm_limiter`).

Wrapping the `Model` (rather than counting credential selections) counts what the provider
counts: one agent run can make several requests (tool calls, output retries), and one chat
request shares a credential across several graph nodes. A refused call raises a
`CredentialLocallyLimitedError` before any request is sent, so streaming callers fail over with
nothing yet streamed.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

from pydantic_ai import RunContext
from pydantic_ai.messages import ModelMessage, ModelResponse
from pydantic_ai.models import Model, ModelRequestParameters, StreamedResponse
from pydantic_ai.models.wrapper import WrapperModel
from pydantic_ai.settings import ModelSettings

from app.core.registry.concurrency_limiter import (
    ConcurrencyLease,
    ConcurrencyLimiter,
    get_default_concurrency_limiter,
)
from app.core.registry.errors import (
    CredentialConcurrencySaturatedError,
    CredentialRpmSaturatedError,
)
from app.core.registry.model_registry import CredentialConfig
from app.core.registry.rpm_limiter import RpmLimiter, get_default_limiter

# No way to know when an in-flight call ends - check back shortly.
_CONCURRENCY_RETRY_AFTER_SECONDS = 2.0


@dataclass(init=False)
class RateLimitedModel(WrapperModel):
    credential: CredentialConfig
    limiter: RpmLimiter | None
    concurrency_limiter: ConcurrencyLimiter | None

    def __init__(
        self,
        wrapped: Model,
        credential: CredentialConfig,
        limiter: RpmLimiter | None = None,
        concurrency_limiter: ConcurrencyLimiter | None = None,
    ) -> None:
        super().__init__(wrapped)
        self.credential = credential
        # `None` resolves to the process-wide limiters at call time, so a model built at import
        # time never pins a limiter tests or settings would swap out.
        self.limiter = limiter
        self.concurrency_limiter = concurrency_limiter

    async def _take_slots(self) -> ConcurrencyLease:
        concurrency_limiter = self.concurrency_limiter or get_default_concurrency_limiter()
        lease = await concurrency_limiter.acquire(self.credential)
        if lease is None:
            raise CredentialConcurrencySaturatedError(
                self.credential.id,
                self.credential.max_concurrency or 0,
                _CONCURRENCY_RETRY_AFTER_SECONDS,
            )
        try:
            limiter = self.limiter or get_default_limiter()
            wait_seconds = await limiter.acquire(self.credential)
        except BaseException:
            await lease.release()
            raise
        if wait_seconds is not None:
            await lease.release()
            raise CredentialRpmSaturatedError(
                self.credential.id, self.credential.max_rpm or 0, wait_seconds
            )
        return lease

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        lease = await self._take_slots()
        try:
            return await super().request(messages, model_settings, model_request_parameters)
        finally:
            await lease.release()

    @asynccontextmanager
    async def request_stream(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
        run_context: RunContext[Any] | None = None,
    ) -> AsyncGenerator[StreamedResponse]:
        lease = await self._take_slots()
        try:
            async with self.wrapped.request_stream(
                messages, model_settings, model_request_parameters, run_context
            ) as response_stream:
                yield response_stream
        finally:
            await lease.release()
