"""`RateLimitedModel` - a PydanticAI `WrapperModel` that takes a slot from the credential's
`max_rpm` window (`app.core.registry.rpm_limiter`) before every provider request.

Wrapping the `Model` (rather than counting credential selections) counts what the provider
counts: one agent run can make several requests (tool calls, output retries), and one chat
request shares a credential across several graph nodes. A full window raises
`CredentialRpmSaturatedError` before any request is sent, so streaming callers fail over with
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

from app.core.registry.errors import CredentialRpmSaturatedError
from app.core.registry.model_registry import CredentialConfig
from app.core.registry.rpm_limiter import RpmLimiter, get_default_limiter


@dataclass(init=False)
class RateLimitedModel(WrapperModel):
    credential: CredentialConfig
    limiter: RpmLimiter | None

    def __init__(
        self, wrapped: Model, credential: CredentialConfig, limiter: RpmLimiter | None = None
    ) -> None:
        super().__init__(wrapped)
        self.credential = credential
        # `None` resolves to the process-wide limiter at call time, so a model built at import
        # time never pins a limiter tests or settings would swap out.
        self.limiter = limiter

    async def _take_slot(self) -> None:
        limiter = self.limiter or get_default_limiter()
        wait_seconds = await limiter.acquire(self.credential)
        if wait_seconds is not None:
            raise CredentialRpmSaturatedError(
                self.credential.id, self.credential.max_rpm or 0, wait_seconds
            )

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        await self._take_slot()
        return await super().request(messages, model_settings, model_request_parameters)

    @asynccontextmanager
    async def request_stream(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
        run_context: RunContext[Any] | None = None,
    ) -> AsyncGenerator[StreamedResponse]:
        await self._take_slot()
        async with self.wrapped.request_stream(
            messages, model_settings, model_request_parameters, run_context
        ) as response_stream:
            yield response_stream
