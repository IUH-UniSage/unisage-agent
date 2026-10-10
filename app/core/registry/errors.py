"""Exceptions raised by credential selection (`app.core.registry.model_router`).

Kept in their own dependency-free module so code that only needs to recognise them
(`app.core.errors.llm_failure`, the API error handlers, the ingest task) doesn't have
to import the router itself - which would create an import cycle, since the router
uses `llm_failure` to describe the failures it records.
"""

from __future__ import annotations


class NoAvailableCredentialError(Exception):
    """No usable (non-cooling-down, non-excluded) credential exists for `purpose` -
    the trigger condition for the "no available credential" alert.

    `last_error`, when set, is the provider failure that just knocked out the last
    usable credential within the same call (see `app.graph.streaming`'s failover
    loops) - the real cause the client should be told about (e.g. a 401), rather
    than the generic "nothing left to try".

    `suspension_reasons` is the stored reason (an `llm_failure` reason code such as
    "LLM_AUTH_FAILED") of each candidate that is cooling down/excluded - so even when
    the failure happened in an EARLIER request, the client still learns why."""

    def __init__(
        self,
        purpose: str,
        *,
        last_error: BaseException | None = None,
        suspension_reasons: tuple[str, ...] = (),
    ) -> None:
        self.purpose = purpose
        self.last_error = last_error
        self.suspension_reasons = suspension_reasons
        super().__init__(f"No available credential for purpose={purpose!r}")


class NoBudgetAvailableError(Exception):
    """Every remaining candidate for `purpose` was denied by budget enforcement
    (PROVIDER-scope BLOCK/THROTTLE), distinct from `NoAvailableCredentialError` so
    callers can map it to a budget-specific error response instead of
    LLM_UNAVAILABLE."""

    def __init__(self, purpose: str, last_deny_reason: str) -> None:
        self.purpose = purpose
        self.last_deny_reason = last_deny_reason
        super().__init__(
            f"Every credential for purpose={purpose!r} was denied by budget "
            f"enforcement (last reason: {last_deny_reason})"
        )


class CredentialLocallyLimitedError(Exception):
    """A call refused on this side by one of the credential's own limits (`max_rpm`,
    `max_concurrency`) - before anything reached the provider. Raised by
    `app.core.llm.rate_limited_model.RateLimitedModel`; the router cools the credential down for
    `retry_after_seconds` without alerting or reporting health, and the failover loops move on
    to the next credential."""

    def __init__(self, credential_id: str, retry_after_seconds: float, message: str) -> None:
        self.credential_id = credential_id
        self.retry_after_seconds = retry_after_seconds
        super().__init__(message)


class CredentialRpmSaturatedError(CredentialLocallyLimitedError):
    """The credential already made its `max_rpm` calls within the last minute;
    `retry_after_seconds` is when the oldest call in the window ages out."""

    def __init__(self, credential_id: str, max_rpm: int, retry_after_seconds: float) -> None:
        self.max_rpm = max_rpm
        super().__init__(
            credential_id,
            retry_after_seconds,
            f"Credential {credential_id!r} reached its max_rpm={max_rpm}; "
            f"next slot in {retry_after_seconds:.1f}s",
        )


class CredentialConcurrencySaturatedError(CredentialLocallyLimitedError):
    """The credential already has `max_concurrency` calls in flight."""

    def __init__(
        self, credential_id: str, max_concurrency: int, retry_after_seconds: float
    ) -> None:
        self.max_concurrency = max_concurrency
        super().__init__(
            credential_id,
            retry_after_seconds,
            f"Credential {credential_id!r} already has {max_concurrency} call(s) in flight "
            "(max_concurrency)",
        )
