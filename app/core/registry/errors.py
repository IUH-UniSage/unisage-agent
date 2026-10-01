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
