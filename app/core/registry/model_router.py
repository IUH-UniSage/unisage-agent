"""Circuit breaker + cooldown + priority failover across `ModelRegistry` credentials -
the single place this logic lives. Also gates credential selection against the
PROVIDER-scope budget when a caller opts in (see `select_credential_with_budget`).

Two credential-level states, keyed by `(credential_id, credential_revision)` so a
rotated credential (new revision) always starts with clean state:

- **Cooling down** (TRANSIENT failure): skipped for a TTL — the provider's own
  `Retry-After` if it sent one, otherwise a short default backoff.
- **Excluded** (PERMANENT failure): skipped immediately, and a health report is sent
  to `backend-java` so it can set the row `DISABLED`. The exclusion marker also
  carries a TTL — a long one — since the credential is expected to actually disappear
  from the snapshot once Java disables it and the next hot-reload picks that up; the
  marker is just the immediate stop-gap until that happens.

State lives in Redis (`REDIS_URL`, DB 0 — shared with the registry/hot-reload signal,
never Celery's broker/backend DBs) under key prefix `mr:cb:` so every worker process
agrees. When Redis is unreachable, the router degrades to **per-process, in-memory**
state instead of raising — each worker may then make a locally-inconsistent decision,
which is an accepted tradeoff (degrade, don't crash). This mirrors the
try/log/swallow-on-Redis-failure pattern already used by `app.core.observability.events` (Celery
worker publishing ingestion progress) and `app.core.registry.registry_subscriber` (a dropped
pub/sub connection is logged and retried, never fatal).

The router always reports health to Java (`report_health`) on failure — TRANSIENT and
PERMANENT alike ("Ghi errorCount/lastErrorAt, PERMANENT → DISABLED"): Java decides
what a given `errorType` means, this side just relays it. A failed health-report call
is itself best-effort (logged, swallowed) — losing one health ping must never blow up
the request/task that just failed against the provider.

`credentialRevision`/`snapshotVersion` on that report must reflect the moment the
failure happened, not whatever the registry snapshot has drifted to by the time the
report actually reaches Java. This module cannot
capture that "moment" on its own — the caller is the one holding both the credential
(already carries its own `.revision`) and the snapshot version at the instant the
provider call failed, before it goes on to pick a fallback credential (which may
itself trigger a hot-reload). So `record_failure()` takes `snapshot_version` as an
explicit argument rather than reading `model_registry.get_current_snapshot()` itself —
reading it fresh here would reintroduce exactly the drift this is meant to prevent.
"""

from __future__ import annotations

import logging
import threading
import time
from datetime import UTC, datetime
from decimal import Decimal
from typing import TYPE_CHECKING, Any, Protocol

import redis.asyncio as redis_asyncio

from app.core.config import settings
from app.core.errors.llm_error_classifier import ErrorType, classify_llm_error
from app.core.errors.llm_failure import admin_failure_message, describe_llm_failure
from app.core.observability.alerting import alert_credential_failure
from app.core.registry.errors import NoAvailableCredentialError, NoBudgetAvailableError
from app.core.registry.model_registry import CredentialConfig, active_credentials_for
from app.integrations.backend_java_client import BackendJavaClient

if TYPE_CHECKING:
    from app.core.budget.tracker import BudgetTracker

logger = logging.getLogger(__name__)

_KEY_PREFIX = "mr:cb"
# No Retry-After header from the provider: back off this long before retrying the
# same credential again.
_DEFAULT_COOLDOWN_SECONDS = 30.0
# How long a PERMANENT exclusion marker lives in Redis - long enough that it's not the
# thing doing the real work (Java setting the row DISABLED, and the next snapshot
# reload dropping it, is), just the immediate stop-gap until that's picked up.
_EXCLUDED_TTL_SECONDS = 24 * 60 * 60.0


class _RedisLike(Protocol):
    """Structural subset of `redis.asyncio.Redis` this module actually calls - lets
    tests hand in a bare fake instead of a real connection."""

    async def set(self, name: str, value: Any, *, ex: int | None = None) -> Any: ...

    async def exists(self, name: str) -> int: ...

    async def aclose(self) -> Any: ...


# Stored as the marker's value when the failure's reason isn't known (also what markers
# written before reasons were stored hold: "1").
_UNKNOWN_REASON = "1"


def _state_key(credential_id: str, revision: int) -> str:
    return f"{_KEY_PREFIX}:{credential_id}:{revision}"


class _InMemoryCircuitState:
    """Per-process fallback used only when Redis is unreachable. Not shared across
    workers - that inconsistency between workers during a Redis outage is an
    accepted tradeoff (degrade, don't crash)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._expires_at: dict[str, float] = {}
        self._reasons: dict[str, str] = {}

    def mark(self, key: str, ttl_seconds: float, reason: str = _UNKNOWN_REASON) -> None:
        with self._lock:
            self._expires_at[key] = time.monotonic() + max(0.0, ttl_seconds)
            self._reasons[key] = reason

    def reason(self, key: str) -> str | None:
        return self._reasons.get(key) if self.is_marked(key) else None

    def is_marked(self, key: str) -> bool:
        with self._lock:
            expiry = self._expires_at.get(key)
            if expiry is None:
                return False
            if expiry <= time.monotonic():
                del self._expires_at[key]
                self._reasons.pop(key, None)
                return False
            return True


def _extract_retry_after_seconds(exc: Exception) -> float | None:
    """Best-effort `Retry-After` (seconds) off a provider exception's HTTP response, if
    it carries one. Every SDK this router deals with (openai's
    `APIStatusError`-alikes, and PydanticAI's `ModelHTTPError`) exposes the underlying
    HTTP response as `.response`, an `httpx.Response`-shaped object with `.headers`."""

    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if headers is None or not hasattr(headers, "get"):
        return None
    try:
        raw = headers.get("Retry-After")
    except Exception:
        return None
    if not raw:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def _error_code_for(exc: Exception) -> str:
    """A short, non-secret-bearing code for the health report's `errorCode` field."""

    code = getattr(exc, "code", None)
    if isinstance(code, str) and code:
        return code
    status_code = getattr(exc, "status_code", None)
    if isinstance(status_code, int):
        return f"{type(exc).__name__}:{status_code}"
    return type(exc).__name__


class ModelRouter:
    """Purpose-agnostic credential selection + circuit breaker.

    Construct one per call site (or share a module-level default via
    `get_default_router()`) - it holds no long-lived connection, just configuration
    and, if Redis is ever unreachable, this instance's own in-memory fallback state.

    Test doubles: pass `redis_client` (anything satisfying `_RedisLike`) and/or
    `backend_client` to avoid a live Redis/`backend-java` entirely.
    """

    def __init__(
        self,
        *,
        redis_client: _RedisLike | None = None,
        backend_client: BackendJavaClient | None = None,
        default_cooldown_seconds: float = _DEFAULT_COOLDOWN_SECONDS,
        excluded_ttl_seconds: float = _EXCLUDED_TTL_SECONDS,
    ) -> None:
        self._injected_redis_client = redis_client
        self._backend_client = backend_client if backend_client is not None else BackendJavaClient()
        self._default_cooldown_seconds = default_cooldown_seconds
        self._excluded_ttl_seconds = excluded_ttl_seconds
        self._in_memory = _InMemoryCircuitState()

    def _new_redis_connection(self) -> _RedisLike:
        return redis_asyncio.Redis.from_url(settings.REDIS_URL)

    async def _is_blocked(self, key: str) -> bool:
        """Best-effort Redis check; falls back to this process's in-memory state on any
        Redis failure rather than raising."""

        if self._injected_redis_client is not None:
            try:
                return bool(await self._injected_redis_client.exists(key))
            except Exception:
                logger.warning(
                    "model_router: Redis unavailable checking %s - degrading to in-memory state",
                    key,
                    exc_info=True,
                )
                return self._in_memory.is_marked(key)

        try:
            conn = self._new_redis_connection()
        except Exception:
            logger.warning(
                "model_router: Redis unavailable checking %s - degrading to in-memory state",
                key,
                exc_info=True,
            )
            return self._in_memory.is_marked(key)

        try:
            return bool(await conn.exists(key))
        except Exception:
            logger.warning(
                "model_router: Redis unavailable checking %s - degrading to in-memory state",
                key,
                exc_info=True,
            )
            return self._in_memory.is_marked(key)
        finally:
            try:
                await conn.aclose()
            except Exception:  # pragma: no cover - best-effort cleanup only
                pass

    async def _mark(self, key: str, ttl_seconds: float, reason: str = _UNKNOWN_REASON) -> None:
        """Best-effort Redis write; falls back to this process's in-memory state on any
        Redis failure rather than raising. Always also marks in-memory, so an
        in-flight failure that happened to hit Redis-down keeps working even if Redis
        later comes back up mid-outage-recovery for THIS process's own view."""

        ttl_int = max(1, int(ttl_seconds))

        if self._injected_redis_client is not None:
            try:
                await self._injected_redis_client.set(key, reason, ex=ttl_int)
                return
            except Exception:
                logger.warning(
                    "model_router: Redis unavailable marking %s - degrading to in-memory state",
                    key,
                    exc_info=True,
                )
                self._in_memory.mark(key, ttl_seconds, reason)
                return

        try:
            conn = self._new_redis_connection()
        except Exception:
            logger.warning(
                "model_router: Redis unavailable marking %s - degrading to in-memory state",
                key,
                exc_info=True,
            )
            self._in_memory.mark(key, ttl_seconds, reason)
            return

        try:
            await conn.set(key, reason, ex=ttl_int)
        except Exception:
            logger.warning(
                "model_router: Redis unavailable marking %s - degrading to in-memory state",
                key,
                exc_info=True,
            )
            self._in_memory.mark(key, ttl_seconds, reason)
        finally:
            try:
                await conn.aclose()
            except Exception:  # pragma: no cover - best-effort cleanup only
                pass

    async def get_next_credential(
        self, purpose: str, *, exclude_ids: set[str] | None = None
    ) -> CredentialConfig:
        """The highest-priority ACTIVE credential for `purpose` that is neither
        cooling down, excluded, nor in `exclude_ids` (credentials the caller already
        tried and rejected for a reason this router doesn't track itself - e.g. a
        budget denial within the same request). Raises `NoAvailableCredentialError`
        if none - never loops/retries on its own."""

        candidates = sorted(
            active_credentials_for(purpose),
            key=lambda credential: (credential.priority is None, credential.priority),
        )
        for credential in candidates:
            if exclude_ids is not None and credential.id in exclude_ids:
                continue
            key = _state_key(credential.id, credential.revision)
            if not await self._is_blocked(key):
                return credential
        await alert_credential_failure(
            None,
            "NO_AVAILABLE_CREDENTIAL",
            f"Every credential for purpose={purpose!r} is cooling down or excluded",
            purpose=purpose,
        )
        reasons = [
            await self._suspension_reason(_state_key(credential.id, credential.revision))
            for credential in candidates
            if exclude_ids is None or credential.id not in exclude_ids
        ]
        raise NoAvailableCredentialError(
            purpose,
            suspension_reasons=tuple(
                reason for reason in reasons if reason and reason != _UNKNOWN_REASON
            ),
        )

    async def _suspension_reason(self, key: str) -> str | None:
        """Best-effort read of why `key` is cooling down/excluded - only used to explain
        an already-decided "no credential available", so any failure (Redis down, a test
        fake without `get`) just means "reason unknown", never an error."""

        in_memory = self._in_memory.reason(key)
        if in_memory is not None:
            return in_memory
        try:
            if self._injected_redis_client is not None:
                raw = await self._injected_redis_client.get(key)  # type: ignore[attr-defined]
            else:
                conn = self._new_redis_connection()
                try:
                    raw = await conn.get(key)  # type: ignore[attr-defined]
                finally:
                    await conn.aclose()
        except Exception:
            return None
        if isinstance(raw, bytes):
            return raw.decode("utf-8", errors="replace")
        return raw if isinstance(raw, str) else None

    async def record_failure(
        self,
        credential: CredentialConfig,
        exc: Exception,
        *,
        snapshot_version: int,
        purpose: str | None = None,
    ) -> None:
        """Records one provider-call failure for `credential` and reports it to
        `backend-java`.

        `snapshot_version` must be the registry snapshot version that was current at
        the moment `credential` was selected/used - the caller captures this itself
        (e.g. from `model_registry.get_current_snapshot().version` right when it got
        the credential), never re-derived here, so a hot-reload that happens while
        this call is in flight (or before it's even made) can never make the health
        report carry a newer value than what was actually active when the failure
        occurred.

        `purpose` is optional and only used to enrich the alert's message -
        callers that don't have it handy (or don't care about alerting context) can
        omit it.

        Every failure alerts, PERMANENT and TRANSIENT alike (product decision,
        superseding the original "TRANSIENT never alerts" design - see spec.md
        "Slack alerting"): a human should hear about a 503/high-demand blip the
        first time it happens, not only once the circuit breaker gives up on the
        credential entirely. This does not reintroduce the flood the original
        design was avoiding - `alert_credential_failure`'s own 15-minute debounce
        (per credential + incident type) still collapses a burst of the same
        transient error into one Slack message.
        """

        error_type = classify_llm_error(exc)
        key = _state_key(credential.id, credential.revision)
        message = admin_failure_message(exc, purpose=purpose, api_key=credential.api_key)
        # Kept as the marker's value so a LATER request that finds this credential
        # suspended can still tell the client why (see `NoAvailableCredentialError`).
        reason = describe_llm_failure(exc, purpose).reason.value

        if error_type is ErrorType.PERMANENT:
            await self._mark(key, self._excluded_ttl_seconds, reason)
        else:
            ttl = _extract_retry_after_seconds(exc) or self._default_cooldown_seconds
            await self._mark(key, ttl, reason)

        await alert_credential_failure(credential, error_type.value, message, purpose=purpose)

        try:
            await self._backend_client.report_health(
                credential_id=credential.id,
                credential_revision=credential.revision,
                snapshot_version=snapshot_version,
                error_type=error_type.value,
                error_code=_error_code_for(exc),
                message=message,
                occurred_at=datetime.now(UTC).isoformat(),
            )
        except Exception:
            # Best-effort - Java not hearing about this failure right now is not a
            # reason to blow up the caller that already has a bigger problem (the
            # provider call itself failing).
            logger.warning(
                "model_router: failed to report health for credential=%s revision=%s",
                credential.id,
                credential.revision,
                exc_info=True,
            )


_default_router: ModelRouter | None = None
_default_router_lock = threading.Lock()


def get_default_router() -> ModelRouter:
    """Lazily-constructed, process-wide `ModelRouter` for call sites that don't need
    their own instance/test doubles."""

    global _default_router
    if _default_router is None:
        with _default_router_lock:
            if _default_router is None:
                _default_router = ModelRouter()
    return _default_router


async def get_next_credential(purpose: str) -> CredentialConfig:
    """Convenience wrapper around `get_default_router().get_next_credential()`."""

    return await get_default_router().get_next_credential(purpose)


async def select_credential_with_budget(
    purpose: str,
    *,
    budget_tracker: BudgetTracker,
    request_id: str,
    seq: int,
    estimate_usd: Decimal,
    initial_credential: CredentialConfig | None = None,
    router: ModelRouter | None = None,
) -> CredentialConfig:
    """Reserves a credential against the PROVIDER-scope budget, retrying with the
    next `get_next_credential` candidate whenever `acquire_provider` denies the one
    just tried. `seq` is reused across every candidate tried in this one call - only
    the winning `acquire_provider` call actually writes anything to Redis (a denial
    writes nothing), so reusing it is safe and keeps this call's budget bookkeeping
    under one hash field.

    `initial_credential`, when given, is tried FIRST without going through
    `get_next_credential` at all - the caller already picked it (e.g. the request's
    shared `GraphModels` credential); only a budget denial for it falls through to
    the normal circuit-breaker-driven candidate loop, excluding it from there on.

    Raises `NoAvailableCredentialError` if circuit-breaker state runs out of
    candidates first, or `NoBudgetAvailableError` if every remaining candidate was
    denied by budget instead.
    """

    active_router = router if router is not None else get_default_router()
    excluded: set[str] = set()
    last_deny_reason = "DENY_EXCEEDED"

    if initial_credential is not None:
        result = await budget_tracker.acquire_provider(
            request_id=request_id,
            seq=seq,
            provider=initial_credential.provider,
            estimate_usd=estimate_usd,
        )
        if result == "OK":
            return initial_credential
        last_deny_reason = result
        excluded.add(initial_credential.id)

    while True:
        try:
            credential = await active_router.get_next_credential(purpose, exclude_ids=excluded)
        except NoAvailableCredentialError:
            if excluded:
                raise NoBudgetAvailableError(purpose, last_deny_reason) from None
            raise

        result = await budget_tracker.acquire_provider(
            request_id=request_id, seq=seq, provider=credential.provider, estimate_usd=estimate_usd
        )
        if result == "OK":
            return credential
        last_deny_reason = result
        excluded.add(credential.id)


async def record_failure(
    credential: CredentialConfig,
    exc: Exception,
    *,
    snapshot_version: int,
    purpose: str | None = None,
) -> None:
    """Convenience wrapper around `get_default_router().record_failure()`."""

    await get_default_router().record_failure(
        credential, exc, snapshot_version=snapshot_version, purpose=purpose
    )
