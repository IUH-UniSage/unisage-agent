"""Shared LLM-call helpers for graph nodes: one streaming, one plain `agent.run()`.

`TokenSink` is deliberately just "an async callable that accepts one token
string" - in production it is a queue push (see `app/graph/streaming_session.py`);
tests can pass any async callable, e.g. one that appends to a list.

`stream_agent_text()` is the ONLY place `agent.run_stream()` is called across
the whole graph (`app/graph/nodes/generation_synthesis.py` and
`app/graph/nodes/ticket_fallback.py` are the two call sites), which is also
why the pre-first-chunk failover boundary lives here instead of
being duplicated in each node: every streaming call gets the same "retry
before any chunk streamed, never after" boundary for free.

`run_agent_text_with_failover()` is the non-streaming equivalent for
`MessageClassificationNode`/`QueryTransformationNode` (plain `agent.run()`,
not `run_stream()`): there is no partial-output boundary to worry about there
- a call either returns fully or raises - so every failure is safe to retry
from scratch, no "already streamed some of it" case to preserve.

Failover is opt-in via the `purpose`/`credential`/`snapshot_version`/
`agent_factory` keyword-only arguments on both helpers. When any of them is
omitted (the default), behavior is unchanged from before this helper existed
- a failure propagates immediately, uncaught. This keeps every
existing call site/test that hands in a bare `Model`/`FunctionModel` double,
with no `model_router` wiring at all, working unchanged.

`on_attempt` is a second, independent
opt-in callback: fired exactly once per attempt (every call to
`agent.run()`/`run_stream()`, including one that fails and gets retried by
failover), on BOTH the success and the failure path, with the credential
actually used, the usage (`None` on failure), latency and status. It has no
prerequisite on the failover kwargs being set - unlike `on_failover`, cost
must still be recorded for a bare call with no retry wiring at all. There is
no separate "before attempt" callback: the caller's `UsageRecorder` reads
`time.monotonic()` itself right before invoking `agent.run()`/`run_stream()`
in `on_attempt`'s SAME call - no separate hook is needed for that half
because nothing about "before" needs anything failure/success-shaped.
"""

import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from decimal import Decimal
from typing import TYPE_CHECKING

from pydantic_ai import Agent
from pydantic_ai.models import Model
from pydantic_ai.settings import ModelSettings
from pydantic_ai.usage import RunUsage

from app.core.config import settings
from app.core.llm.provider_models import build_model
from app.core.registry.errors import NoAvailableCredentialError
from app.core.registry.model_registry import CredentialConfig
from app.core.registry.model_router import (
    ModelRouter,
    get_default_router,
    select_credential_with_budget,
)
from app.core.security.redaction import safe_error_message

if TYPE_CHECKING:
    from app.core.budget.tracker import BudgetTracker

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AttemptOutcome:
    """One `on_attempt` callback firing."""

    credential: CredentialConfig | None
    attempt: int
    status: str  # "SUCCESS" | "ERROR"
    usage: RunUsage | None  # None on ERROR
    latency_ms: int
    error_code: str | None = None


# Called once per attempt (success or failure) - see module docstring. Never raises: a
# `UsageRecorder` callback that fails would take the whole graph down with it, so implementations
# must swallow their own errors (logging, not raising). Returns the attempt's committed dollar
# amount (Decimal("0") if unknown/not tracked) - used to release a matching budget reservation.
AttemptRecorder = Callable[[AttemptOutcome], Decimal]


def _safe_record_attempt(on_attempt: "AttemptRecorder | None", outcome: AttemptOutcome) -> Decimal:
    if on_attempt is None:
        return Decimal("0")
    try:
        return on_attempt(outcome)
    except Exception:
        logger.exception("on_attempt callback raised - usage for this attempt was NOT recorded")
        return Decimal("0")


@dataclass(frozen=True)
class BudgetContext:
    """Bundles what `stream_agent_text`/`run_agent_text_with_failover` need to gate
    each attempt against the PROVIDER-scope budget and release it afterwards.
    `reserve_seq` is normally `UsageRecorder.reserve_budget_seq` - a fresh seq per
    attempt, reused across every candidate tried within one failed-over selection
    (see `select_credential_with_budget`).

    `per_attempt_estimate_usd` is ONE call's upper bound, used for every
    PROVIDER-scope `acquire_provider` - deliberately smaller than whatever
    multiplied total the caller reserved at the SYSTEM/PURPOSE scope up front
    (that reservation covers every attempt across every node in the request;
    this one is checked per attempt, so reusing the multiplied total here would
    over-reserve PROVIDER budget by roughly the multiplier for every attempt)."""

    tracker: "BudgetTracker"
    request_id: str
    reserve_seq: Callable[[], int]
    per_attempt_estimate_usd: Decimal


def _credential_label(credential: CredentialConfig) -> str:
    """The operator-chosen nickname when there is one, else the raw id - what
    a failover log line should print so an admin can tell which credential
    without cross-referencing a UUID against the admin UI."""

    return credential.display_name or credential.id


async def _next_credential_or_raise(
    router: ModelRouter, purpose: str, failure: Exception
) -> CredentialConfig:
    """`router.get_next_credential(purpose)`, but when nothing is left the raised
    `NoAvailableCredentialError` carries `failure` - the provider error that just took
    out the last candidate - so the client is told the real cause (bad API key, quota,
    ...) instead of a generic "no credential available"."""

    try:
        return await router.get_next_credential(purpose)
    except NoAvailableCredentialError as exhausted:
        if exhausted.last_error is not None:
            raise
        raise NoAvailableCredentialError(
            purpose, last_error=failure, suspension_reasons=exhausted.suspension_reasons
        ) from failure


TokenSink = Callable[[str], Awaitable[None]]

# Rebuilds an `Agent` around a freshly-built fallback `Model` - the two
# current call sites' `build_generation_agent`/`build_ticket_fallback_agent`
# (both just `Agent(model=model)`, no tools/deps attached) satisfy this.
AgentFactory = Callable[[Model | str], Agent[None, str]]


def auxiliary_model_settings() -> ModelSettings:
    return ModelSettings(thinking=settings.CHAT_AUX_THINKING)


# Called once, right after a failover picks a replacement credential/model -
# lets the caller propagate the switch to the REST of the same request (see
# `on_failover`'s docstring on both helpers below for why this exists).
FailoverCallback = Callable[[CredentialConfig, Model | str], None]


async def stream_agent_text(
    agent: Agent[None, str],
    prompt: str,
    token_sink: TokenSink,
    *,
    purpose: str | None = None,
    credential: CredentialConfig | None = None,
    snapshot_version: int | None = None,
    agent_factory: AgentFactory | None = None,
    router: ModelRouter | None = None,
    on_failover: FailoverCallback | None = None,
    on_attempt: AttemptRecorder | None = None,
    budget: BudgetContext | None = None,
) -> str:
    """Run `agent.run_stream(prompt)`, forwarding every text delta to `token_sink`.

    Returns the full accumulated text once the stream completes.

    The "before vs after first chunk" boundary is the entire point of the
    SSE error contract: once a single chunk
    has been forwarded to `token_sink` for THIS call, a failure is no longer
    retried here - it propagates immediately and uncaught, because retrying
    would mean building a response out of two different models' output. Only
    a failure that happens before the first chunk of THIS call triggers
    failover: `model_router.record_failure()` is called with the credential
    and snapshot version that were current when it was selected/used (per
    that function's own "moment of failure, not later drift" contract), a
    replacement credential is fetched via `model_router.get_next_credential()`,
    a fresh `Agent` is built around it (`agent_factory`), and the SAME prompt
    is retried from scratch. This repeats until either a call succeeds or
    `NoAvailableCredentialError` propagates (the
    `LLM_UNAVAILABLE` case - handled by the caller, not here).

    `on_failover`, when supplied, is called once right after a replacement
    credential/model is picked - `run_graph`'s `GraphModels.classification`/
    `query_transformation`/`generation` are all built from the SAME
    top-priority credential once at request start, so without this callback
    a credential failing in one node leaves every LATER node in the same
    request still holding the dead credential, forcing each of them to
    independently rediscover the same failure and redo the same failover
    dance (extra failed provider calls + latency, once per node) instead of
    starting directly with the credential the request already knows is bad.
    `streaming_graph.py` wires this to mutate the shared `GraphModels`
    instance so a failover in an early node is visible to every node after it.

    `budget`, when supplied, gates every attempt (the first and every failover
    retry) against the PROVIDER-scope budget before it is made: a denial tries the
    next `purpose` candidate (never the provider itself, no wasted call), and
    `NoBudgetAvailableError` propagates uncaught if every candidate is denied. Each
    attempt's reservation is released right after it ends (success or failure) with
    the dollar amount `on_attempt` reports, before this function returns or retries.
    """

    active_agent = agent
    active_credential = credential
    failover_router = router if router is not None else get_default_router()
    attempt_index = 0

    # Same all-or-nothing precondition `has_failover_wiring` uses below - a bare
    # test-double model with no credential/agent_factory/purpose/snapshot_version
    # must never touch `model_router` (and therefore the registry) at all, budget
    # included, so this stays a strict opt-in exactly like failover already is.
    has_budget_wiring = budget is not None and not (
        purpose is None or credential is None or agent_factory is None or snapshot_version is None
    )

    while True:
        if has_budget_wiring:
            budget_seq = budget.reserve_seq()
            selected = await select_credential_with_budget(
                purpose,
                budget_tracker=budget.tracker,
                request_id=budget.request_id,
                seq=budget_seq,
                estimate_usd=budget.per_attempt_estimate_usd,
                initial_credential=active_credential,
                router=failover_router,
            )
            if selected is not active_credential:
                active_credential = selected
                active_model = build_model(active_credential)
                active_agent = agent_factory(active_model) if agent_factory else active_agent
                if on_failover is not None:
                    on_failover(active_credential, active_model)
        else:
            budget_seq = None

        collected: list[str] = []
        streamed_any = False
        started_at = time.monotonic()
        try:
            async with active_agent.run_stream(prompt) as result:
                async for chunk in result.stream_text(delta=True):
                    collected.append(chunk)
                    await token_sink(chunk)
                    if chunk:
                        # Only real (non-empty) content counts as "already
                        # streamed" - a thinking-model provider (Gemini 3)
                        # can emit an empty/marker delta before any visible
                        # text, and treating that as the point of no return
                        # would block a perfectly safe failover on a request
                        # the user never actually saw any output for.
                        streamed_any = True
                # Usage is only final once the stream has been fully consumed -
                # must be read here, still inside the `async with`, not after.
                usage = result.usage
            committed_usd = _safe_record_attempt(
                on_attempt,
                AttemptOutcome(
                    credential=active_credential,
                    attempt=attempt_index,
                    status="SUCCESS",
                    usage=usage,
                    latency_ms=int((time.monotonic() - started_at) * 1000),
                ),
            )
            if budget is not None and budget_seq is not None:
                await budget.tracker.release_provider(
                    request_id=budget.request_id, seq=budget_seq, actual_usd=committed_usd
                )
            return "".join(collected)
        except Exception as exc:
            latency_ms = int((time.monotonic() - started_at) * 1000)
            committed_usd = _safe_record_attempt(
                on_attempt,
                AttemptOutcome(
                    credential=active_credential,
                    attempt=attempt_index,
                    status="ERROR",
                    usage=None,
                    latency_ms=latency_ms,
                    error_code=type(exc).__name__,
                ),
            )
            if budget is not None and budget_seq is not None:
                await budget.tracker.release_provider(
                    request_id=budget.request_id, seq=budget_seq, actual_usd=committed_usd
                )
            has_failover_wiring = not (
                purpose is None
                or active_credential is None
                or agent_factory is None
                or snapshot_version is None
            )
            if streamed_any:
                # Past the point of no return for this call - some of the
                # current model's output already reached the client via
                # `token_sink`, so no retry can happen without mixing two
                # models' text into one response. The credential still gets
                # marked/alerted (if wiring was supplied) so the *next*
                # request picks a different one instead of hitting the same
                # failing credential again - only the retry-this-call part is
                # skipped, not the failure bookkeeping.
                if has_failover_wiring:
                    await failover_router.record_failure(
                        active_credential, exc, snapshot_version=snapshot_version, purpose=purpose
                    )
                raise
            if not has_failover_wiring:
                # No failover wiring supplied - preserve the immediate-propagation
                # behavior from before this failover machinery existed.
                raise
            await failover_router.record_failure(
                active_credential, exc, snapshot_version=snapshot_version, purpose=purpose
            )
            # Raises `NoAvailableCredentialError` (carrying `exc` as its
            # `last_error`) if every credential for `purpose` is cooling
            # down/excluded - left uncaught here, it propagates to the caller.
            failed_credential = active_credential
            active_credential = await _next_credential_or_raise(failover_router, purpose, exc)
            active_model = build_model(active_credential)
            active_agent = agent_factory(active_model)
            attempt_index += 1
            logger.warning(
                "stream_agent_text: credential %s failed (%s) before any output was "
                "streamed - failing over to credential %s",
                _credential_label(failed_credential),
                safe_error_message(exc, failed_credential.api_key),
                _credential_label(active_credential),
            )
            if on_failover is not None:
                on_failover(active_credential, active_model)


async def run_agent_text_with_failover(
    agent: Agent[None, str],
    prompt: str,
    *,
    purpose: str | None = None,
    credential: CredentialConfig | None = None,
    snapshot_version: int | None = None,
    agent_factory: AgentFactory | None = None,
    router: ModelRouter | None = None,
    on_failover: FailoverCallback | None = None,
    on_attempt: AttemptRecorder | None = None,
    budget: BudgetContext | None = None,
) -> str:
    """Runs `agent.run(prompt)`, returning `result.output or ""`, with the same
    opt-in failover wiring as `stream_agent_text()` - see the module docstring
    for why this needs no "already streamed" boundary.

    `on_failover`/`on_attempt`/`budget` - see `stream_agent_text()`'s docstring;
    same purpose here.
    """

    active_agent = agent
    active_credential = credential
    failover_router = router if router is not None else get_default_router()
    attempt_index = 0

    # Same all-or-nothing precondition `has_failover_wiring` uses below - a bare
    # test-double model with no credential/agent_factory/purpose/snapshot_version
    # must never touch `model_router` (and therefore the registry) at all, budget
    # included, so this stays a strict opt-in exactly like failover already is.
    has_budget_wiring = budget is not None and not (
        purpose is None or credential is None or agent_factory is None or snapshot_version is None
    )

    while True:
        if has_budget_wiring:
            budget_seq = budget.reserve_seq()
            selected = await select_credential_with_budget(
                purpose,
                budget_tracker=budget.tracker,
                request_id=budget.request_id,
                seq=budget_seq,
                estimate_usd=budget.per_attempt_estimate_usd,
                initial_credential=active_credential,
                router=failover_router,
            )
            if selected is not active_credential:
                active_credential = selected
                active_model = build_model(active_credential)
                active_agent = agent_factory(active_model) if agent_factory else active_agent
                if on_failover is not None:
                    on_failover(active_credential, active_model)
        else:
            budget_seq = None

        started_at = time.monotonic()
        try:
            result = await active_agent.run(prompt)
            committed_usd = _safe_record_attempt(
                on_attempt,
                AttemptOutcome(
                    credential=active_credential,
                    attempt=attempt_index,
                    status="SUCCESS",
                    usage=result.usage,
                    latency_ms=int((time.monotonic() - started_at) * 1000),
                ),
            )
            if budget is not None and budget_seq is not None:
                await budget.tracker.release_provider(
                    request_id=budget.request_id, seq=budget_seq, actual_usd=committed_usd
                )
            return result.output or ""
        except Exception as exc:
            committed_usd = _safe_record_attempt(
                on_attempt,
                AttemptOutcome(
                    credential=active_credential,
                    attempt=attempt_index,
                    status="ERROR",
                    usage=None,
                    latency_ms=int((time.monotonic() - started_at) * 1000),
                    error_code=type(exc).__name__,
                ),
            )
            if budget is not None and budget_seq is not None:
                await budget.tracker.release_provider(
                    request_id=budget.request_id, seq=budget_seq, actual_usd=committed_usd
                )
            if (
                purpose is None
                or active_credential is None
                or agent_factory is None
                or snapshot_version is None
            ):
                raise
            await failover_router.record_failure(
                active_credential, exc, snapshot_version=snapshot_version, purpose=purpose
            )
            failed_credential = active_credential
            active_credential = await _next_credential_or_raise(failover_router, purpose, exc)
            active_model = build_model(active_credential)
            active_agent = agent_factory(active_model)
            attempt_index += 1
            logger.warning(
                "run_agent_text_with_failover: credential %s failed (%s) - failing over "
                "to credential %s",
                _credential_label(failed_credential),
                safe_error_message(exc, failed_credential.api_key),
                _credential_label(active_credential),
            )
            if on_failover is not None:
                on_failover(active_credential, active_model)
