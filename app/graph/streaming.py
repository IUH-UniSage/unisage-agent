"""Shared LLM-call helpers for graph nodes: one streaming, one plain `agent.run()`.

`TokenSink` is deliberately just "an async callable that accepts one token
string" - in production it is a queue push (see `app/graph/streaming_session.py`);
tests can pass any async callable, e.g. one that appends to a list.

`stream_agent_text()` is the ONLY place `agent.run_stream()` is called across
the whole graph (`app/graph/nodes/generation_synthesis.py` and
`app/graph/nodes/ticket_fallback.py` are the two call sites), which is also
why the pre-first-chunk failover from todo.md Task 11 lives here instead of
being duplicated in each node: every streaming call gets the same "retry
before any chunk streamed, never after" boundary for free.

`run_agent_text_with_failover()` is the non-streaming equivalent for
`MessageClassificationNode`/`QueryTransformationNode` (plain `agent.run()`,
not `run_stream()`): there is no partial-output boundary to worry about there
- a call either returns fully or raises - so every failure is safe to retry
from scratch, no "already streamed some of it" case to preserve.

Failover is opt-in via the `purpose`/`credential`/`snapshot_version`/
`agent_factory` keyword-only arguments on both helpers. When any of them is
omitted (the default), behavior is exactly what it was before Task 11/this
helper existed - a failure propagates immediately, uncaught. This keeps every
existing call site/test that hands in a bare `Model`/`FunctionModel` double,
with no `model_router` wiring at all, working unchanged.
"""

import logging
from collections.abc import Awaitable, Callable

from pydantic_ai import Agent
from pydantic_ai.models import Model

from app.core.llm.provider_models import build_model
from app.core.model_registry import CredentialConfig
from app.core.model_router import ModelRouter, get_default_router
from app.core.redaction import safe_error_message

logger = logging.getLogger(__name__)


def _credential_label(credential: CredentialConfig) -> str:
    """The operator-chosen nickname when there is one, else the raw id - what
    a failover log line should print so an admin can tell which credential
    without cross-referencing a UUID against the admin UI."""

    return credential.display_name or credential.id

TokenSink = Callable[[str], Awaitable[None]]

# Rebuilds an `Agent` around a freshly-built fallback `Model` - the two
# current call sites' `build_generation_agent`/`build_ticket_fallback_agent`
# (both just `Agent(model=model)`, no tools/deps attached) satisfy this.
AgentFactory = Callable[[Model | str], Agent[None, str]]

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
) -> str:
    """Run `agent.run_stream(prompt)`, forwarding every text delta to `token_sink`.

    Returns the full accumulated text once the stream completes.

    The "before vs after first chunk" boundary is the entire point of the
    SSE error contract (plan.md "SSE error contract"): once a single chunk
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
    `model_router.NoAvailableCredentialError` propagates (todo.md's
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
    """

    active_agent = agent
    active_credential = credential
    failover_router = router if router is not None else get_default_router()

    while True:
        collected: list[str] = []
        streamed_any = False
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
            return "".join(collected)
        except Exception as exc:
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
                # No failover wiring supplied - preserve pre-Task-11 behavior
                # exactly (immediate propagation).
                raise
            await failover_router.record_failure(
                active_credential, exc, snapshot_version=snapshot_version, purpose=purpose
            )
            # Raises `NoAvailableCredentialError` if every credential for
            # `purpose` is cooling down/excluded - left uncaught here, it
            # propagates to the caller as the `LLM_UNAVAILABLE` trigger.
            failed_credential = active_credential
            active_credential = await failover_router.get_next_credential(purpose)
            active_model = build_model(active_credential)
            active_agent = agent_factory(active_model)
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
) -> str:
    """Runs `agent.run(prompt)`, returning `result.output or ""`, with the same
    opt-in failover wiring as `stream_agent_text()` - see the module docstring
    for why this needs no "already streamed" boundary.

    `on_failover` - see `stream_agent_text()`'s docstring; same purpose here.
    """

    active_agent = agent
    active_credential = credential
    failover_router = router if router is not None else get_default_router()

    while True:
        try:
            result = await active_agent.run(prompt)
            return result.output or ""
        except Exception as exc:
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
            active_credential = await failover_router.get_next_credential(purpose)
            active_model = build_model(active_credential)
            active_agent = agent_factory(active_model)
            logger.warning(
                "run_agent_text_with_failover: credential %s failed (%s) - failing over "
                "to credential %s",
                _credential_label(failed_credential),
                safe_error_message(exc, failed_credential.api_key),
                _credential_label(active_credential),
            )
            if on_failover is not None:
                on_failover(active_credential, active_model)
