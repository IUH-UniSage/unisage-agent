"""Shared token-streaming helper for LLM-driving graph nodes.

`TokenSink` is deliberately just "an async callable that accepts one token
string" - in production it is a queue push (see `app/graph/streaming_session.py`);
tests can pass any async callable, e.g. one that appends to a list.

This is the ONLY place `agent.run_stream()` is called across the whole graph
(`app/graph/nodes/generation_synthesis.py` and `app/graph/nodes/ticket_fallback.py`
are the two call sites), which is also why the pre-first-chunk failover from
todo.md Task 11 lives here instead of being duplicated in each node: every
streaming call gets the same "retry before any chunk streamed, never after"
boundary for free.

Failover is opt-in via the `purpose`/`credential`/`snapshot_version`/
`agent_factory` keyword-only arguments. When any of them is omitted (the
default), this behaves exactly as before Task 11 - a failure propagates
immediately, uncaught. This keeps every existing call site/test that hands
in a bare `Model`/`FunctionModel` double, with no `model_router` wiring at
all, working unchanged.
"""

from collections.abc import Awaitable, Callable

from pydantic_ai import Agent
from pydantic_ai.models import Model

from app.core.llm.provider_models import build_model
from app.core.model_registry import CredentialConfig
from app.core.model_router import ModelRouter, get_default_router

TokenSink = Callable[[str], Awaitable[None]]

# Rebuilds an `Agent` around a freshly-built fallback `Model` - the two
# current call sites' `build_generation_agent`/`build_ticket_fallback_agent`
# (both just `Agent(model=model)`, no tools/deps attached) satisfy this.
AgentFactory = Callable[[Model | str], Agent[None, str]]


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
                    streamed_any = True
                    collected.append(chunk)
                    await token_sink(chunk)
            return "".join(collected)
        except Exception as exc:
            if streamed_any:
                # Past the point of no return for this call - some of the
                # current model's output already reached the client via
                # `token_sink`, so no retry can happen without mixing two
                # models' text into one response.
                raise
            if (
                purpose is None
                or active_credential is None
                or agent_factory is None
                or snapshot_version is None
            ):
                # No failover wiring supplied - preserve pre-Task-11 behavior
                # exactly (immediate propagation).
                raise
            await failover_router.record_failure(
                active_credential, exc, snapshot_version=snapshot_version
            )
            # Raises `NoAvailableCredentialError` if every credential for
            # `purpose` is cooling down/excluded - left uncaught here, it
            # propagates to the caller as the `LLM_UNAVAILABLE` trigger.
            active_credential = await failover_router.get_next_credential(purpose)
            active_agent = agent_factory(build_model(active_credential))
