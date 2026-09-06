"""Shared token-streaming helper for LLM-driving graph nodes.

`TokenSink` is deliberately just "an async callable that accepts one token
string" - in production it is `asyncio.Queue.put`, wired up by
`app/graph/streaming_session.py`; tests can pass any async callable, e.g.
one that appends to a list.
"""

from collections.abc import Awaitable, Callable

from pydantic_ai import Agent

TokenSink = Callable[[str], Awaitable[None]]


async def stream_agent_text(
    agent: Agent[None, str],
    prompt: str,
    token_sink: TokenSink,
) -> str:
    """Run `agent.run_stream(prompt)`, forwarding every text delta to `token_sink`.

    Returns the full accumulated text once the stream completes. This is the
    ONLY place graph nodes call `run_stream` - keeping it in one function
    means both DirectLLMNode and GenerationSynthesisNode get identical
    delta-forwarding behavior.
    """

    collected: list[str] = []
    async with agent.run_stream(prompt) as result:
        async for chunk in result.stream_text(delta=True):
            collected.append(chunk)
            await token_sink(chunk)
    return "".join(collected)
