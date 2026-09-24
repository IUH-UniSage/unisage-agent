"""Reusable `pydantic_ai` model doubles for tests.

Two flavors, matching how the graph nodes call the LLM:
- `make_streaming_llm_model` — for nodes using `Agent.run_stream()`
  (GenerationSynthesisNode): yields a fixed sequence of
  text token deltas.
- `make_sync_llm_model` — for nodes using `Agent.run()`/`run_sync()`
  (MessageClassificationNode, QueryTransformationNode, and any future
  non-streaming node): returns one fixed text response, no streaming.

Both are built on `pydantic_ai.models.function.FunctionModel`, which is the
project's LLM double of choice — no network access, no API key needed, and
it exercises the real `Agent.run`/`run_stream` code path rather than
stubbing the graph node itself.
"""

import asyncio
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass, field

from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from app.schemas.retrieval import RetrievedChunk


def make_streaming_llm_model(tokens: Sequence[str]) -> FunctionModel:
    """Build a `FunctionModel` whose `run_stream()` yields exactly `tokens`, in order."""

    async def stream_function(
        _messages: list[ModelMessage], _agent_info: AgentInfo
    ) -> AsyncIterator[str]:
        for token in tokens:
            yield token

    return FunctionModel(stream_function=stream_function)


def make_sequential_streaming_llm_model(responses: Sequence[Sequence[str]]) -> FunctionModel:
    """Build a `FunctionModel` whose `run_stream()` yields `responses[0]` on the
    first call, `responses[1]` on the second, etc. (staying on the last entry
    for any call beyond the list) - for a node that calls the same `agent`
    more than once per turn (e.g. GenerationSynthesisNode's JSON-repair
    follow-up call), where each call needs a different scripted response."""

    call_index = {"value": 0}

    async def stream_function(
        _messages: list[ModelMessage], _agent_info: AgentInfo
    ) -> AsyncIterator[str]:
        index = min(call_index["value"], len(responses) - 1)
        call_index["value"] += 1
        for token in responses[index]:
            yield token

    return FunctionModel(stream_function=stream_function)


def make_gated_streaming_llm_model(tokens: Sequence[str], gate: asyncio.Event) -> FunctionModel:
    """Like `make_streaming_llm_model`, but pauses after yielding the FIRST
    token until `gate` is set, before yielding the rest.

    For tests that need to deterministically catch a stream "mid-flight"
    (e.g. cancelling the client's SSE connection while the background
    `run_and_persist` task is still running) without racing real wall-clock
    timing: a test can await its own signal that the first token has reached
    the client, act (e.g. cancel), and only then `gate.set()` to let the
    graph - and so the background task - actually finish.
    """

    async def stream_function(
        _messages: list[ModelMessage], _agent_info: AgentInfo
    ) -> AsyncIterator[str]:
        first, *rest = tokens
        yield first
        await gate.wait()
        for token in rest:
            yield token

    return FunctionModel(stream_function=stream_function)


def make_sync_llm_model(text: str) -> FunctionModel:
    """Build a `FunctionModel` whose `run()`/`run_sync()` returns `text` as one response."""

    def function(_messages: list[ModelMessage], _agent_info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[TextPart(content=text)])

    return FunctionModel(function=function)


@dataclass(frozen=True)
class FakeRetrievalService:
    """`GraphModels.retrieval` test double - returns a fixed list of chunks,
    no Qdrant/OpenAI call. Satisfies `RetrievalServiceProtocol` structurally."""

    chunks: list[RetrievedChunk] = field(default_factory=list)

    def retrieve(self, query: str, *, limit: int | None = None) -> list[RetrievedChunk]:
        del query
        return self.chunks if limit is None else self.chunks[:limit]
