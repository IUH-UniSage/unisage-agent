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
import json
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass, field

from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from app.schemas.retrieval import RetrievedChunk
from app.schemas.security import AcademicSecurityContext


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


# Intents that never reach QueryTransformationNode - mirrors
# app.graph.nodes.message_classification._NO_ROUTING_MODE_INTENTS.
_NO_ROUTING_MODE_INTENTS = frozenset(
    {"social_chat", "off_topic", "academic_calculation", "greeting"}
)

# `append_recent_history` puts the message first, then this separator and
# the recent history - everything before it is the user's own message.
_HISTORY_SEPARATOR = "\n\nLịch sử hội thoại gần đây"


def _task_payload(intent: str, query: str, routing_mode: str | None) -> dict[str, object]:
    if routing_mode is None and intent not in _NO_ROUTING_MODE_INTENTS:
        routing_mode = "SINGLE"
    return {"intent": intent, "query": query, "routing_mode": routing_mode}


def make_classification_llm_model(*intents: str, routing_mode: str | None = None) -> FunctionModel:
    """Build a `FunctionModel` for `GraphModels.classification` that returns
    a valid `IntentClassification` JSON payload - most tests just want a
    fixed route and don't care about the JSON shape MessageClassificationNode
    parses (see tests/graph/test_message_classification_node.py for that).

    One intent (the common case) → one task whose `query` is the user's
    message copied verbatim, exactly what the real prompt asks the model to
    do for a one-question message. Several intents → one task each, with a
    placeholder `query` per task. `routing_mode` applies to every task that
    needs one (`"SINGLE"` by default, `None` for intents that never reach
    node 06), so callers only pass it for a MULTI case.
    """

    if not intents:
        raise ValueError("make_classification_llm_model needs at least one intent")

    def function(messages: list[ModelMessage], _agent_info: AgentInfo) -> ModelResponse:
        prompt = ""
        for part in messages[-1].parts:
            content = getattr(part, "content", None)
            if isinstance(content, str):
                prompt = content
        message = prompt.split(_HISTORY_SEPARATOR, 1)[0]
        if len(intents) == 1:
            tasks = [_task_payload(intents[0], message, routing_mode)]
        else:
            tasks = [
                _task_payload(intent, f"câu hỏi {index} ({intent})", routing_mode)
                for index, intent in enumerate(intents, 1)
            ]
        payload = {"tasks": tasks, "confidence": 0.95}
        return ModelResponse(parts=[TextPart(content=json.dumps(payload, ensure_ascii=False))])

    return FunctionModel(function=function)


@dataclass(frozen=True)
class FakeRetrievalService:
    """`GraphModels.retrieval` test double - returns a fixed list of chunks,
    no Qdrant/OpenAI call. Satisfies `RetrievalServiceProtocol` structurally.
    Ignores `security` - tests that care about permission filtering use
    `app.rag.vectorstore.qdrant_store.build_access_filter` directly instead."""

    chunks: list[RetrievedChunk] = field(default_factory=list)

    def retrieve(
        self,
        query: str,
        *,
        security: AcademicSecurityContext,
        limit: int | None = None,
    ) -> list[RetrievedChunk]:
        del query, security
        return self.chunks if limit is None else self.chunks[:limit]
