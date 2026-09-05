"""Reusable `pydantic_ai` model doubles for tests (T0.3).

Two flavors, matching how the graph nodes will call the LLM:
- `make_streaming_llm_model` — for nodes using `Agent.run_stream()`
  (DirectLLMNode/05A, GenerationSynthesisNode/12): yields a fixed sequence of
  text token deltas.
- `make_sync_llm_model` — for nodes using `Agent.run()`/`run_sync()`
  (MessageClassificationNode/03, QueryTransformationNode/06, and the
  Phase 2/3 nodes): returns one fixed text response, no streaming.

Both are built on `pydantic_ai.models.function.FunctionModel`, which is the
project's LLM double of choice — no network access, no API key needed, and
it exercises the real `Agent.run`/`run_stream` code path rather than
stubbing the graph node itself.
"""

from collections.abc import AsyncIterator, Sequence

from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart
from pydantic_ai.models.function import AgentInfo, FunctionModel


def make_streaming_llm_model(tokens: Sequence[str]) -> FunctionModel:
    """Build a `FunctionModel` whose `run_stream()` yields exactly `tokens`, in order."""

    async def stream_function(
        _messages: list[ModelMessage], _agent_info: AgentInfo
    ) -> AsyncIterator[str]:
        for token in tokens:
            yield token

    return FunctionModel(stream_function=stream_function)


def make_sync_llm_model(text: str) -> FunctionModel:
    """Build a `FunctionModel` whose `run()`/`run_sync()` returns `text` as one response."""

    def function(_messages: list[ModelMessage], _agent_info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[TextPart(content=text)])

    return FunctionModel(function=function)
