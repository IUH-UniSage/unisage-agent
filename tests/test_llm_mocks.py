"""Prove the mock-LLM test harness actually drives `pydantic_ai.Agent`
the way graph nodes do (`run_stream()` for streaming nodes, `run()` for
non-streaming ones) — not just that the raw `FunctionModel` objects exist.
"""

from collections.abc import Callable, Sequence

import pytest
from pydantic_ai import Agent
from pydantic_ai.models.function import FunctionModel


@pytest.mark.asyncio
async def test_streaming_fixture_yields_tokens_in_order(
    mock_streaming_llm_model: Callable[[Sequence[str]], FunctionModel],
) -> None:
    model = mock_streaming_llm_model(["Xin ", "chào ", "bạn"])
    agent: Agent[None, str] = Agent(model=model)

    collected: list[str] = []
    async with agent.run_stream("hi") as result:
        async for chunk in result.stream_text(delta=True):
            collected.append(chunk)

    assert "".join(collected) == "Xin chào bạn"


@pytest.mark.asyncio
async def test_sync_fixture_returns_fixed_text(
    mock_sync_llm_model: Callable[[str], FunctionModel],
) -> None:
    model = mock_sync_llm_model("câu trả lời cố định")
    agent: Agent[None, str] = Agent(model=model)

    result = await agent.run("hi")

    assert result.output == "câu trả lời cố định"
