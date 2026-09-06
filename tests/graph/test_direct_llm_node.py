from collections.abc import Callable, Sequence

import pytest
from pydantic_ai.models.function import FunctionModel

from app.graph.nodes.direct_llm import build_direct_llm_agent, run_direct_llm
from app.schemas.security import AcademicSecurityContext


@pytest.mark.asyncio
async def test_run_direct_llm_streams_tokens_and_returns_full_text(
    mock_streaming_llm_model: Callable[[Sequence[str]], FunctionModel],
) -> None:
    agent = build_direct_llm_agent(mock_streaming_llm_model(["2", " (một cộng một bằng hai)"]))
    received: list[str] = []

    async def sink(token: str) -> None:
        received.append(token)

    full_text = await run_direct_llm(
        agent,
        user_query="1 + 1 bằng mấy?",
        security=AcademicSecurityContext(),
        confirmed_metadata={},
        token_sink=sink,
    )

    assert full_text == "2 (một cộng một bằng hai)"
    # pydantic_ai may coalesce adjacent deltas internally - assert the sink
    # was actually driven (streaming happened) and reconstructs the same text,
    # not that it preserves the exact chunk boundaries we fed the mock model.
    assert received
    assert "".join(received) == full_text
