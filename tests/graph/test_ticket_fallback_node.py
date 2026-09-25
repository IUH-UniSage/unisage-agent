from collections.abc import Awaitable, Callable, Sequence

import pytest
from pydantic_ai.messages import ModelMessage
from pydantic_ai.models.function import AgentInfo, FunctionModel

from app.graph.nodes.ticket_fallback import build_ticket_fallback_agent, run_ticket_fallback
from app.schemas.security import AcademicSecurityContext


def _sink(target: list[str]) -> Callable[[str], Awaitable[None]]:
    async def sink(token: str) -> None:
        target.append(token)

    return sink


def _prompt_capturing_model(seen: list[str]) -> FunctionModel:
    async def stream(messages: list[ModelMessage], _agent_info: AgentInfo):  # type: ignore[no-untyped-def]
        last_part = messages[-1].parts[-1]
        seen.append(getattr(last_part, "content", ""))
        yield "Chưa tìm thấy quy định phù hợp."

    return FunctionModel(stream_function=stream)


@pytest.mark.asyncio
async def test_streams_the_model_output_token_by_token(
    mock_streaming_llm_model: Callable[[Sequence[str]], FunctionModel],
) -> None:
    agent = build_ticket_fallback_agent(mock_streaming_llm_model(["Chưa tìm thấy ", "quy định."]))
    seen: list[str] = []

    text = await run_ticket_fallback(
        agent,
        "Câu hỏi không tìm thấy trong quy chế nào cả",
        security=AcademicSecurityContext(),
        confirmed_metadata={},
        token_sink=_sink(seen),
    )

    assert text == "Chưa tìm thấy quy định."
    assert "".join(seen) == text


@pytest.mark.asyncio
async def test_prompt_carries_no_retrieved_chunk_and_has_the_security_block() -> None:
    """The model must not be able to cite a regulation it has no source for -
    the prompt should never contain retrieved-chunk data, only the static
    fallback rules, the security block, and the user's question."""

    seen_prompts: list[str] = []
    agent = build_ticket_fallback_agent(_prompt_capturing_model(seen_prompts))

    await run_ticket_fallback(
        agent,
        "Câu hỏi lạ chưa có quy định",
        security=AcademicSecurityContext(),
        confirmed_metadata={},
        token_sink=_sink([]),
    )

    (prompt,) = seen_prompts
    assert "UNIQUE_CHUNK_MARKER_MUST_NOT_APPEAR" not in prompt
    assert "{prepared_context}" not in prompt
    assert "Quy Tắc Bảo Mật" in prompt
