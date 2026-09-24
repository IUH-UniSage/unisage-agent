from collections.abc import Callable

import pytest
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from app.graph.nodes.message_classification import build_classification_agent, classify_intent
from app.schemas.chat_history import HistoryMessage


@pytest.mark.asyncio
async def test_classify_intent_returns_recognized_label(
    mock_sync_llm_model: Callable[[str], FunctionModel],
) -> None:
    agent = build_classification_agent(mock_sync_llm_model("academic_advisory"))

    intent = await classify_intent(agent, "Điều kiện học bổng loại giỏi là gì?")

    assert intent == "academic_advisory"


@pytest.mark.asyncio
async def test_classify_intent_falls_back_on_unrecognized_output(
    mock_sync_llm_model: Callable[[str], FunctionModel],
) -> None:
    agent = build_classification_agent(mock_sync_llm_model("this is not a valid label"))

    intent = await classify_intent(agent, "some message")

    assert intent == "academic_advisory"


@pytest.mark.asyncio
async def test_classify_intent_is_case_insensitive(
    mock_sync_llm_model: Callable[[str], FunctionModel],
) -> None:
    agent = build_classification_agent(mock_sync_llm_model("OFF_TOPIC"))

    intent = await classify_intent(agent, "Cho mình công thức nấu phở")

    assert intent == "off_topic"


@pytest.mark.asyncio
async def test_classify_intent_sends_recent_history_with_message() -> None:
    seen_prompts: list[str] = []

    def capturing_function(messages: list[ModelMessage], _agent_info: AgentInfo) -> ModelResponse:
        last_part = messages[-1].parts[-1]
        seen_prompts.append(getattr(last_part, "content", ""))
        return ModelResponse(parts=[TextPart(content="`academic_advisory`")])

    agent = build_classification_agent(FunctionModel(capturing_function))

    intent = await classify_intent(
        agent,
        "sao không có quản lý xây dựng?",
        history=[HistoryMessage(role="USER", content="tổ hợp toán, vật lý xét được ngành nào")],
    )

    assert intent == "academic_advisory"
    assert seen_prompts[-1].startswith("sao không có quản lý xây dựng?")
    assert "tổ hợp toán, vật lý xét được ngành nào" in seen_prompts[-1]
