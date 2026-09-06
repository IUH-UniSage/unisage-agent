from collections.abc import Callable

import pytest
from pydantic_ai.models.function import FunctionModel

from app.graph.nodes.message_classification import build_classification_agent, classify_intent


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
