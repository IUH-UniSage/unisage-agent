from collections.abc import Callable

import pytest
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from app.graph.nodes.query_transformation import build_query_transformation_agent, transform_query


@pytest.mark.asyncio
async def test_transform_query_returns_hyde_text(
    mock_sync_llm_model: Callable[[str], FunctionModel],
) -> None:
    agent = build_query_transformation_agent(
        mock_sync_llm_model("Điều 8: Sinh viên được miễn học phần GDQP nếu...")
    )

    hyde = await transform_query(agent, "Sinh viên năm cuối có được miễn GDQP không?")

    assert hyde == "Điều 8: Sinh viên được miễn học phần GDQP nếu..."


@pytest.mark.asyncio
async def test_transform_query_folds_confirmed_metadata_into_prompt(
    mock_sync_llm_model: Callable[[str], FunctionModel],
) -> None:
    seen_prompts: list[str] = []

    def capturing_function(messages: list[ModelMessage], _agent_info: AgentInfo) -> ModelResponse:
        last_part = messages[-1].parts[-1]
        seen_prompts.append(getattr(last_part, "content", ""))
        return ModelResponse(parts=[TextPart(content="hyde doc")])

    agent = build_query_transformation_agent(FunctionModel(capturing_function))

    await transform_query(
        agent,
        "Sinh viên năm cuối có được miễn GDQP không?",
        confirmed_metadata={"training_type": "chinh_quy"},
    )

    assert "chinh_quy" in seen_prompts[-1]
