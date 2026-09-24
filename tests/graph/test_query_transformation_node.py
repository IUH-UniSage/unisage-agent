from collections.abc import Callable

import pytest
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from app.graph.nodes.query_transformation import (
    build_query_transformation_agent,
    transform_query,
    transform_tasks,
)
from app.schemas.chat_history import HistoryMessage
from app.schemas.intent import ClassifiedTask, RoutingMode


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


@pytest.mark.asyncio
async def test_transform_query_appends_recent_history_without_citation_markers() -> None:
    seen_prompts: list[str] = []

    def capturing_function(messages: list[ModelMessage], _agent_info: AgentInfo) -> ModelResponse:
        last_part = messages[-1].parts[-1]
        seen_prompts.append(getattr(last_part, "content", ""))
        return ModelResponse(parts=[TextPart(content="hyde doc")])

    agent = build_query_transformation_agent(FunctionModel(capturing_function))

    await transform_query(
        agent,
        "còn khóa 2024-2025",
        history=[
            HistoryMessage(role="USER", content="học phí nghiên cứu sinh khóa 2025-2026"),
            HistoryMessage(role="ASSISTANT", content="Mức thu là 60.000.000 đồng [1]."),
        ],
    )

    prompt = seen_prompts[-1]
    assert prompt.startswith("còn khóa 2024-2025")
    assert "học phí nghiên cứu sinh khóa 2025-2026" in prompt
    assert "Mức thu là ... đồng" in prompt
    assert "60.000.000" not in prompt
    assert "[1]" not in prompt


def _echo_first_line_model() -> FunctionModel:
    """Returns "HyDE: <first line of its prompt>", so each output shows which
    task's query produced it."""

    def function(messages: list[ModelMessage], _agent_info: AgentInfo) -> ModelResponse:
        prompt = str(getattr(messages[-1].parts[-1], "content", ""))
        return ModelResponse(parts=[TextPart(content=f"HyDE: {prompt.splitlines()[0]}")])

    return FunctionModel(function=function)


@pytest.mark.asyncio
async def test_transform_tasks_returns_one_query_per_task_in_task_order() -> None:
    agent = build_query_transformation_agent(_echo_first_line_model())
    tasks: list[tuple[ClassifiedTask, RoutingMode]] = [
        (ClassifiedTask(intent="academic_advisory", query="Học phí ngành CNTT?"), "SINGLE"),
        (ClassifiedTask(intent="academic_advisory", query="Điều kiện học bổng?"), "SINGLE"),
    ]

    queries = await transform_tasks(agent, tasks)

    assert queries == ["HyDE: Học phí ngành CNTT?", "HyDE: Điều kiện học bổng?"]


@pytest.mark.asyncio
async def test_transform_tasks_single_task_matches_transform_query() -> None:
    agent = build_query_transformation_agent(_echo_first_line_model())
    task = ClassifiedTask(intent="academic_advisory", query="Thủ tục bảo lưu?")

    queries = await transform_tasks(agent, [(task, "SINGLE")])

    assert queries == [await transform_query(agent, "Thủ tục bảo lưu?")]
