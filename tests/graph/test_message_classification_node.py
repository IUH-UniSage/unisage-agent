import json
from collections.abc import Callable

import pytest
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from app.graph.nodes.message_classification import (
    MAX_TASKS,
    build_classification_agent,
    classify_intent,
    parse_classification,
)
from app.schemas.chat_history import HistoryMessage

_MESSAGE = "Điều kiện học bổng loại giỏi là gì?"


def _payload(*tasks: dict[str, object], confidence: float | None = 0.9) -> str:
    return json.dumps({"tasks": list(tasks), "confidence": confidence}, ensure_ascii=False)


def _task(intent: str, query: str, routing_mode: str | None) -> dict[str, object]:
    return {"intent": intent, "query": query, "routing_mode": routing_mode}


@pytest.mark.asyncio
async def test_classify_intent_parses_a_single_task(
    mock_sync_llm_model: Callable[[str], FunctionModel],
) -> None:
    agent = build_classification_agent(
        mock_sync_llm_model(
            _payload(_task("academic_advisory", _MESSAGE, "SINGLE"), confidence=0.95)
        )
    )

    classification = await classify_intent(agent, _MESSAGE)

    assert len(classification.tasks) == 1
    task = classification.tasks[0]
    assert (task.intent, task.query, task.routing_mode) == ("academic_advisory", _MESSAGE, "SINGLE")
    assert classification.confidence == 0.95


def test_parse_keeps_two_tasks_of_different_kinds_in_order() -> None:
    message = "Tính giúp điểm GPA cho mình và cho mình biết thủ tục đăng ký tốt nghiệp"
    output = _payload(
        _task("academic_calculation", "Tính giúp điểm GPA cho mình", None),
        _task("academic_advisory", "Thủ tục đăng ký tốt nghiệp là gì?", "SINGLE"),
    )

    classification = parse_classification(output, message)

    assert [(t.intent, t.query, t.routing_mode) for t in classification.tasks] == [
        ("academic_calculation", "Tính giúp điểm GPA cho mình", None),
        ("academic_advisory", "Thủ tục đăng ký tốt nghiệp là gì?", "SINGLE"),
    ]


def test_parse_keeps_multi_on_a_comparison_task() -> None:
    message = "Ngành CNTT và Kế toán học phí chênh bao nhiêu?"

    classification = parse_classification(
        _payload(_task("academic_advisory", message, "MULTI")), message
    )

    assert len(classification.tasks) == 1
    assert classification.tasks[0].routing_mode == "MULTI"


def test_parse_accepts_json_wrapped_in_a_code_fence() -> None:
    output = "```json\n" + _payload(_task("off_topic", "1+1 bằng mấy?", None)) + "\n```"

    classification = parse_classification(output, "1+1 bằng mấy?")

    assert classification.tasks[0].intent == "off_topic"
    assert classification.tasks[0].routing_mode is None


def test_parse_defaults_missing_confidence_to_none() -> None:
    output = json.dumps({"tasks": [_task("academic_calculation", "Tính GPA giúp em", None)]})

    classification = parse_classification(output, "Tính GPA giúp em")

    assert classification.confidence is None


@pytest.mark.parametrize(
    "output",
    [
        "this is not valid JSON at all",
        "academic_advisory",  # the pre-Task-6 bare-label output shape
        '{"primary_intent": "academic_advisory", "routing_mode": "SINGLE"}',  # old 4-field shape
        '{"tasks": []}',
        '{"tasks": "academic_advisory"}',
        '{"tasks": ["not an object"]}',
    ],
)
def test_parse_falls_back_to_one_advisory_single_task_over_the_whole_message(
    output: str,
) -> None:
    classification = parse_classification(output, _MESSAGE)

    assert len(classification.tasks) == 1
    task = classification.tasks[0]
    assert (task.intent, task.query, task.routing_mode) == ("academic_advisory", _MESSAGE, "SINGLE")


def test_parse_maps_an_unknown_intent_to_academic_advisory() -> None:
    classification = parse_classification(
        _payload(_task("general_knowledge", "Python là gì?", None)), "Python là gì?"
    )

    task = classification.tasks[0]
    assert (task.intent, task.routing_mode) == ("academic_advisory", "SINGLE")


@pytest.mark.parametrize(
    "merged_intent", ["academic_procedure", "academic_calendar", "academic_document"]
)
def test_parse_maps_merged_away_intents_to_academic_advisory(merged_intent: str) -> None:
    """These three labels were folded into `academic_advisory` - a model that
    still returns one (stale prompt cache, a few-shot habit) must land on the
    same advisory path, keeping a comparison's MULTI."""

    classification = parse_classification(
        _payload(_task(merged_intent, "câu hỏi", "MULTI")), "câu hỏi"
    )

    task = classification.tasks[0]
    assert (task.intent, task.routing_mode) == ("academic_advisory", "MULTI")


def test_parse_replaces_an_empty_query_with_the_whole_message() -> None:
    classification = parse_classification(
        _payload(_task("academic_advisory", "   ", "SINGLE")), _MESSAGE
    )

    assert classification.tasks[0].query == _MESSAGE


@pytest.mark.parametrize(
    ("intent", "raw_mode", "expected_mode"),
    [
        ("off_topic", "SINGLE", None),  # never reaches node 06
        ("academic_calculation", "MULTI", None),
        ("academic_advisory", None, "SINGLE"),  # missing mode on an intent that needs one
        ("academic_advisory", "bogus", "SINGLE"),
        ("academic_advisory", "MULTI", "MULTI"),
    ],
)
def test_parse_forces_routing_mode_to_match_the_intent(
    intent: str, raw_mode: str | None, expected_mode: str | None
) -> None:
    classification = parse_classification(_payload(_task(intent, "câu hỏi", raw_mode)), "câu hỏi")

    assert classification.tasks[0].routing_mode == expected_mode


def test_parse_keeps_only_the_first_max_tasks() -> None:
    tasks = [_task("academic_advisory", f"câu hỏi {i}", "SINGLE") for i in range(MAX_TASKS + 2)]

    classification = parse_classification(_payload(*tasks), "nhiều câu hỏi")

    assert [t.query for t in classification.tasks] == [f"câu hỏi {i}" for i in range(MAX_TASKS)]


@pytest.mark.asyncio
async def test_classify_intent_sends_recent_history_with_message() -> None:
    seen_prompts: list[str] = []
    message = "sao không có quản lý xây dựng?"

    def capturing_function(messages: list[ModelMessage], _agent_info: AgentInfo) -> ModelResponse:
        last_part = messages[-1].parts[-1]
        seen_prompts.append(getattr(last_part, "content", ""))
        return ModelResponse(
            parts=[TextPart(content=_payload(_task("academic_advisory", message, "SINGLE")))]
        )

    agent = build_classification_agent(FunctionModel(capturing_function))

    classification = await classify_intent(
        agent,
        message,
        history=[HistoryMessage(role="USER", content="tổ hợp toán, vật lý xét được ngành nào")],
    )

    assert classification.tasks[0].intent == "academic_advisory"
    assert seen_prompts[-1].startswith(message)
    assert "tổ hợp toán, vật lý xét được ngành nào" in seen_prompts[-1]


@pytest.mark.asyncio
async def test_classification_mock_copies_the_message_into_a_single_task(
    mock_classification_llm_model: Callable[..., FunctionModel],
) -> None:
    """The shared mock the rest of the suite routes on behaves like the real
    prompt asks: one intent → one task whose `query` is the message
    verbatim, even when recent history is appended to the prompt."""

    agent = build_classification_agent(mock_classification_llm_model("academic_advisory"))

    classification = await classify_intent(
        agent,
        "Thủ tục bảo lưu thế nào?",
        history=[HistoryMessage(role="USER", content="chào bot")],
    )

    task = classification.tasks[0]
    assert (task.intent, task.query, task.routing_mode) == (
        "academic_advisory",
        "Thủ tục bảo lưu thế nào?",
        "SINGLE",
    )
