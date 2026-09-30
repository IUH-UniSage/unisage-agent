from collections.abc import Callable
from typing import Any

import pytest
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from app.core.registry.model_registry import CredentialConfig
from app.core.registry.model_router import ModelRouter
from app.graph.nodes.query_transformation import (
    SubQuery,
    build_decomposer_agent,
    build_query_transformation_agent,
    decompose_query,
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


class _FakeRedis:
    """No-expiry stand-in for `redis.asyncio.Redis` - a key marked once stays
    marked for the test's lifetime, same shape as
    `tests/api/test_chat_stream_errors.py`'s fixture."""

    def __init__(self) -> None:
        self._blocked: set[str] = set()

    async def set(self, name: str, _value: Any, *, ex: int | None = None) -> Any:
        del ex
        self._blocked.add(name)
        return True

    async def exists(self, name: str) -> int:
        return 1 if name in self._blocked else 0

    async def aclose(self) -> Any:
        return None


class _FakeBackendClient:
    def __init__(self) -> None:
        self.reports: list[dict[str, Any]] = []

    async def report_health(self, **kwargs: Any) -> None:
        self.reports.append(kwargs)


def _credential(credential_id: str, priority: int) -> CredentialConfig:
    return CredentialConfig(
        id=credential_id,
        revision=1,
        source_type="CLOUD_API",
        provider="google",
        model_name="gemini-2.5-flash",
        api_base_url="https://generativelanguage.googleapis.com",
        priority=priority,
        max_rpm=60,
        api_key="key",
    )


@pytest.mark.asyncio
async def test_transform_query_fails_over_to_the_next_credential_on_a_transient_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Same gap this closes for classification (see
    test_message_classification_node.py) applies to QueryTransformationNode's
    HyDE call: a transient provider error before `agent.run()` returns must
    retry with the next ACTIVE credential instead of crashing the graph run."""

    primary = _credential("cred-primary", priority=1)
    fallback = _credential("cred-fallback", priority=2)
    router = ModelRouter(redis_client=_FakeRedis(), backend_client=_FakeBackendClient())
    monkeypatch.setattr(
        "app.core.registry.model_router.active_credentials_for", lambda purpose: [primary, fallback]
    )

    def failing_function(_messages: list[ModelMessage], _agent_info: AgentInfo) -> ModelResponse:
        raise RuntimeError("primary down")

    def fallback_function(_messages: list[ModelMessage], _agent_info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[TextPart(content="hyde doc from fallback")])

    def fake_build_model(credential: CredentialConfig) -> FunctionModel:
        assert credential.id == fallback.id
        return FunctionModel(fallback_function)

    monkeypatch.setattr("app.graph.streaming.build_model", fake_build_model)

    agent = build_query_transformation_agent(FunctionModel(failing_function))

    hyde = await transform_query(
        agent,
        "Sinh viên năm cuối có được miễn GDQP không?",
        purpose="CHAT",
        credential=primary,
        snapshot_version=1,
        agent_factory=build_query_transformation_agent,
        router=router,
    )

    assert hyde == "hyde doc from fallback"


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

    sub_queries = await transform_tasks(agent, tasks)

    assert sub_queries == [
        SubQuery(question="Học phí ngành CNTT?", retrieval_text="HyDE: Học phí ngành CNTT?"),
        SubQuery(question="Điều kiện học bổng?", retrieval_text="HyDE: Điều kiện học bổng?"),
    ]


@pytest.mark.asyncio
async def test_transform_tasks_single_task_matches_transform_query() -> None:
    agent = build_query_transformation_agent(_echo_first_line_model())
    task = ClassifiedTask(intent="academic_advisory", query="Thủ tục bảo lưu?")

    (sub_query,) = await transform_tasks(agent, [(task, "SINGLE")])

    assert sub_query.retrieval_text == await transform_query(agent, "Thủ tục bảo lưu?")


def _fixed_model(text: str) -> FunctionModel:
    def function(_messages: list[ModelMessage], _agent_info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[TextPart(content=text)])

    return FunctionModel(function=function)


_COMPARISON = "Ngành CNTT và Kế toán học phí chênh bao nhiêu?"
_DECOMPOSED = (
    '{"sub_queries": ["Định mức học phí ngành Công nghệ thông tin", '
    '"Định mức học phí ngành Kế toán"]}'
)


@pytest.mark.parametrize(
    ("output", "expected"),
    [
        (
            _DECOMPOSED,
            ["Định mức học phí ngành Công nghệ thông tin", "Định mức học phí ngành Kế toán"],
        ),
        (
            "```json\n" + _DECOMPOSED + "\n```",
            ["Định mức học phí ngành Công nghệ thông tin", "Định mức học phí ngành Kế toán"],
        ),
        ('{"sub_queries": ["a", "b", "c", "d"]}', ["a", "b", "c"]),
        ('{"sub_queries": ["a", "a", " ", 5]}', ["a"]),
        ("không phải JSON", []),
        ('{"sub_queries": "a"}', []),
    ],
)
@pytest.mark.asyncio
async def test_decompose_query_parses_and_bounds_sub_queries(
    output: str, expected: list[str]
) -> None:
    agent = build_decomposer_agent(_fixed_model(output))

    assert await decompose_query(agent, _COMPARISON) == expected


@pytest.mark.asyncio
async def test_multi_task_uses_the_decomposer_instead_of_hyde() -> None:
    hyde_calls: list[str] = []

    def hyde(messages: list[ModelMessage], _agent_info: AgentInfo) -> ModelResponse:
        hyde_calls.append("called")
        return ModelResponse(parts=[TextPart(content="HyDE")])

    task = ClassifiedTask(intent="academic_advisory", query=_COMPARISON, routing_mode="MULTI")

    sub_queries = await transform_tasks(
        build_query_transformation_agent(FunctionModel(hyde)),
        [(task, "MULTI")],
        decomposer_agent=build_decomposer_agent(_fixed_model(_DECOMPOSED)),
    )

    assert hyde_calls == []
    assert [sq.question for sq in sub_queries] == [
        "Định mức học phí ngành Công nghệ thông tin",
        "Định mức học phí ngành Kế toán",
    ]
    assert all(sq.question == sq.retrieval_text for sq in sub_queries)


@pytest.mark.parametrize("decomposer_output", ["rác", '{"sub_queries": ["chỉ một câu"]}'])
@pytest.mark.asyncio
async def test_multi_task_falls_back_to_hyde_when_decomposition_is_unusable(
    decomposer_output: str,
) -> None:
    task = ClassifiedTask(intent="academic_advisory", query=_COMPARISON, routing_mode="MULTI")

    (sub_query,) = await transform_tasks(
        build_query_transformation_agent(_fixed_model("HyDE doc")),
        [(task, "MULTI")],
        decomposer_agent=build_decomposer_agent(_fixed_model(decomposer_output)),
    )

    assert sub_query == SubQuery(question=_COMPARISON, retrieval_text="HyDE doc")


@pytest.mark.asyncio
async def test_sub_queries_of_several_tasks_are_flattened_in_task_order() -> None:
    single = ClassifiedTask(intent="academic_advisory", query="Điều kiện học bổng?")
    comparison = ClassifiedTask(intent="academic_advisory", query=_COMPARISON, routing_mode="MULTI")
    tasks: list[tuple[ClassifiedTask, RoutingMode]] = [(single, "SINGLE"), (comparison, "MULTI")]

    sub_queries = await transform_tasks(
        build_query_transformation_agent(_fixed_model("HyDE doc")),
        tasks,
        decomposer_agent=build_decomposer_agent(_fixed_model(_DECOMPOSED)),
    )

    assert [sq.question for sq in sub_queries] == [
        "Điều kiện học bổng?",
        "Định mức học phí ngành Công nghệ thông tin",
        "Định mức học phí ngành Kế toán",
    ]
