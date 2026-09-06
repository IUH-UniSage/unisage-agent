from collections.abc import Callable, Sequence

import pytest
from pydantic_ai.models.function import FunctionModel

from app.core.graph_trace import GraphTrace
from app.graph.nodes.generation_synthesis import (
    build_generation_agent,
    collect_confirmed_metadata_updates,
    collect_pending_clarification,
    run_generation_synthesis,
)
from app.schemas.clarification import PendingClarification
from app.schemas.retrieval import RetrievedChunk
from app.schemas.security import AcademicSecurityContext

_TRACE = GraphTrace(conversation_id="conv-1", message_id="msg-1", user_id=None, client_ip=None)


@pytest.mark.asyncio
async def test_streams_full_response_and_no_pending_when_no_json_block(
    mock_streaming_llm_model: Callable[[Sequence[str]], FunctionModel],
) -> None:
    agent = build_generation_agent(
        mock_streaming_llm_model(["Câu trả lời cuối cùng ", "kèm trích dẫn [1]."])
    )
    received: list[str] = []

    async def sink(token: str) -> None:
        received.append(token)

    result = await run_generation_synthesis(
        agent,
        user_query="Hạn đăng ký học kỳ này là khi nào?",
        security=AcademicSecurityContext(),
        confirmed_metadata={},
        chunks=[RetrievedChunk(chunk_id="c1", content="nội dung", source="s", score=0.9)],
        previous_pending=None,
        origin_node="QueryTransformationNode",
        token_sink=sink,
        trace=_TRACE,
    )

    assert result.response_text == "Câu trả lời cuối cùng kèm trích dẫn [1]."
    assert result.pending_clarification is None
    assert "".join(received) == result.response_text


@pytest.mark.asyncio
async def test_extracts_pending_clarification_from_trailing_json_block(
    mock_streaming_llm_model: Callable[[Sequence[str]], FunctionModel],
) -> None:
    response_with_json = (
        "Quy định miễn giảm khác nhau tuỳ hệ đào tạo.\n\n"
        "```json\n"
        '{"type": "ask_user_form", "fields": [{"field": "training_type", '
        '"options": [{"id": "chinh_quy"}, {"id": "lien_thong"}]}]}\n'
        "```"
    )
    agent = build_generation_agent(mock_streaming_llm_model([response_with_json]))

    async def sink(_token: str) -> None:
        return None

    result = await run_generation_synthesis(
        agent,
        user_query="...",
        security=AcademicSecurityContext(),
        confirmed_metadata={},
        chunks=[],
        previous_pending=None,
        origin_node="QueryTransformationNode",
        token_sink=sink,
        trace=_TRACE,
    )

    assert result.pending_clarification is not None
    assert result.pending_clarification.origin_node == "QueryTransformationNode"
    assert result.pending_clarification.missing_fields == ["training_type"]
    assert result.pending_clarification.options == [["chinh_quy", "lien_thong"]]
    assert result.pending_clarification.retry_count == 0


def test_collect_pending_clarification_keeps_retry_count_when_same_field() -> None:
    previous = PendingClarification(
        origin_node="QueryTransformationNode",
        missing_fields=["training_type"],
        options=[["chinh_quy", "lien_thong"]],
        retry_count=1,
    )
    text = (
        "```json\n"
        '{"type": "ask_user_form", "fields": [{"field": "training_type", '
        '"options": [{"id": "chinh_quy"}, {"id": "lien_thong"}]}]}\n'
        "```"
    )

    result = collect_pending_clarification(
        text, origin_node="QueryTransformationNode", previous=previous
    )

    assert result is not None
    assert result.retry_count == 1


def test_collect_pending_clarification_resets_retry_count_for_new_field() -> None:
    previous = PendingClarification(
        origin_node="QueryTransformationNode",
        missing_fields=["training_type"],
        options=[["chinh_quy", "lien_thong"]],
        retry_count=2,
    )
    text = (
        "```json\n"
        '{"type": "ask_user_form", "fields": [{"field": "khoa_nhap_hoc", "options": null}]}\n'
        "```"
    )

    result = collect_pending_clarification(
        text, origin_node="QueryTransformationNode", previous=previous
    )

    assert result is not None
    assert result.retry_count == 0
    assert result.missing_fields == ["khoa_nhap_hoc"]
    assert result.options == [None]


@pytest.mark.asyncio
async def test_repairs_missing_ask_form_when_prose_asks_for_missing_attribute(
    mock_sequential_streaming_llm_model: Callable[[Sequence[Sequence[str]]], FunctionModel],
) -> None:
    """Known model failure mode: a clarification request in prose, without the
    mandatory JSON block. The repair follow-up call (second scripted response)
    should supply it, and it should reach the caller both in `response_text`
    and via `token_sink`."""

    prose_without_json = "Bạn vui lòng cho biết ngành học của bạn để mình tra học phí nhé!"
    repair_json = (
        '```json\n{"type": "ask_user_form", "fields": [{"field": "nganh", '
        '"options": [{"id": "cntt"}, {"id": "logistics"}]}]}\n```'
    )
    agent = build_generation_agent(
        mock_sequential_streaming_llm_model([[prose_without_json], [repair_json]])
    )
    received: list[str] = []

    async def sink(token: str) -> None:
        received.append(token)

    result = await run_generation_synthesis(
        agent,
        user_query="học phí ngành tôi học là bao nhiêu?",
        security=AcademicSecurityContext(),
        confirmed_metadata={},
        chunks=[],
        previous_pending=None,
        origin_node="QueryTransformationNode",
        token_sink=sink,
        trace=_TRACE,
    )

    assert result.pending_clarification is not None
    assert result.pending_clarification.missing_fields == ["nganh"]
    assert result.pending_clarification.options == [["cntt", "logistics"]]
    assert prose_without_json in result.response_text
    assert '"type": "ask_user_form"' in result.response_text
    # The repaired JSON block must have reached the client via token_sink too,
    # not just the returned response_text.
    assert '"type": "ask_user_form"' in "".join(received)


@pytest.mark.asyncio
async def test_does_not_repair_when_prose_is_not_a_clarification_request(
    mock_streaming_llm_model: Callable[[Sequence[str]], FunctionModel],
) -> None:
    """A normal answer that happens to contain no JSON block must NOT trigger
    the repair call - only text that heuristically reads like a clarification
    request does."""

    agent = build_generation_agent(
        mock_streaming_llm_model(["Hạn nộp học phí học kỳ này là 15/03 [1]."])
    )

    async def sink(_token: str) -> None:
        return None

    result = await run_generation_synthesis(
        agent,
        user_query="hạn nộp học phí học kỳ này là khi nào?",
        security=AcademicSecurityContext(),
        confirmed_metadata={},
        chunks=[],
        previous_pending=None,
        origin_node="QueryTransformationNode",
        token_sink=sink,
        trace=_TRACE,
    )

    assert result.pending_clarification is None
    assert result.response_text == "Hạn nộp học phí học kỳ này là 15/03 [1]."


def test_collect_pending_clarification_none_when_no_json_block() -> None:
    result = collect_pending_clarification(
        "Câu trả lời bình thường, không hỏi lại.",
        origin_node="QueryTransformationNode",
        previous=None,
    )

    assert result is None


def test_collect_confirmed_metadata_updates_accepts_fields_that_were_asked_about() -> None:
    previous = PendingClarification(
        origin_node="QueryTransformationNode",
        missing_fields=["he_dao_tao", "khoa_nhap_hoc"],
        options=[["chinh_quy", "lien_thong"], ["k21", "k22"]],
        retry_count=0,
    )
    text = (
        "Bạn học chính quy khóa 21 nhé.\n\n"
        "```json\n"
        '{"type": "confirmed_metadata", "fields": {"he_dao_tao": "chinh_quy", '
        '"khoa_nhap_hoc": "k21"}}\n'
        "```"
    )

    result = collect_confirmed_metadata_updates(text, previous=previous)

    assert result == {"he_dao_tao": "chinh_quy", "khoa_nhap_hoc": "k21"}


def test_collect_confirmed_metadata_updates_drops_fields_not_asked_about() -> None:
    """A model that misreads the instruction and confirms something outside
    this turn's pending fields must not be able to inject arbitrary metadata."""

    previous = PendingClarification(
        origin_node="QueryTransformationNode",
        missing_fields=["he_dao_tao"],
        options=[["chinh_quy", "lien_thong"]],
        retry_count=0,
    )
    text = (
        "```json\n"
        '{"type": "confirmed_metadata", "fields": {"he_dao_tao": "chinh_quy", '
        '"some_other_field": "hacked"}}\n'
        "```"
    )

    result = collect_confirmed_metadata_updates(text, previous=previous)

    assert result == {"he_dao_tao": "chinh_quy"}


def test_collect_confirmed_metadata_updates_empty_when_no_pending() -> None:
    text = '```json\n{"type": "confirmed_metadata", "fields": {"he_dao_tao": "chinh_quy"}}\n```'

    result = collect_confirmed_metadata_updates(text, previous=None)

    assert result == {}


@pytest.mark.asyncio
async def test_run_generation_synthesis_merges_confirmed_metadata_fallback(
    mock_streaming_llm_model: Callable[[Sequence[str]], FunctionModel],
) -> None:
    previous_pending = PendingClarification(
        origin_node="QueryTransformationNode",
        missing_fields=["he_dao_tao"],
        options=[["chinh_quy", "lien_thong"]],
        retry_count=0,
    )
    response_with_confirmation = (
        "Bạn học hệ chính quy nên được áp dụng quy định X [1].\n\n"
        "```json\n"
        '{"type": "confirmed_metadata", "fields": {"he_dao_tao": "chinh_quy"}}\n'
        "```"
    )
    agent = build_generation_agent(mock_streaming_llm_model([response_with_confirmation]))

    async def sink(_token: str) -> None:
        return None

    result = await run_generation_synthesis(
        agent,
        user_query="tôi học chính quy á",
        security=AcademicSecurityContext(),
        confirmed_metadata={"existing": "value"},
        chunks=[],
        previous_pending=previous_pending,
        origin_node="QueryTransformationNode",
        token_sink=sink,
        trace=_TRACE,
    )

    assert result.confirmed_metadata == {"existing": "value", "he_dao_tao": "chinh_quy"}
    assert result.pending_clarification is None
