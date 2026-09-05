from collections.abc import Callable, Sequence

import pytest
from pydantic_ai.models.function import FunctionModel

from app.core.graph_trace import GraphTrace
from app.graph.nodes.generation_synthesis import (
    build_generation_agent,
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


def test_collect_pending_clarification_none_when_no_json_block() -> None:
    result = collect_pending_clarification(
        "Câu trả lời bình thường, không hỏi lại.",
        origin_node="QueryTransformationNode",
        previous=None,
    )

    assert result is None
