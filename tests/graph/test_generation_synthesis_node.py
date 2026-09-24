from collections.abc import Callable, Sequence

import pytest
from pydantic_ai.models.function import FunctionModel

from app.core.config import settings
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
        '{"type": "ask_user_form", "fields": [{"field": "khoa_nhap_hoc", '
        '"options": [{"id": "k21"}, {"id": "k22"}]}]}\n'
        "```"
    )

    result = collect_pending_clarification(
        text, origin_node="QueryTransformationNode", previous=previous
    )

    assert result is not None
    assert result.retry_count == 0
    assert result.missing_fields == ["khoa_nhap_hoc"]
    assert result.options == [["k21", "k22"]]


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
async def test_skips_repair_call_when_allow_repair_json_is_false(
    mock_sequential_streaming_llm_model: Callable[[Sequence[Sequence[str]]], FunctionModel],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`CHAT_ALLOW_REPAIR_JSON=false` must skip `_repair_missing_ask_form`
    entirely, not just suppress its effect - only ONE response is scripted
    below (unlike `test_repairs_missing_ask_form_when_prose_asks_for_missing
    _attribute`'s two), so a repair call that fired anyway would exhaust the
    mock and fail the test."""

    monkeypatch.setattr(settings, "CHAT_ALLOW_REPAIR_JSON", False)
    prose_without_json = "Bạn vui lòng cho biết ngành học của bạn để mình tra học phí nhé!"
    agent = build_generation_agent(mock_sequential_streaming_llm_model([[prose_without_json]]))

    async def sink(_token: str) -> None:
        return None

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

    assert result.response_text == prose_without_json
    assert result.pending_clarification is None


@pytest.mark.asyncio
async def test_repairs_missing_ask_form_with_words_wedged_between_anchor_phrase(
    mock_sequential_streaming_llm_model: Callable[[Sequence[Sequence[str]]], FunctionModel],
) -> None:
    """Regression test for a real live miss: the heuristic's first version
    required "cung cấp" to be immediately followed by "thông tin", but the
    actual model output was "cung cấp CHO MÌNH thông tin" - words wedged in
    between made the fixed-phrase regex miss it entirely, so no repair fired
    and the turn genuinely lost the clarification request (see
    tasks/report.md)."""

    prose_without_json = (
        "Về mức học phí, xin vui lòng cung cấp cho mình thông tin về ngành học của bạn."
    )
    repair_json = (
        '```json\n{"type": "ask_user_form", "fields": [{"field": "nganh_hoc", '
        '"options": [{"id": "cntt", "label": "Công nghệ Thông tin"}]}]}\n```'
    )
    agent = build_generation_agent(
        mock_sequential_streaming_llm_model([[prose_without_json], [repair_json]])
    )

    async def sink(_token: str) -> None:
        return None

    result = await run_generation_synthesis(
        agent,
        user_query="học phí của ngành tôi học",
        security=AcademicSecurityContext(),
        confirmed_metadata={},
        chunks=[],
        previous_pending=None,
        origin_node="QueryTransformationNode",
        token_sink=sink,
        trace=_TRACE,
    )

    assert result.pending_clarification is not None
    assert result.pending_clarification.missing_fields == ["nganh_hoc"]


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


@pytest.mark.asyncio
async def test_does_not_repair_a_conditional_offer_about_an_unrelated_topic(
    mock_streaming_llm_model: Callable[[Sequence[str]], FunctionModel],
) -> None:
    """Live-observed bug: after fully answering an unrelated question, the
    model appended a generic conditional offer naming a DIFFERENT topic
    ('điều kiện học bổng') the user never asked about this turn. This reads
    like a clarification request to `_CLARIFICATION_PHRASE_PATTERN` alone
    (it contains "cho biết thêm thông tin"), but must NOT trigger repair -
    only ONE response is scripted below, so if repair fired it would run out
    of scripted turns and the test would fail with a mock exhaustion error."""

    prose = (
        "Chương trình đào tạo ngành Logistics gồm 141 tín chỉ, 4 năm [1]. "
        "Nếu bạn cần thêm thông tin chi tiết về điều kiện học bổng hoặc các "
        "thủ tục hành chính cụ thể cho ngành này, vui lòng cho biết thêm "
        "thông tin."
    )
    agent = build_generation_agent(mock_streaming_llm_model([prose]))

    async def sink(_token: str) -> None:
        return None

    result = await run_generation_synthesis(
        agent,
        user_query="chương trình đào tạo ngành Logistics như thế nào?",
        security=AcademicSecurityContext(),
        confirmed_metadata={},
        chunks=[],
        previous_pending=None,
        origin_node="QueryTransformationNode",
        token_sink=sink,
        trace=_TRACE,
    )

    assert result.pending_clarification is None
    assert result.response_text == prose


@pytest.mark.asyncio
async def test_does_not_repair_a_trailing_conditional_offer(
    mock_streaming_llm_model: Callable[[Sequence[str]], FunctionModel],
) -> None:
    """Live-observed bug: the earlier heuristic only recognized "Nếu bạn
    cần..." when it OPENED the sentence, missing the equally common reversed
    phrasing where the "nếu" clause trails the main clause - exactly what
    the model wrote here. Only ONE response is scripted below, so if repair
    (wrongly) fired it would run out of scripted turns and the test would
    fail with a mock exhaustion error."""

    prose = (
        "Mức thu học phí dành cho sinh viên hệ đại học chính quy khóa tuyển sinh "
        "năm học 2025-2026 thuộc khối Công nghệ là 38.350.000 đồng cho một năm "
        "học [1].\n\nBạn vui lòng cho tôi biết thêm thông tin nếu bạn cần hỗ trợ "
        "gì khác."
    )
    agent = build_generation_agent(mock_streaming_llm_model([prose]))

    async def sink(_token: str) -> None:
        return None

    result = await run_generation_synthesis(
        agent,
        user_query="học phí đại học chính quy khối công nghệ khóa 2025-2026 là bao nhiêu?",
        security=AcademicSecurityContext(),
        confirmed_metadata={},
        chunks=[],
        previous_pending=None,
        origin_node="QueryTransformationNode",
        token_sink=sink,
        trace=_TRACE,
    )

    assert result.pending_clarification is None
    assert result.response_text == prose


def test_collect_pending_clarification_none_when_no_json_block() -> None:
    result = collect_pending_clarification(
        "Câu trả lời bình thường, không hỏi lại.",
        origin_node="QueryTransformationNode",
        previous=None,
    )

    assert result is None


def test_collect_pending_clarification_none_when_fields_array_is_empty() -> None:
    """Known model slip: appending `{"type": "ask_user_form", "fields": []}`
    after a complete answer (often triggered by an innocuous closing courtesy
    line). An empty `fields` array must be treated like no block at all -
    never persisted as a hollow PendingClarification."""

    result = collect_pending_clarification(
        'Câu trả lời đầy đủ.\n\n```json\n{"type": "ask_user_form", "fields": []}\n```',
        origin_node="QueryTransformationNode",
        previous=None,
    )

    assert result is None


def test_collect_pending_clarification_none_when_lead_in_is_a_conditional_offer() -> None:
    """Live-observed bug via the PRIMARY generation call (no repair
    involved): the model attached a populated, hallucinated `ask_user_form`
    directly after a conditional offer ("Nếu bạn có nhu cầu tìm hiểu thêm...,
    vui lòng cho biết nhé!") for a topic the user never asked about. Must be
    dropped even though the JSON block itself is well-formed."""

    full_text = (
        "Chương trình đào tạo ngành Logistics gồm 141 tín chỉ [1]. "
        "Nếu bạn có nhu cầu tìm hiểu thêm chi tiết khác về ngành học, "
        "vui lòng cho biết nhé!\n\n"
        '```json\n{"type": "ask_user_form", "fields": [\n'
        '  {"field": "he_dao_tao", "label": "Hệ đào tạo", "options": ['
        '{"id": "chinh_quy", "label": "Chính quy"}, '
        '{"id": "lien_thong", "label": "Liên thông"}]}\n'
        "]}\n```"
    )

    result = collect_pending_clarification(
        full_text,
        origin_node="QueryTransformationNode",
        previous=None,
        user_query="chương trình đào tạo ngành Logistics như thế nào?",
    )

    assert result is None


def test_collect_pending_clarification_none_when_offer_split_across_two_sentences() -> None:
    """Live-observed bug (second occurrence, after the single-sentence-only
    guard already fixed the first one): "Nếu bạn cần X (...), Y." is its own
    sentence, and "Hãy cho mình biết nhé!" - the one that actually matches
    `_CLARIFICATION_PHRASE_PATTERN` - is a SEPARATE sentence right after it.
    Must still be dropped by looking one sentence back, not just at the
    matching sentence itself."""

    full_text = (
        "Hiện tại, mình chưa xác nhận được lịch thi cho kỳ này. "
        "Để có thông tin chính xác, bạn nên kiểm tra trang thông báo của trường. "
        "Nếu bạn cần thêm thông tin cụ thể hơn (ví dụ: năm học, kỳ thi cụ thể), "
        "mình có thể giúp bạn tìm kiếm thông tin đó. "
        "Hãy cho mình biết nhé!\n\n"
        '```json\n{"type": "ask_user_form", "fields": [\n'
        '  {"field": "khoa_nhap_hoc", "label": "Khóa nhập học", "options": ['
        '{"id": "k21", "label": "K21"}, {"id": "k22", "label": "K22"}]},\n'
        '  {"field": "he_dao_tao", "label": "Hệ đào tạo", "options": ['
        '{"id": "chinh_quy", "label": "Chính quy"}, '
        '{"id": "lien_thong", "label": "Liên thông"}]}\n'
        "]}\n```"
    )

    result = collect_pending_clarification(
        full_text,
        origin_node="QueryTransformationNode",
        previous=None,
        user_query="có lịch thi kì này chưa",
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
