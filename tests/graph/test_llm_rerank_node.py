"""LLMRerankNode - narrowing reranked chunks to the ones each sub-query's
answer is actually in, and failing open, with model doubles (no network)."""

import pytest
from pydantic_ai.exceptions import ModelHTTPError
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from app.core.config import settings
from app.graph.nodes.llm_rerank import build_llm_rerank_agent, llm_rerank
from app.graph.nodes.post_retrieval_rerank import rerank_chunks
from app.schemas.retrieval import RetrievedChunk
from tests.llm_mocks import make_sync_llm_model


@pytest.fixture(autouse=True)
def _no_threshold(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "CHAT_RERANK_SCORE_THRESHOLD", 0.0)


def _chunk(chunk_id: str, score: float, content: str = "", heading: str = "") -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        content=content or chunk_id,
        source=f"{chunk_id}.pdf",
        score=score,
        heading_path=[heading] if heading else [],
    )


_CAO_HOC = _chunk("cao-hoc", 0.9, "Cao học Khối Công nghệ 48.000.000", "2 Cao học")
_CHINH_QUY = _chunk("chinh-quy", 0.8, "Khối Công nghệ 38.350.000", "3 Đại học chính quy")
_TO_HOP = _chunk("to-hop", 0.85, "Nhóm ngành Công nghệ thông tin 7480201 Toán, Vật lí")


def _two_sub_queries() -> object:
    # SQ1 "chương trình khung CNTT" retrieved only the admission table;
    # SQ2 "học phí CNTT chính quy" retrieved both tuition tables.
    return rerank_chunks([[_TO_HOP], [_CAO_HOC, _CHINH_QUY]])


@pytest.mark.asyncio
async def test_keyword_only_matches_are_dropped_and_their_sub_query_fails() -> None:
    # Candidates are numbered in first-seen order: C1 to-hop, C2 cao-hoc, C3 chinh-quy.
    agent = build_llm_rerank_agent(
        make_sync_llm_model('{"SQ1": {}, "SQ2": {"C3": "có dòng Khối Công nghệ chính quy"}}')
    )

    outcome = await llm_rerank(
        agent, ["Chương trình khung ngành CNTT", "Học phí CNTT chính quy"], _two_sub_queries()
    )

    assert outcome.failure is None
    assert outcome.result.failed_query_indexes == [0]
    assert [chunk.chunk_id for chunk in outcome.result.chunks] == ["chinh-quy"]


@pytest.mark.asyncio
async def test_a_chunk_found_by_one_sub_query_can_answer_another() -> None:
    agent = build_llm_rerank_agent(make_sync_llm_model('{"SQ1": [3], "SQ2": [3]}'))

    outcome = await llm_rerank(agent, ["a", "b"], _two_sub_queries())

    assert outcome.result.failed_query_indexes == []
    assert [c.chunk_id for c in outcome.result.per_query[0].chunks] == ["chinh-quy"]


@pytest.mark.asyncio
async def test_prompt_lists_sub_queries_and_numbered_trimmed_chunks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "CHAT_LLM_RERANK_SNIPPET_CHARS", 100)
    seen: list[str] = []

    def judge(messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
        user_prompt = messages[-1].parts[-1]
        seen.append(str(getattr(user_prompt, "content", "")))
        return ModelResponse(parts=[TextPart(content='{"SQ1": [1]}')])

    long_chunk = _chunk("long", 0.9, "x" * 500, "3 Đại học chính quy")
    await llm_rerank(
        build_llm_rerank_agent(FunctionModel(function=judge)),
        ["Học phí chính quy"],
        rerank_chunks([[long_chunk]]),
    )

    (prompt,) = seen
    assert "SQ1. Học phí chính quy" in prompt
    assert "[C1] (Tài liệu: long.pdf; Mục: 3 Đại học chính quy)" in prompt
    assert "x" * 100 in prompt and "x" * 101 not in prompt


@pytest.mark.parametrize("output", ["không phải JSON", '{"foo": [1]}', "[1, 2]"])
@pytest.mark.asyncio
async def test_malformed_output_keeps_the_score_ranking(output: str) -> None:
    before = _two_sub_queries()

    outcome = await llm_rerank(
        build_llm_rerank_agent(make_sync_llm_model(output)), ["a", "b"], before
    )

    assert outcome.result is before
    assert outcome.failure is not None
    assert "không đúng định dạng" in outcome.failure


@pytest.mark.asyncio
async def test_a_failing_model_keeps_the_score_ranking_and_names_the_cause() -> None:
    def broken(_messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
        raise ModelHTTPError(401, "gpt-x")

    before = _two_sub_queries()

    outcome = await llm_rerank(
        build_llm_rerank_agent(FunctionModel(function=broken)), ["a", "b"], before
    )

    assert outcome.result is before
    assert outcome.failure is not None
    assert outcome.failure.startswith("Mô hình Extraction")


@pytest.mark.asyncio
async def test_no_chunks_means_no_model_call() -> None:
    def must_not_run(_messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
        raise AssertionError("no chunk to judge")

    empty = rerank_chunks([[]])

    outcome = await llm_rerank(
        build_llm_rerank_agent(FunctionModel(function=must_not_run)), ["a"], empty
    )

    assert outcome.result is empty


@pytest.mark.asyncio
async def test_a_chunk_kept_without_a_reason_is_dropped() -> None:
    output = '{"SQ1": {"C1": ""}, "SQ2": {"C3": "có dòng Khối Công nghệ chính quy", "C2": " "}}'

    outcome = await llm_rerank(
        build_llm_rerank_agent(make_sync_llm_model(output)), ["a", "b"], _two_sub_queries()
    )

    assert outcome.result.failed_query_indexes == [0]
    assert [chunk.chunk_id for chunk in outcome.result.chunks] == ["chinh-quy"]


@pytest.mark.asyncio
async def test_each_sub_query_decision_is_logged_with_its_reason(
    caplog: pytest.LogCaptureFixture,
) -> None:
    output = '{"SQ1": {}, "SQ2": {"C3": "có dòng Khối Công nghệ chính quy"}}'

    with caplog.at_level("INFO", logger="app.graph.nodes.llm_rerank"):
        await llm_rerank(
            build_llm_rerank_agent(make_sync_llm_model(output)),
            ["Chương trình khung CNTT", "Học phí chính quy"],
            _two_sub_queries(),
        )

    assert "SQ1 'Chương trình khung CNTT': kept []; dropped [C1 to-hop" in caplog.text
    assert "kept [C3 chinh-quy > 3 Đại học chính quy (0.80): có dòng Khối Công nghệ" in caplog.text
