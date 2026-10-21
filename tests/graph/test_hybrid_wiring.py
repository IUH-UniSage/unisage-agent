"""Hybrid retrieval in the graph (docs/specs/SPEC-hybrid-retrieval.md): the standalone
questions reach the retrieval service as lexical queries, and the score rerank applies the
dense floor when an LLM rerank follows, the old threshold when none does."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Sequence

import pytest
from pydantic_ai.models.function import FunctionModel

from app.core.config import settings
from app.core.observability.graph_trace import GraphTrace
from app.core.usage.usage_recorder import UsageRecorder
from app.graph.nodes.post_retrieval_rerank import rerank_chunks
from app.graph.nodes.retrieval_filtering import retrieve_chunks
from app.graph.streaming import TokenSink
from app.graph.streaming_graph import run_graph
from app.graph.streaming_state import GraphInput, GraphModels
from app.schemas.retrieval import RetrievedChunk
from app.schemas.security import AcademicSecurityContext
from tests.llm_mocks import FakeRetrievalService, make_classification_llm_model

_TRACE = GraphTrace(conversation_id="c1", message_id="m1", user_id=None, client_ip=None)
_CHUNK = RetrievedChunk(
    chunk_id="d:0", content="Học phần 2105618 có 3 tín chỉ", source="s", score=0.9
)


def _sink() -> TokenSink:
    async def sink(_token: str) -> None:
        return None

    return sink


def test_retrieve_chunks_forwards_the_lexical_queries() -> None:
    service = FakeRetrievalService([_CHUNK])

    asyncio.run(
        retrieve_chunks(
            ["HyDE passage"],
            service,
            AcademicSecurityContext(),
            lexical_queries=["Học phần 2105618 có bao nhiêu tín chỉ?"],
        )
    )

    assert service.last_lexical_queries == ["Học phần 2105618 có bao nhiêu tín chỉ?"]  # type: ignore[attr-defined]


def test_rerank_chunks_uses_the_given_floor_instead_of_the_threshold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "CHAT_RERANK_SCORE_THRESHOLD", 0.70)
    low = RetrievedChunk(chunk_id="a", content="x", source="s", score=0.50)

    assert rerank_chunks([[low]]).has_valid_context is False
    assert rerank_chunks([[low]], score_threshold=0.45).has_valid_context is True
    assert rerank_chunks([[low]], score_threshold=0.60).has_valid_context is False


@pytest.mark.asyncio
async def test_an_advisory_turn_sends_the_standalone_question_as_lexical_query(
    mock_sync_llm_model: Callable[[str], FunctionModel],
    mock_streaming_llm_model: Callable[[Sequence[str]], FunctionModel],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "CHAT_RERANK_SCORE_THRESHOLD", 0.0)
    service = FakeRetrievalService([_CHUNK])
    models = GraphModels(
        classification=make_classification_llm_model("academic_advisory"),
        # HyDE output: standalone question, blank line, hypothetical passage.
        query_transformation=mock_sync_llm_model(
            "Học phần 2105618 có bao nhiêu tín chỉ?\n\nHọc phần 2105618 gồm ... tín chỉ."
        ),
        generation=mock_streaming_llm_model(["Trả lời [1]."]),
        retrieval=service,
    )
    graph_input = GraphInput(
        conversation_id="c1",
        user_message="học phần 2105618 mấy tín?",
        is_first_turn=False,
        security=AcademicSecurityContext(),
    )

    result = await run_graph(
        graph_input, models, _sink(), _TRACE, UsageRecorder(request_id="t", purpose="CHAT")
    )

    assert result.response_text == "Trả lời [1]."
    # The lexical side gets the standalone question (first line), not the HyDE passage.
    assert service.last_lexical_queries == ["Học phần 2105618 có bao nhiêu tín chỉ?"]  # type: ignore[attr-defined]


def test_score_rerank_keeps_the_retrieval_order_so_lexical_hits_do_not_jump_the_queue(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """RRF put the dense hits first; a BM25-only hit's fixed 0.70 must not move it ahead of
    them (21/10/2026: with the old sort-by-score, 7 keyword-only chunks took the 8-chunk
    context and the tuition table was cut)."""

    monkeypatch.setattr(settings, "CHAT_CONTEXT_MAX_CHUNKS", 2)
    dense = [
        RetrievedChunk(chunk_id=f"d{i}", content="", source="s", score=0.60 - i * 0.01)
        for i in range(2)
    ]
    lexical = [
        RetrievedChunk(chunk_id=f"l{i}", content="", source="s", score=0.70) for i in range(3)
    ]

    result = rerank_chunks([dense + lexical], score_threshold=0.45)

    assert [c.chunk_id for c in result.per_query[0].chunks] == ["d0", "d1", "l0", "l1", "l2"]
    assert [c.chunk_id for c in result.chunks] == ["d0", "d1"]
