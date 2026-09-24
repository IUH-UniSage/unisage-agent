from dataclasses import dataclass, field

import pytest

from app.core.config import settings
from app.graph.nodes.post_retrieval_rerank import rerank_chunks
from app.graph.nodes.retrieval_filtering import retrieve_chunks
from app.schemas.retrieval import RetrievedChunk
from app.schemas.security import AcademicSecurityContext
from tests.llm_mocks import FakeRetrievalService


def test_retrieve_chunks_delegates_to_the_injected_retrieval_service() -> None:
    service = FakeRetrievalService(
        [RetrievedChunk(chunk_id="a", content="x", source="s", score=0.9)]
    )

    chunks = retrieve_chunks(["quy chế đào tạo"], service, AcademicSecurityContext())

    assert [c.chunk_id for c in chunks] == ["a"]


def test_rerank_chunks_filters_below_threshold(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "CHAT_RERANK_SCORE_THRESHOLD", 0.9)
    chunks = [
        RetrievedChunk(chunk_id="a", content="x", source="s", score=0.95),
        RetrievedChunk(chunk_id="b", content="y", source="s", score=0.5),
    ]

    result = rerank_chunks(chunks)

    assert result.has_valid_context is True
    assert [c.chunk_id for c in result.chunks] == ["a"]


def test_rerank_chunks_no_valid_context_when_all_below_threshold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "CHAT_RERANK_SCORE_THRESHOLD", 0.9)
    chunks = [RetrievedChunk(chunk_id="a", content="x", source="s", score=0.1)]

    result = rerank_chunks(chunks)

    assert result.has_valid_context is False
    assert result.chunks == []


@dataclass
class _PerQueryRetrieval:
    """Returns a different fixed chunk list per query, and records the order
    queries were searched in."""

    results: dict[str, list[RetrievedChunk]]
    queries: list[str] = field(default_factory=list)

    def retrieve(
        self, query: str, *, security: AcademicSecurityContext, limit: int | None = None
    ) -> list[RetrievedChunk]:
        del security, limit
        self.queries.append(query)
        return self.results[query]


def _chunk(chunk_id: str, score: float) -> RetrievedChunk:
    return RetrievedChunk(chunk_id=chunk_id, content=chunk_id, source="s", score=score)


def test_retrieve_chunks_searches_every_query_in_order_and_merges_by_best_score() -> None:
    service = _PerQueryRetrieval(
        {
            "hoc phi": [_chunk("a", 0.7), _chunk("shared", 0.6)],
            "hoc bong": [_chunk("shared", 0.9), _chunk("b", 0.5)],
        }
    )

    chunks = retrieve_chunks(["hoc phi", "hoc bong"], service, AcademicSecurityContext())

    assert service.queries == ["hoc phi", "hoc bong"]
    assert [(c.chunk_id, c.score) for c in chunks] == [("shared", 0.9), ("a", 0.7), ("b", 0.5)]


def test_retrieve_chunks_keeps_the_earlier_query_copy_on_a_score_tie() -> None:
    first = RetrievedChunk(chunk_id="x", content="from first", source="s", score=0.8)
    second = RetrievedChunk(chunk_id="x", content="from second", source="s", score=0.8)
    service = _PerQueryRetrieval({"q1": [first], "q2": [second]})

    (chunk,) = retrieve_chunks(["q1", "q2"], service, AcademicSecurityContext())

    assert chunk.content == "from first"


def test_retrieve_chunks_caps_the_merged_list(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "CHAT_RETRIEVAL_MAX_CHUNKS", 2)
    service = _PerQueryRetrieval(
        {"q1": [_chunk("a", 0.9), _chunk("b", 0.8)], "q2": [_chunk("c", 0.7)]}
    )

    chunks = retrieve_chunks(["q1", "q2"], service, AcademicSecurityContext())

    assert [c.chunk_id for c in chunks] == ["a", "b"]
