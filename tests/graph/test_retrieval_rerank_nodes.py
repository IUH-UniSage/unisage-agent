import asyncio
from dataclasses import dataclass, field

import pytest

from app.core.config import settings
from app.graph.nodes.post_retrieval_rerank import rerank_chunks
from app.graph.nodes.retrieval_filtering import retrieve_chunks
from app.schemas.retrieval import RetrievedChunk
from app.schemas.security import AcademicSecurityContext
from tests.llm_mocks import FakeRetrievalService, RetrieveManyMixin


def test_retrieve_chunks_delegates_to_the_injected_retrieval_service() -> None:
    service = FakeRetrievalService(
        [RetrievedChunk(chunk_id="a", content="x", source="s", score=0.9)]
    )

    (chunks,) = asyncio.run(
        retrieve_chunks(["quy chế đào tạo"], service, AcademicSecurityContext())
    )

    assert [c.chunk_id for c in chunks] == ["a"]


def test_rerank_chunks_filters_below_threshold(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "CHAT_RERANK_SCORE_THRESHOLD", 0.9)
    chunks = [
        RetrievedChunk(chunk_id="a", content="x", source="s", score=0.95),
        RetrievedChunk(chunk_id="b", content="y", source="s", score=0.5),
    ]

    result = rerank_chunks([chunks])

    assert result.has_valid_context is True
    assert [c.chunk_id for c in result.chunks] == ["a"]


def test_rerank_chunks_no_valid_context_when_all_below_threshold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "CHAT_RERANK_SCORE_THRESHOLD", 0.9)
    chunks = [RetrievedChunk(chunk_id="a", content="x", source="s", score=0.1)]

    result = rerank_chunks([chunks])

    assert result.has_valid_context is False
    assert result.chunks == []
    assert result.failed_query_indexes == [0]


@dataclass
class _PerQueryRetrieval(RetrieveManyMixin):
    """Returns a different fixed chunk list per query, and records the order
    queries were searched in."""

    results: dict[str, list[RetrievedChunk]]
    queries: list[str] = field(default_factory=list)
    limits: list[int | None] = field(default_factory=list)

    def retrieve(
        self, query: str, *, security: AcademicSecurityContext, limit: int | None = None
    ) -> list[RetrievedChunk]:
        del security
        self.queries.append(query)
        self.limits.append(limit)
        chunks = self.results[query]
        return chunks if limit is None else chunks[:limit]


def _chunk(chunk_id: str, score: float) -> RetrievedChunk:
    return RetrievedChunk(chunk_id=chunk_id, content=chunk_id, source="s", score=score)


def _merged(queries: list[str], service: _PerQueryRetrieval) -> list[RetrievedChunk]:
    return rerank_chunks(
        asyncio.run(retrieve_chunks(queries, service, AcademicSecurityContext()))
    ).chunks


@pytest.fixture
def _no_threshold(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "CHAT_RERANK_SCORE_THRESHOLD", 0.0)


@pytest.mark.usefixtures("_no_threshold")
def test_every_query_is_searched_in_order_and_merged_by_best_score() -> None:
    service = _PerQueryRetrieval(
        {
            "hoc phi": [_chunk("a", 0.7), _chunk("shared", 0.6)],
            "hoc bong": [_chunk("shared", 0.9), _chunk("b", 0.5)],
        }
    )

    chunks = _merged(["hoc phi", "hoc bong"], service)

    assert service.queries == ["hoc phi", "hoc bong"]
    assert [(c.chunk_id, c.score) for c in chunks] == [("shared", 0.9), ("a", 0.7), ("b", 0.5)]


@pytest.mark.usefixtures("_no_threshold")
def test_merge_keeps_the_earlier_query_copy_on_a_score_tie() -> None:
    first = RetrievedChunk(chunk_id="x", content="from first", source="s", score=0.8)
    second = RetrievedChunk(chunk_id="x", content="from second", source="s", score=0.8)
    service = _PerQueryRetrieval({"q1": [first], "q2": [second]})

    (chunk,) = _merged(["q1", "q2"], service)

    assert chunk.content == "from first"


@pytest.mark.usefixtures("_no_threshold")
def test_merge_caps_the_merged_list(monkeypatch: pytest.MonkeyPatch) -> None:
    # 3 candidate slots → quota 1 each; the prompt takes only the best 2.
    monkeypatch.setattr(settings, "CHAT_RETRIEVAL_MAX_CHUNKS", 3)
    monkeypatch.setattr(settings, "CHAT_CONTEXT_MAX_CHUNKS", 2)
    service = _PerQueryRetrieval(
        {"q1": [_chunk("a", 0.9)], "q2": [_chunk("b", 0.8)], "q3": [_chunk("c", 0.7)]}
    )

    chunks = _merged(["q1", "q2", "q3"], service)

    assert [c.chunk_id for c in chunks] == ["a", "b"]


@pytest.mark.usefixtures("_no_threshold")
def test_retrieve_chunks_gives_each_query_an_equal_quota(monkeypatch: pytest.MonkeyPatch) -> None:
    """A high-scoring sub-query must not fill the whole top-k: with 8 slots
    and 2 queries, each contributes at most 4 chunks."""

    monkeypatch.setattr(settings, "CHAT_RETRIEVAL_MAX_CHUNKS", 8)
    strong = [_chunk(f"cntt-{i}", 0.9 - i * 0.01) for i in range(8)]
    weak = [_chunk(f"ketoan-{i}", 0.5 - i * 0.01) for i in range(8)]
    service = _PerQueryRetrieval({"cntt": strong, "ketoan": weak})

    chunks = _merged(["cntt", "ketoan"], service)

    assert service.limits == [4, 4]
    ids = [c.chunk_id for c in chunks]
    assert sum(i.startswith("cntt") for i in ids) == 4
    assert sum(i.startswith("ketoan") for i in ids) == 4


def test_retrieve_chunks_single_query_keeps_the_default_limit() -> None:
    service = _PerQueryRetrieval({"q": [_chunk("a", 0.9)]})

    asyncio.run(retrieve_chunks(["q"], service, AcademicSecurityContext()))

    assert service.limits == [None]


def test_rerank_reports_which_sub_query_found_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "CHAT_RERANK_SCORE_THRESHOLD", 0.7)
    service = _PerQueryRetrieval(
        {"hoc phi": [_chunk("a", 0.9)], "lich thi": [_chunk("b", 0.3)], "hoc bong": []}
    )

    result = rerank_chunks(
        asyncio.run(
            retrieve_chunks(["hoc phi", "lich thi", "hoc bong"], service, AcademicSecurityContext())
        )
    )

    assert result.has_valid_context is True
    # Worst miss first: "hoc bong" found nothing (0.0), "lich thi" a 0.3 chunk.
    assert result.failed_query_indexes == [2, 1]
    assert [c.chunk_id for c in result.chunks] == ["a"]


def test_threshold_then_cap_equals_the_old_cap_then_threshold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Old flow merged + capped every query's chunks first, then thresholded;
    per-query rerank thresholds first. Both must keep the same chunks."""

    monkeypatch.setattr(settings, "CHAT_CONTEXT_MAX_CHUNKS", 3)
    monkeypatch.setattr(settings, "CHAT_RERANK_SCORE_THRESHOLD", 0.6)
    service = _PerQueryRetrieval(
        {
            "q1": [_chunk("a", 0.95), _chunk("b", 0.65)],
            "q2": [_chunk("c", 0.9), _chunk("d", 0.55)],
        }
    )

    chunks = _merged(["q1", "q2"], service)

    # Old: merge+cap → a .95, c .9, b .65 → threshold .6 → same three.
    assert [c.chunk_id for c in chunks] == ["a", "c", "b"]


@pytest.mark.usefixtures("_no_threshold")
def test_context_cap_trims_a_single_query_but_keeps_the_rerank_pool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "CHAT_CONTEXT_MAX_CHUNKS", 2)
    candidates = [_chunk(f"c{i}", 0.9 - i * 0.01) for i in range(5)]

    result = rerank_chunks([candidates])

    assert [c.chunk_id for c in result.chunks] == ["c0", "c1"]
    assert len(result.per_query[0].chunks) == 5
