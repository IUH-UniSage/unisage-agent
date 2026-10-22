"""CrossEncoderRerankNode - top-K per sub-query by cross-encoder score, scores left
alone, fail-over on a service error; a client double, no HTTP."""

import pytest

from app.core.config import settings
from app.graph.nodes.cross_encoder_rerank import (
    CE_SCORE_KEY,
    cross_encoder_rerank,
    passage_text,
)
from app.graph.nodes.llm_rerank import _candidate_chunks
from app.graph.nodes.post_retrieval_rerank import rerank_chunks
from app.integrations.cross_encoder_client import (
    CrossEncoderScores,
    CrossEncoderUnavailableError,
)
from app.schemas.retrieval import RetrievedChunk


@pytest.fixture(autouse=True)
def _no_threshold(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "CHAT_RERANK_SCORE_THRESHOLD", 0.0)


class FakeClient:
    """Scores a passage by a table keyed on its content; unknown passages score 0."""

    def __init__(self, table: dict[str, float] | None = None, *, fail: str | None = None):
        self.table = table or {}
        self.fail = fail
        self.calls: list[tuple[str, list[str]]] = []

    async def rerank(self, query, passages, *, top_k=None) -> CrossEncoderScores:
        self.calls.append((query, list(passages)))
        if self.fail:
            raise CrossEncoderUnavailableError(self.fail)
        scores = [self.table.get(p.split("\n")[-1], 0.0) for p in passages]
        ranking = sorted(range(len(scores)), key=lambda i: (-scores[i], i))
        return CrossEncoderScores(
            scores=scores, ranking=ranking, truncated=0, took_ms=1, model="fake"
        )


def _chunk(chunk_id: str, score: float, heading: str = "") -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        content=chunk_id,
        source=f"{chunk_id}.pdf",
        score=score,
        heading_path=[heading] if heading else [],
        metadata={"lexical_only": True} if score == 0.70 else {},
    )


_A, _B, _C, _D = _chunk("a", 0.9, "Học phí"), _chunk("b", 0.8), _chunk("c", 0.70), _chunk("d", 0.6)


@pytest.mark.asyncio
async def test_keeps_top_k_by_cross_encoder_score_and_leaves_the_score_alone() -> None:
    client = FakeClient({"c": 5.0, "d": 3.0, "a": -1.0, "b": -4.0})

    outcome = await cross_encoder_rerank(
        client, ["học phí CNTT"], rerank_chunks([[_A, _B, _C, _D]]), keep=2
    )

    assert outcome.failure is None
    kept = outcome.result.per_query[0].chunks
    assert [c.chunk_id for c in kept] == ["c", "d"]
    assert [c.metadata[CE_SCORE_KEY] for c in kept] == [5.0, 3.0]
    # The retrieval score and the existing metadata survive the copy.
    assert kept[0].score == 0.70 and kept[0].metadata["lexical_only"] is True
    assert [c.chunk_id for c in outcome.result.chunks] == ["c", "d"]
    assert outcome.result.best_scores == [0.9]


@pytest.mark.asyncio
async def test_ties_keep_the_retrieval_order_and_a_sub_query_never_empties() -> None:
    client = FakeClient({})  # every passage scores 0

    outcome = await cross_encoder_rerank(client, ["q"], rerank_chunks([[_A, _B, _C]]), keep=1)

    assert [c.chunk_id for c in outcome.result.per_query[0].chunks] == ["a"]
    assert outcome.result.failed_query_indexes == []


@pytest.mark.asyncio
async def test_each_sub_query_is_scored_with_its_own_question_concurrently() -> None:
    client = FakeClient({"a": 1.0, "b": 2.0})

    outcome = await cross_encoder_rerank(
        client, ["q1", "q2"], rerank_chunks([[_A, _B], [_B, _A]]), keep=1
    )

    assert sorted(q for q, _ in client.calls) == ["q1", "q2"]
    assert client.calls[0][1] == ["Học phí\na", "b"]  # heading path precedes the content
    assert [c.chunk_id for c in outcome.result.per_query[0].chunks] == ["b"]
    assert [c.chunk_id for c in outcome.result.per_query[1].chunks] == ["b"]
    assert [c.chunk_id for c in outcome.result.chunks] == ["b"]


@pytest.mark.asyncio
async def test_a_service_failure_returns_the_input_untouched_with_the_reason() -> None:
    client = FakeClient(fail="Không kết nối được tới service cross-encoder tại http://x.")
    before = rerank_chunks([[_A, _B, _C, _D]])

    outcome = await cross_encoder_rerank(client, ["q"], before, keep=2)

    assert outcome.result is before
    assert outcome.failure is not None and "Không kết nối được" in outcome.failure
    assert outcome.elapsed_ms >= 0


@pytest.mark.asyncio
async def test_two_stage_mode_hands_the_llm_rerank_exactly_k_candidates_per_sub_query() -> None:
    client = FakeClient({"d": 9.0, "a": 8.0})

    outcome = await cross_encoder_rerank(client, ["q"], rerank_chunks([[_A, _B, _C, _D]]), keep=2)

    assert [c.chunk_id for c in _candidate_chunks(outcome.result)] == ["d", "a"]


def test_passage_text_puts_the_heading_path_before_the_content() -> None:
    chunk = RetrievedChunk(
        chunk_id="x",
        content="Khối Công nghệ 38.350.000",
        source="s",
        score=0.5,
        heading_path=["Học phí 2025", "3 Đại học chính quy"],
    )

    assert passage_text(chunk) == "Học phí 2025 > 3 Đại học chính quy\nKhối Công nghệ 38.350.000"
    assert passage_text(_B) == "b"
