import pytest

from app.core.config import settings
from app.graph.nodes.post_retrieval_rerank import rerank_chunks
from app.graph.nodes.retrieval_filtering import retrieve_chunks
from app.schemas.retrieval import RetrievedChunk


def test_retrieve_chunks_respects_max_chunks_setting(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "RETRIEVAL_MAX_CHUNKS", 1)

    chunks = retrieve_chunks("quy chế đào tạo", user_faculty="GLOBAL", user_level=1)

    assert len(chunks) <= 1


def test_rerank_chunks_filters_below_threshold(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "RERANK_SCORE_THRESHOLD", 0.9)
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
    monkeypatch.setattr(settings, "RERANK_SCORE_THRESHOLD", 0.9)
    chunks = [RetrievedChunk(chunk_id="a", content="x", source="s", score=0.1)]

    result = rerank_chunks(chunks)

    assert result.has_valid_context is False
    assert result.chunks == []
