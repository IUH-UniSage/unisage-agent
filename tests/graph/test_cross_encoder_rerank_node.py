"""CrossEncoderRerankNode - top-K per sub-query by cross-encoder score, scores left
alone, fail-over on a service error; a client double, no HTTP."""

import pytest

from app.core.config import settings
from app.graph.nodes.cross_encoder_rerank import (
    CE_SCORE_KEY,
    cross_encoder_rerank,
    passage_text,
    passage_windows,
    person_names,
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
    # The title, then the heading path, precede the content.
    assert client.calls[0][1] == ["Tài liệu: a\nHọc phí\na", "Tài liệu: b\nb"]
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


def test_passage_text_puts_the_title_and_heading_path_before_the_content() -> None:
    chunk = RetrievedChunk(
        chunk_id="x",
        content="Khối Công nghệ 38.350.000",
        source="docs/3fa85f64-5717-4562-b3fc-2c963f66afa6_Quyet Dinh 1035 - Hoc phi.pdf",
        score=0.5,
        heading_path=["Học phí 2025", "3 Đại học chính quy"],
    )

    assert passage_text(chunk) == (
        "Tài liệu: Quyet Dinh 1035 - Hoc phi\n"
        "Học phí 2025 > 3 Đại học chính quy\n"
        "Khối Công nghệ 38.350.000"
    )
    no_title = RetrievedChunk(chunk_id="y", content="b", source="", score=0.5)
    assert passage_text(no_title) == "b"


def _table_chunk(
    chunk_id: str, rows: int, *, source: str = "Kế hoạch đào tạo.pdf"
) -> RetrievedChunk:
    lines = ["KẾ HOẠCH ĐÀO TẠO", "", "| STT | Tên môn học | Số tín chỉ |", "| --- | --- | --- |"]
    lines += [
        f"| {n} | Học phần số {n} có tên khá dài để chiếm chỗ | 3(3,0,6) |"
        for n in range(1, rows + 1)
    ]
    return RetrievedChunk(chunk_id=chunk_id, content="\n".join(lines), source=source, score=0.8)


def test_a_short_chunk_is_one_passage() -> None:
    chunk = _table_chunk("t", 2)
    assert passage_windows(chunk) == [passage_text(chunk)]


def test_a_long_table_chunk_is_scored_as_row_windows_that_repeat_the_header() -> None:
    chunk = _table_chunk("t", 40)
    windows = passage_windows(chunk, budget=400)
    assert len(windows) > 1
    for window in windows:
        assert window.startswith(
            "Tài liệu: Kế hoạch đào tạo\nKẾ HOẠCH ĐÀO TẠO\n| STT | Tên môn học"
        )
    body = [
        line
        for window in windows
        for line in window.split("\n")
        if line.startswith("| ") and "Học phần số" in line
    ]
    assert body == [line for line in chunk.content.split("\n") if "Học phần số" in line]


@pytest.mark.asyncio
async def test_a_chunk_takes_its_best_window_score_and_requests_stay_under_64_passages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "CHAT_CROSS_ENCODER_WINDOW_CHARS", 120)
    long = _table_chunk("long", 30)
    last_row = long.content.split("\n")[-1]
    client = FakeClient({last_row: 5.0})
    chunks = [long, *(_table_chunk(f"x{i}", 30) for i in range(5))]
    outcome = await cross_encoder_rerank(
        client, ["Học phần số 30?"], rerank_chunks([chunks]), keep=2
    )
    kept = outcome.result.per_query[0].chunks
    assert kept[0].chunk_id == "long"
    assert kept[0].metadata[CE_SCORE_KEY] == 5.0
    assert len(client.calls) > 1
    assert all(len(passages) <= 64 for _, passages in client.calls)


def test_person_names_reads_a_full_vietnamese_name_and_not_a_school_name() -> None:
    assert person_names("Portfolio của sinh viên Nguyễn Chí Thanh ngành Công nghệ May") == [
        "nguyen chi thanh"
    ]
    assert person_names("Trường Đại học Công nghiệp TP.HCM ở Hà Nội") == []
    assert person_names("TS. Huỳnh Trung Hiếu Phó trưởng khoa") == [
        "huynh trung hieu",
        "huynh trung hieu pho",
    ]


@pytest.mark.asyncio
async def test_a_document_titled_with_the_person_asked_about_goes_first() -> None:
    mine = RetrievedChunk(
        chunk_id="mine",
        content="mine",
        source="u_Student portfolio - Nguyen Chi Thanh.pdf",
        score=0.7,
    )
    other = RetrievedChunk(
        chunk_id="other",
        content="other",
        source="u_Student portfolio - Truong Thi Yen Nhi.pdf",
        score=0.9,
    )
    client = FakeClient({"other": 3.0, "mine": 1.0})
    outcome = await cross_encoder_rerank(
        client,
        ["Portfolio của sinh viên Nguyễn Chí Thanh gồm những phần nào?"],
        rerank_chunks([[other, mine]]),
        keep=2,
    )
    assert [c.chunk_id for c in outcome.result.per_query[0].chunks] == ["mine", "other"]
    # no name in the question: cross-encoder order
    outcome = await cross_encoder_rerank(
        client, ["Portfolio gồm những phần nào?"], rerank_chunks([[other, mine]]), keep=2
    )
    assert [c.chunk_id for c in outcome.result.per_query[0].chunks] == ["other", "mine"]


@pytest.mark.asyncio
async def test_a_name_the_standalone_rewrite_dropped_is_read_from_the_users_message() -> None:
    mine = RetrievedChunk(
        chunk_id="mine",
        content="mine",
        source="u_Student portfolio - Nguyen Chi Thanh.pdf",
        score=0.7,
    )
    other = RetrievedChunk(
        chunk_id="other",
        content="other",
        source="u_Student portfolio - Truong Thi Yen Nhi.pdf",
        score=0.9,
    )
    client = FakeClient({"other": 3.0, "mine": 1.0})
    outcome = await cross_encoder_rerank(
        client,
        ["Portfolio của sinh viên ngành Công nghệ May gồm những phần nào?"],
        rerank_chunks([[other, mine]]),
        keep=2,
        asked="Portfolio của sinh viên Nguyễn Chí Thanh ngành Công nghệ May gồm những phần nào?",
    )
    assert [c.chunk_id for c in outcome.result.per_query[0].chunks] == ["mine", "other"]


@pytest.mark.asyncio
async def test_expanded_chunks_take_at_most_the_capped_number_of_kept_slots(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "CHAT_RETRIEVAL_EXPAND_KEEP", 1)
    found = RetrievedChunk(chunk_id="found", content="found", source="a.pdf", score=0.8)
    added = [
        RetrievedChunk(
            chunk_id=f"e{i}",
            content=f"e{i}",
            source="b.pdf",
            score=0.6,
            metadata={"expanded_from_document": True},
        )
        for i in range(3)
    ]
    client = FakeClient({"e0": 9.0, "e1": 8.0, "e2": 7.0, "found": 1.0})
    outcome = await cross_encoder_rerank(client, ["q"], rerank_chunks([[found, *added]]), keep=2)
    assert [c.chunk_id for c in outcome.result.per_query[0].chunks] == ["e0", "found"]
