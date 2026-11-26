import pytest

from app.graph.cohort_scope import asked_cohort, check_cohort_scope, document_cohort
from app.schemas.retrieval import RetrievedChunk


def _chunk(chunk_id: str, title: str) -> RetrievedChunk:
    return RetrievedChunk(chunk_id=chunk_id, content="| row |", source=f"{title}.pdf", score=0.7)


K16 = _chunk(
    "k16", "Chương trình đào tạo ngành Công nghệ kỹ thuật cơ khí khóa 16 năm học 2020-2021"
)
K16B = _chunk(
    "k16b", "Chương trình đào tạo ngành Công nghệ kỹ thuật cơ khí khóa 16 năm học 2020-2021"
)
K19 = _chunk(
    "k19", "Chương trình đào tạo ngành Công nghệ kỹ thuật cơ khí khóa 19 năm học 2023-2024"
)
FEES = _chunk("fees", "Quyết định mức thu học phí các hệ đào tạo năm học 2025-2026")


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("Tiếng Anh 2 ngành cơ khí khóa K18 (nhập học 2022) học trước môn gì?", 2022),
        ("Tiếng Anh 2 khóa 18 học trước môn gì?", 2022),
        ("Học phí K19 1 tín chỉ?", 2023),
        ("Học phí khóa tuyển sinh năm học 2024-2025?", 2024),
        ("Học phí năm học 2025-2026 là bao nhiêu?", None),  # a school year, not a cohort
        ("So sánh K18 và K19", None),  # two cohorts: not one cohort's question
        ("Thủ tục bảo lưu?", None),
    ],
)
def test_asked_cohort(question: str, expected: int | None) -> None:
    assert asked_cohort(question) == expected


def test_document_cohort_reads_the_title_only() -> None:
    assert document_cohort(K16.source) == 2020
    assert document_cohort("uploads/1b2c3d4e-0000-4000-8000-000000000000_CTĐT khóa 19.pdf") == 2023
    assert document_cohort(FEES.source) is None


def test_other_cohorts_are_dropped_when_the_asked_one_is_there() -> None:
    scope = check_cohort_scope(["Tiếng Anh 2 khóa 19 học trước môn gì?"], [K16, K19, FEES])

    assert [c.chunk_id for c in scope.chunks] == ["k19", "fees"]
    assert scope.notice == ""
    assert scope.dropped == 1


def test_a_missing_cohort_keeps_the_documents_grouped_and_says_so() -> None:
    scope = check_cohort_scope(
        [
            "Tiếng Anh 2 khóa K18 (nhập học 2022) học trước môn gì?",
            "Tiếng Anh 2 khóa K18 mấy tín chỉ?",
        ],
        [K16, K19, K16B],
    )

    assert [c.chunk_id for c in scope.chunks] == ["k16", "k16b", "k19"]
    assert "KHÔNG có tài liệu của khóa này" in scope.notice
    assert "khóa 18 (nhập học 2022)" in scope.notice
    assert "khóa 16 (nhập học 2020), khóa 19 (nhập học 2023)" in scope.notice


def test_an_unnamed_cohort_asks_to_name_it_per_figure() -> None:
    scope = check_cohort_scope(["Tiếng Anh 2 ngành cơ khí học trước môn gì?"], [K16])

    assert scope.chunks == [K16]
    assert "Câu hỏi không nêu khóa" in scope.notice


@pytest.mark.parametrize(
    ("questions", "chunks"),
    [
        (["Học phí khóa 2024?"], [FEES]),  # no document names a cohort
        (["Tiếng Anh 2 khóa 16?", "Tiếng Anh 2 khóa 19?"], [K16, K19]),  # a comparison
        ([], [K16]),
    ],
)
def test_nothing_happens_without_a_cohort_document_or_one_asked_cohort(
    questions: list[str], chunks: list[RetrievedChunk]
) -> None:
    scope = check_cohort_scope(questions, chunks)

    assert scope.chunks == chunks
    assert scope.notice == ""


@pytest.mark.parametrize(
    ("title", "expected"),
    [
        # 25/11/2026: "K15" was not read, so a K18 question got K15's plan as K18's.
        (
            "Thông báo kế hoạch thực hiện khóa luận tốt nghiệp K15 ngành Công nghệ thực phẩm "
            "năm học 2022-2023",
            2019,
        ),
        ("Chương trình đào tạo ngành Kế toán khóa K22 (chính quy)", 2026),
        ("Lịch thi HK2 năm học 2025-2026", None),  # "HK2" is a term, not a cohort
    ],
)
def test_document_cohort_reads_k_numbers(title: str, expected: int | None) -> None:
    assert document_cohort(f"{title}.pdf") == expected


def test_a_cohort_without_documents_is_told_even_when_titles_write_k15() -> None:
    k15 = _chunk(
        "k15",
        "Thông báo kế hoạch thực hiện khóa luận tốt nghiệp K15 ngành Công nghệ thực phẩm "
        "năm học 2022-2023",
    )
    scope = check_cohort_scope(
        ["Kế hoạch thực hiện khóa luận tốt nghiệp ngành công nghệ thực phẩm K18"], [k15]
    )
    assert scope.asked == 2022 and scope.available == (2019,)
    assert "KHÔNG có tài liệu của khóa này" in scope.notice
