"""Major (ngành) scope: a question about one major answered from another major's document
(25/11/2026: KTPM thesis weights given from the QTDL thesis plan)."""

import pytest

from app.graph.major_scope import asked_majors, check_major_scope, document_major
from app.schemas.retrieval import RetrievedChunk


def _chunk(chunk_id: str, title: str) -> RetrievedChunk:
    return RetrievedChunk(chunk_id=chunk_id, content="x", source=f"{title}.pdf", score=0.7)


QTDL = _chunk(
    "qtdl",
    "Kế hoạch khóa luận tốt nghiệp khóa 18 (2022-2026) ngành Quản trị du lịch, khách sạn, "
    "nhà hàng HK2 2025-2026",
)
CNPM_RULES = _chunk(
    "a2", "Quy chế nội bộ chấm khóa luận tốt nghiệp và tổ chức hội đồng phản biện BM CNPM"
)
CO_KHI = _chunk(
    "ck", "Chương trình đào tạo ngành Công nghệ kỹ thuật cơ khí khóa 21 năm học 2025-2026"
)
KE_TOAN = _chunk("kt", "Chương trình đào tạo ngành Kế toán khóa K22 (chính quy)")


@pytest.mark.parametrize(
    ("title", "expected"),
    [
        (QTDL.source, ("quan tri du lich", "Quản trị du lịch")),
        (CO_KHI.source, ("cong nghe ky thuat co khi", "Công nghệ kỹ thuật cơ khí")),
        (KE_TOAN.source, ("ke toan", "Kế toán")),
        (
            "Chương trình đào tạo ngành Tài chính \u2013 Ngân hàng khóa 20.pdf",
            ("tai chinh ngan hang", "Tài chính \u2013 Ngân hàng"),
        ),
        (CNPM_RULES.source, None),
        ("Quyet Dinh 1035 - Hoc phi 2025-2026.pdf", None),
    ],
)
def test_document_major(title: str, expected: tuple[str, str] | None) -> None:
    assert document_major(title) == expected


def test_asked_majors_reads_abbreviations_full_names_and_document_majors() -> None:
    assert asked_majors("điểm khóa luận ktpm tính thế nào", []) == {"ky thuat phan mem"}
    assert asked_majors("học phí ngành tài chính ngân hàng", []) == {"tai chinh ngan hang"}
    assert asked_majors("CTĐT ngành công nghệ kỹ thuật cơ khí", ["cong nghe ky thuat co khi"]) == {
        "cong nghe ky thuat co khi"
    }
    assert asked_majors("seminar bộ môn tổ chức lúc nào", ["ke toan"]) == set()


def test_another_majors_document_is_kept_but_generation_is_told() -> None:
    scope = check_major_scope(
        ["Điểm khóa luận KTPM tính trọng số GVHD, phản biện, hội đồng thế nào?"],
        [QTDL, CNPM_RULES],
    )
    assert scope.asked == "Kỹ thuật phần mềm"
    assert scope.dropped == 0 and len(scope.chunks) == 2
    assert "chưa có văn bản cho ngành Kỹ thuật phần mềm" in scope.notice
    assert "ngành Quản trị du lịch" in scope.notice
    assert "KHÔNG gọi đó là thông tin của ngành Kỹ thuật phần mềm" in scope.notice


def test_other_majors_documents_are_dropped_when_the_asked_one_has_some() -> None:
    scope = check_major_scope(
        ["CTĐT ngành kế toán khóa 22 có bao nhiêu tín chỉ"], [CO_KHI, KE_TOAN, CNPM_RULES]
    )
    assert scope.dropped == 1
    assert [chunk.chunk_id for chunk in scope.chunks] == ["kt", "a2"]
    assert scope.notice == ""


def test_no_major_asked_but_documents_name_one() -> None:
    scope = check_major_scope(["điểm khóa luận tính trọng số thế nào"], [QTDL])
    assert "Câu hỏi không nêu ngành" in scope.notice
    assert "ngành Quản trị du lịch" in scope.notice


def test_neutral_documents_or_a_comparison_are_left_alone() -> None:
    assert check_major_scope(["seminar tổ chức lúc nào"], [CNPM_RULES]).notice == ""
    comparison = check_major_scope(
        ["so sánh CTĐT ngành kế toán và ngành cơ khí"], [CO_KHI, KE_TOAN]
    )
    assert comparison.dropped == 0 and comparison.notice == ""


def test_the_users_own_words_decide_the_major() -> None:
    """A standalone rewrite that adds another major must not turn the question into a
    comparison (25/11/2026)."""

    scope = check_major_scope(
        ["Điểm khóa luận ngành Kỹ thuật phần mềm (Hệ thống thông tin) tính thế nào?"],
        [QTDL],
        "diem khoa luan ktpm tinh trong so gvhd, phan bien, hoi dong the nao",
    )
    assert scope.asked == "Kỹ thuật phần mềm"
    assert "chưa có văn bản cho ngành Kỹ thuật phần mềm" in scope.notice
