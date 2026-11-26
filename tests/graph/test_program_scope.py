import json

import pytest

from app.graph.program_scope import asked_program, check_program_scope, document_program
from app.schemas.retrieval import RetrievedChunk


def _chunk(chunk_id: str, title: str) -> RetrievedChunk:
    return RetrievedChunk(chunk_id=chunk_id, content="| row |", source=f"{title}.pdf", score=0.7)


ENHANCED = _chunk(
    "ta",
    "Chương trình đào tạo tăng cường tiếng Anh ngành Công nghệ kỹ thuật cơ khí khóa 21 "
    "năm học 2025-2026",
)
STANDARD = _chunk(
    "std", "Chương trình đào tạo ngành Công nghệ kỹ thuật cơ khí khóa 21 năm học 2025-2026"
)
FEES = _chunk("fees", "Quyết định mức thu học phí các hệ đào tạo năm học 2025-2026")
QUESTION = ["Tiếng Anh 2 ngành cơ khí khóa K21 (nhập học 2025) học trước môn gì?"]


def test_document_program_reads_training_programme_titles_only() -> None:
    assert document_program(ENHANCED.source) == "tang_cuong_ta"
    assert document_program(STANDARD.source) == "chuan"
    assert document_program("CTĐT chất lượng cao ngành CNTT khóa 20.pdf") == "clc"
    assert document_program(FEES.source) is None


@pytest.mark.parametrize(
    ("questions", "profile", "expected"),
    [
        (["... hệ chính quy khóa K21?"], None, (None, False)),  # chính quy is both
        (["... chương trình tăng cường tiếng Anh khóa 21?"], None, ("tang_cuong_ta", False)),
        (["... khóa 21?"], "Đại học chính quy (chương trình chuẩn)", ("chuan", False)),
        (["... khóa 21?"], "chinh_quy_tang_cuong_tieng_anh", ("tang_cuong_ta", False)),
        (["... chuẩn đầu ra ngành cơ khí?"], None, (None, False)),
        (["... CLC và tăng cường tiếng Anh khác gì?"], None, (None, True)),
        (["... chất lượng cao?"], "Đại học chính quy (chương trình chuẩn)", ("clc", False)),
    ],
)
def test_asked_program(
    questions: list[str], profile: str | None, expected: tuple[str | None, bool]
) -> None:
    assert asked_program(questions, profile) == expected


def test_only_an_enhanced_programme_found_says_so_and_asks() -> None:
    scope = check_program_scope(QUESTION, None, [ENHANCED, FEES])

    assert scope.chunks == [FEES, ENHANCED]
    assert "thông tin dưới đây thuộc chương trình tăng cường tiếng Anh" in scope.notice
    assert "chương trình chuẩn có thể khác" in scope.notice
    form = json.loads(scope.notice.rsplit("options này: ", 1)[1])
    assert form["field"] == "he_dao_tao"
    assert [option["id"] for option in form["options"]] == [
        "chinh_quy_chuan",
        "chinh_quy_tang_cuong_tieng_anh",
    ]


def test_both_programmes_found_put_the_standard_one_first() -> None:
    scope = check_program_scope(QUESTION, None, [ENHANCED, STANDARD])

    assert scope.chunks == [STANDARD, ENHANCED]
    assert "Trình bày chương trình chuẩn TRƯỚC" in scope.notice
    assert "ask_user_form" in scope.notice


def test_only_the_standard_programme_found_is_labelled_without_a_form() -> None:
    scope = check_program_scope(QUESTION, None, [STANDARD])

    assert "theo chương trình chuẩn" in scope.notice
    assert "ask_user_form" not in scope.notice


def test_the_asked_programme_drops_the_others() -> None:
    scope = check_program_scope(QUESTION, "chinh_quy_chuan", [ENHANCED, STANDARD, FEES])

    assert scope.chunks == [STANDARD, FEES]
    assert scope.notice == ""
    assert scope.dropped == 1


def test_an_asked_programme_without_a_document_is_named_as_missing() -> None:
    scope = check_program_scope(QUESTION, "chinh_quy_chuan", [ENHANCED])

    assert scope.chunks == [ENHANCED]
    assert "không phải chương trình chuẩn" in scope.notice


def test_nothing_happens_without_a_training_programme_document() -> None:
    assert check_program_scope(QUESTION, None, [FEES]).notice == ""
