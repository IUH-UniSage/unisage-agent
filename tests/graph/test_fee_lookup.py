import asyncio
from collections.abc import Sequence

from app.graph.fee_lookup import (
    FEE_ROW_FLAG,
    fee_rows,
    fee_summary,
    groups_of_text,
    is_fee_question,
    lookup_fee_rows,
    without_programmes,
)
from app.schemas.retrieval import RetrievedChunk

DOC = "fee-doc"
SOURCE = "uuid_Quyet Dinh 1035 - Hoc phi 2025-2026.pdf"
HEAD = (
    "| TT | HỆ ĐÀO TẠO / KHÓA TUYỂN SINH | Mức thu trong năm học 2025 - 2026 > Mức thu 01 tín chỉ "
    "| Mức thu trong năm học 2025 - 2026 > Mức thu theo năm học |\n| --- | --- | --- | --- |\n"
)
CHUNKS = {
    "p2": HEAD
    + "[A ĐỐI VỚI TRỤ SỞ CHÍNH > 3 Đại học chính quy > 3.1 Khóa tuyển sinh năm học 2025-2026]\n"
    "| 3.1 | - Khối Kinh tế (các ngành khác): |  | 36.260.000 |\n"
    "| 3.1 | - Khối Công nghệ: |  | 38.350.000 |\n"
    "[A ĐỐI VỚI TRỤ SỞ CHÍNH > 3 Đại học chính quy > 3.1 Khóa tuyển sinh năm học 2025-2026 > "
    "- Khối Công nghệ:]\n"
    "| 3.1 | + Môn lý thuyết: 980.000 \\| Môn thực hành: 1.600.000 | 980k / 1.600k |  |\n",
    "p3": HEAD + "[A ĐỐI VỚI TRỤ SỞ CHÍNH > 3 Đại học chính quy > 3.3 Khóa tuyển sinh năm học "
    "2023-2024]\n"
    "| 3.3 | - Khối Kinh tế [LT: 840.000, TH: 1.010.000] | 840.000 / 1.010.000 | 32.264.000 |\n"
    "| 3.3 | - Khối Công nghệ [LT: 840.000, TH: 1.560.000] | 840.000 / 1.560.000 | 33.428.000 |\n"
    "[A ĐỐI VỚI TRỤ SỞ CHÍNH > 3 Đại học chính quy > 3.4 Khóa tuyển sinh năm học 2022-2023]\n"
    "| 3.4 | - Khối Công nghệ [LT: 750.000, TH: 1.130.000] | 750.000 / 1.130.000 | 27.272.000 |\n",
    "p4": HEAD + "[A ĐỐI VỚI TRỤ SỞ CHÍNH > 3 Đại học chính quy > 3.5 Khóa tuyển sinh năm học "
    "2020-2021 trở về trước]\n"
    "| 3.5 | - Khối Công nghệ [LT: 700.000, TH: 1.000.000] | 700.000 / 1.000.000 | 25.000.000 |\n"
    "[A ĐỐI VỚI TRỤ SỞ CHÍNH > 2 Cao học]\n"
    "| 2.3 | Khóa tuyển sinh năm học 2023-2024 trở về trước - Khối Công nghệ | 1.430.000 "
    "| 42.820.000 |\n",
}


def _chunk(key: str) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=f"{DOC}:{key}",
        content=CHUNKS[key],
        source=SOURCE,
        score=0.5,
        metadata={"document_id": DOC},
    )


async def _fetch(ids: Sequence[str]) -> list[RetrievedChunk]:
    return [_chunk(key) for key in CHUNKS] if DOC in ids else []


async def _titles() -> dict[str, str]:
    return {DOC: SOURCE, "other": "uuid_Chương trình đào tạo ngành CNTT khóa 18.pdf"}


def _lookup(*texts: str, present: Sequence[RetrievedChunk] = ()):
    return asyncio.run(
        lookup_fee_rows(
            [], list(present), texts=list(texts), fetch_documents=_fetch, titles=_titles
        )
    )


def test_the_asked_intake_takes_its_own_row_not_the_merged_one() -> None:
    lookup = _lookup("học phí ngành cntt khóa K18")

    assert lookup.intake == 2022
    assert [row.values[-1][1] for row in lookup.rows] == ["27.272.000"]
    assert [chunk.chunk_id for chunk in lookup.added] == [f"{DOC}:p3"]
    assert lookup.added[0].metadata[FEE_ROW_FLAG] is True


def test_an_old_intake_falls_in_the_merged_row() -> None:
    lookup = _lookup("học phí khối công nghệ khóa 2019")

    assert [row.intake for row in lookup.rows] == [(2020, True)]
    assert "dòng gộp" in fee_summary(lookup.added, lookup)


def test_an_intake_between_its_rows_and_the_merged_one_is_reported_missing() -> None:
    lookup = _lookup("học phí ngành cntt khóa 2021")

    assert lookup.rows == [] and lookup.missing_intake
    assert "không có dòng của khóa tuyển sinh năm học 2021-2022" in fee_summary([], lookup)


def test_no_intake_lists_every_intake_of_the_group() -> None:
    lookup = _lookup("hoc phi nganh cntt")

    values = {row.values[-1][1] for row in lookup.rows}
    assert {"38.350.000", "33.428.000", "27.272.000", "25.000.000"} <= values
    assert "32.264.000" not in values  # Khối Kinh tế


def test_the_system_is_kept_apart() -> None:
    lookup = _lookup("học phí cao học khối công nghệ khóa 2023")

    assert lookup.system == "cao hoc"
    assert [row.values[-1][1] for row in lookup.rows] == ["42.820.000"]


def test_group_of_a_major_by_convention() -> None:
    assert groups_of_text("học phí ngành cntt") == {"khoi cong nghe"}
    assert groups_of_text("học phí ngành kế toán") == {"khoi kinh te"}
    assert groups_of_text("học phí ngành ngôn ngữ anh") == {"khoi kinh te dac biet"}
    assert groups_of_text("học phí ngành dược") == {"nganh duoc"}
    # "được" is not Ngành Dược once accents are gone.
    assert groups_of_text("học phí có được giảm không") == set()
    assert groups_of_text("học phí khối kinh tế") == {"khoi kinh te"}


def test_only_amount_questions_are_looked_up() -> None:
    assert is_fee_question(["học phí 1 tín chỉ lý thuyết bao nhiêu"])
    assert not is_fee_question(["hạn đóng học phí kỳ này là khi nào"])
    assert _lookup("điều kiện học bổng là gì").documents == 0


def test_an_escaped_pipe_stays_inside_its_cell() -> None:
    rows = list(fee_rows(_chunk("p2")))
    split = next(row for row in rows if "Môn lý thuyết" in row.path)

    assert "Môn thực hành: 1.600.000" in split.path
    assert split.values == [("Mức thu 01 tín chỉ", "980k / 1.600k")]


def test_the_summary_numbers_the_rows_of_the_final_context() -> None:
    lookup = _lookup("học phí ngành cntt khóa K18")
    context = [_chunk("p2"), _chunk("p3")]

    summary = fee_summary(context, lookup)

    assert "## DÒNG BIỂU HỌC PHÍ ĐÃ ĐỌC TỪ BẢNG" in summary
    assert "- [2] " in summary and "3.4 Khóa tuyển sinh năm học 2022-2023" in summary
    assert "Mức thu theo năm học: 27.272.000" in summary
    assert 'dòng "Khóa tuyển sinh năm học 2022-2023"' in summary


def test_a_row_already_in_context_is_not_added_again() -> None:
    lookup = _lookup("học phí ngành cntt khóa K18", present=[_chunk("p3")])

    assert lookup.added == [] and len(lookup.rows) == 1


def test_programme_documents_leave_the_context_once_the_rows_are_found() -> None:
    programme = RetrievedChunk(
        chunk_id="ctdt:1",
        content="| 1 | Tiếng Anh 1 |",
        source="0b1c2d3e-4f50-6172-8394-a5b6c7d8e9f0_Chương trình đào tạo ngành cơ khí khóa 18.pdf",
        score=0.9,
    )
    found = _lookup("học phí ngành cơ khí khóa K18")
    nothing = _lookup("điều kiện học bổng là gì")

    assert without_programmes([programme, _chunk("p2")], found) == [_chunk("p2")]
    assert without_programmes([programme], nothing) == [programme]
