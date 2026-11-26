from collections.abc import Sequence

import pytest

from app.graph.course_lookup import (
    COURSE_ROW_FLAG,
    CourseLookup,
    asked_courses,
    course_rows,
    course_summary,
    lookup_course_rows,
    programme_documents,
    sibling_documents,
)
from app.schemas.retrieval import RetrievedChunk

_STANDARD = "Chương trình đào tạo ngành Công nghệ kỹ thuật cơ khí khóa 21 năm học 2025-2026.pdf"
_ENHANCED = (
    "Chương trình đào tạo tăng cường tiếng Anh ngành Công nghệ kỹ thuật cơ khí khóa 21 "
    "năm học 2025-2026.pdf"
)
_HEADER = (
    "CHƯƠNG TRÌNH ĐÀO TẠO 2025 - 2026\n\n"
    "| STT | Mã môn học | Tên môn học | Tên tiếng Anh | Mã học phần | Số tín chỉ | Mã tự quản | "
    "Học phần: học trước (a), tiên quyết (b) song hành (c) |\n"
    "| --- | --- | --- | --- | --- | --- | --- | --- |\n"
)


def _chunk(chunk_id: str, source: str, rows: str) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        content=_HEADER + rows,
        source=source,
        score=0.7,
        metadata={"document_id": chunk_id.split(":")[0], "region_type": "table"},
    )


_TA1 = _chunk(
    "doc-std:3",
    _STANDARD,
    "[Học kỳ 2 > Học phần bắt buộc]\n"
    "| 7 | 015253 | Tiếng Anh 1 | English 1 | 4203015253 | 3(3,0,6) | 2111108 |  |\n",
)
_TA2 = _chunk(
    "doc-std:6",
    _STANDARD,
    "[Học kỳ 3 > Học phần bắt buộc]\n"
    "| 5 | 014167 | Chủ nghĩa xã hội khoa học | Scientific Socialism | 4203014167 | 2(2,0,4) "
    "| 2112014 | 014164(a) |\n"
    "| 6 | 015254 | Tiếng Anh 2 | English 2 | 4203015254 | 3(3,0,6) | 2111188 | 015253(b) |\n",
)
_OTHER = _chunk(
    "doc-std:9",
    _STANDARD,
    "| 1 | 002180 | Vật lý kỹ thuật | Engineering Physics | 4203002180 | 3(2,2,6) | 2113500 |  |\n",
)
_AV2 = _chunk(
    "doc-tc:6",
    _ENHANCED,
    "| 6 | 015256 | Anh văn 2 | English 2 | 4297015256 | 4(4,0,8) | 2511012 | 015255(b) |\n",
)
_FEE = RetrievedChunk(
    chunk_id="fee:1", content="Khối Công nghệ 980.000", source="Học phí.pdf", score=0.6
)


def _fetcher(chunks: Sequence[RetrievedChunk]):  # type: ignore[no-untyped-def]
    calls: list[list[str]] = []

    async def fetch(ids: list[str]) -> list[RetrievedChunk]:
        calls.append(ids)
        return [chunk for chunk in chunks if chunk.metadata["document_id"] in ids]

    return fetch, calls


def test_course_rows_read_code_name_and_prerequisites() -> None:
    rows = list(course_rows(_TA2.content))

    assert [(row.code, row.name, row.prerequisites) for row in rows] == [
        ("014167", "Chủ nghĩa xã hội khoa học", ("014164",)),
        ("015254", "Tiếng Anh 2", ("015253",)),
    ]
    assert list(course_rows(_HEADER)) == []  # the header names no course


def test_asked_courses_prefers_the_classifier_course_then_the_written_name() -> None:
    names = ["Tiếng Anh 1", "Tiếng Anh 2", "Anh văn 2", "Toán"]

    assert asked_courses(names, "Anh văn 2", ["AV2 mấy tín"]) == ["anh van 2"]
    # No accents typed; "Tiếng Anh" alone would be inside "Tiếng Anh 2".
    assert asked_courses(names, None, ["ctdt co khi khoa 22 mon tieng anh 2 may tin"]) == [
        "tieng anh 2"
    ]
    # One-word names are too common to match: "toán" here is not the course.
    assert asked_courses(names, None, ["tính toán điểm trung bình"]) == []
    # A course the classifier named that no table has falls back to the written words.
    assert asked_courses(names, "Tiếng Anh 3", ["tieng anh 1 hoc ky may"]) == ["tieng anh 1"]


def test_programme_documents_keep_programme_titles_in_candidate_order() -> None:
    assert programme_documents([_FEE, _AV2, _TA1, _TA2]) == ["doc-tc", "doc-std"]


@pytest.mark.asyncio
async def test_lookup_adds_the_asked_row_missing_from_the_rerank() -> None:
    """The standard programme's Tiếng Anh 2 row was a candidate but not kept (25/11/2026)."""

    fetch, calls = _fetcher([_TA1, _TA2, _OTHER, _AV2])

    lookup = await lookup_course_rows(
        [_AV2, _TA1, _FEE],
        [_AV2, _TA1, _FEE],
        course="Tiếng Anh 2",
        texts=["chương trình đào tạo ngành cơ khí khóa 22: tiếng anh 2 học trước môn nào"],
        fetch_documents=fetch,
    )

    assert calls == [["doc-tc", "doc-std"]]
    # The table's own spelling, for `hoc_phan` when the classifier left it out.
    assert lookup.course_name == "Tiếng Anh 2"
    # Tiếng Anh 1 (its prerequisite 015253) is already kept: only the asked row is added.
    assert [chunk.chunk_id for chunk in lookup.added] == ["doc-std:6"]
    assert lookup.added[0].metadata[COURSE_ROW_FLAG] is True
    assert lookup.courses == ["tieng anh 2"]


@pytest.mark.asyncio
async def test_lookup_adds_the_prerequisite_row_after_the_asked_rows() -> None:
    fetch, _ = _fetcher([_TA1, _TA2, _OTHER])

    lookup = await lookup_course_rows(
        [_OTHER], [_OTHER], course=None, texts=["Tiếng Anh 2 mấy tín chỉ"], fetch_documents=fetch
    )

    assert [chunk.chunk_id for chunk in lookup.added] == ["doc-std:6", "doc-std:3"]


@pytest.mark.asyncio
async def test_lookup_reads_a_course_code_the_user_wrote() -> None:
    fetch, _ = _fetcher([_TA1, _TA2, _OTHER])

    lookup = await lookup_course_rows(
        [_OTHER], [_OTHER], course=None, texts=["môn 4203002180 mấy tín"], fetch_documents=fetch
    )

    assert [chunk.chunk_id for chunk in lookup.added] == []  # its row is already kept
    lookup = await lookup_course_rows(
        [_OTHER], [], course=None, texts=["môn 4203002180 mấy tín"], fetch_documents=fetch
    )
    assert [chunk.chunk_id for chunk in lookup.added] == ["doc-std:9"]


@pytest.mark.asyncio
async def test_lookup_changes_nothing_without_a_programme_document_or_a_course() -> None:
    fetch, calls = _fetcher([_TA1, _TA2])

    assert (
        await lookup_course_rows(
            [_FEE], [_FEE], course="Tiếng Anh 2", texts=["tiếng anh 2"], fetch_documents=fetch
        )
    ).added == []
    assert calls == []
    lookup = await lookup_course_rows(
        [_TA1], [_TA1], course=None, texts=["học phí ngành cơ khí"], fetch_documents=fetch
    )
    assert lookup.added == []


_TITLES = {
    "doc-std": _STANDARD,
    "doc-tc": _ENHANCED,
    "doc-std20": "Chương trình đào tạo ngành Công nghệ kỹ thuật cơ khí khóa 20 năm học 2024.pdf",
    "doc-ctm": "Chương trình đào tạo ngành Công nghệ chế tạo máy khóa 21 năm học 2025-2026.pdf",
    "fee": "Học phí 2025-2026.pdf",
}


def test_siblings_are_the_asked_majors_programmes_of_the_nearest_cohort_standard_first() -> None:
    # Khóa 22 has no programme: khóa 21 is the nearest earlier one; chế tạo máy is another major.
    assert sibling_documents([_AV2], _TITLES, ["cơ khí khóa 22 tiếng anh 2"]) == [
        "doc-std",
        "doc-tc",
    ]
    assert sibling_documents([_AV2], _TITLES, ["cơ khí khóa 20 tiếng anh 2"]) == ["doc-std20"]
    # A major none of the candidates is for: looked up by name, never replaced by the others.
    assert sibling_documents([_AV2], _TITLES, ["ngành kỹ thuật phần mềm khóa 22 tiếng anh 2"]) == []
    titles = _TITLES | {"doc-it": "Kế hoạch đào tạo ngành Công nghệ thông tin khóa 16.pdf"}
    assert sibling_documents([_AV2], titles, ["khóa luận tốt nghiệp ngành cntt khóa 21"]) == [
        "doc-it"
    ]
    # No cohort named: every cohort would qualify, so none is added.
    assert sibling_documents([_AV2], _TITLES, ["cơ khí tiếng anh 2"]) == []


@pytest.mark.asyncio
async def test_lookup_reads_the_standard_programme_the_search_did_not_find() -> None:
    """Only the "tăng cường tiếng Anh" programme was a candidate (25/11/2026)."""

    fetch, calls = _fetcher([_TA1, _TA2, _OTHER, _AV2])

    async def titles() -> dict[str, str]:
        return _TITLES

    lookup = await lookup_course_rows(
        [_AV2],
        [_AV2],
        course="Tiếng Anh 2",
        texts=["chương trình đào tạo ngành công nghệ kỹ thuật cơ khí khóa 22: tiếng anh 2"],
        fetch_documents=fetch,
        titles=titles,
    )

    assert calls == [["doc-std", "doc-tc"]]
    assert [chunk.chunk_id for chunk in lookup.added] == ["doc-std:6", "doc-std:3"]


def test_course_summary_writes_the_asked_row_with_its_conditions_and_index() -> None:
    lookup = CourseLookup(
        courses=["tieng anh 2"],
        names={"doc-std": {"015253": "Tiếng Anh 1", "015254": "Tiếng Anh 2"}},
    )

    summary = course_summary([_AV2, _TA2], lookup)

    assert "## DÒNG HỌC PHẦN ĐÃ ĐỌC TỪ BẢNG" in summary
    assert (
        "- [2] Chương trình đào tạo ngành Công nghệ kỹ thuật cơ khí khóa 21 năm học 2025-2026: "
        "học phần Tiếng Anh 2 (mã môn học 015254, mã học phần 4203015254), số tín chỉ "
        "3(3,0,6); điều kiện của học phần này - tiên quyết (b): 015253 Tiếng Anh 1; không ghi "
        "học trước, song hành."
    ) in summary
    # A row with no condition says so once.
    none = course_summary([_TA1], CourseLookup(courses=["tieng anh 1"]))
    assert "không ghi điều kiện học trước, tiên quyết hay song hành." in none
    # Only the asked course: Chủ nghĩa xã hội khoa học shares the chunk.
    assert "Chủ nghĩa xã hội" not in summary
    assert course_summary([_AV2], lookup) == ""
    assert course_summary([_TA2], CourseLookup()) == ""


def test_course_rows_read_a_row_with_merged_cells() -> None:
    row = (
        "| 2<br>015311 | Khoá luận tốt nghiệp | Capstone Project | 4203015311<br> | 8(0,16,16) "
        "| 2103561 | 015307(b)<br>015308(b) |"
    )

    (parsed,) = course_rows(row)

    assert (parsed.code, parsed.name, parsed.full_code, parsed.credits) == (
        "015311",
        "Khoá luận tốt nghiệp",
        "4203015311",
        "8(0,16,16)",
    )
    assert parsed.conditions == (("015307", "b"), ("015308", "b"))
    # "Khoá" (old spelling) and "Khóa" fold the same.
    assert asked_courses([parsed.name], "Khóa luận tốt nghiệp", []) == ["khoa luan tot nghiep"]


def test_course_rows_read_the_older_plan_format() -> None:
    """Kế hoạch đào tạo ngành Công nghệ thông tin khóa 16: 7-digit codes and both languages in
    the name cell - no row was read, and the answer said there was nothing (25/11/2026)."""

    content = (
        "| TT | Mã môn học | Tên môn học | Mã học phần | Số tín chỉ | Học phần: học trước(a), "
        "tiên quyết(b), song hành(c) | Ghi chú |\n"
        "| 1 | 2101700 | Khóa luận tốt nghiệp Capstone project | 2101700 | 8(0,16,16) |  |  |\n"
        "| 2 | 2101622 | Lập trình hướng đối tượng Object-Oriented Programming | 2101622 "
        "| 3(2,2,6) "
        "| 2101539(a) |  |\n"
        "| 3 | 2101000 | Lập trình Java | 2101000 | 3(2,2,6) |  |  |\n"
        "| 4 | 2113431 | Toán cao cấp 1 Calculus 1 | 2113431 | 2(1,2,4) |  |  |\n"
    )

    rows = list(course_rows(content))

    assert [(r.code, r.name, r.credits, r.conditions) for r in rows] == [
        ("2101700", "Khóa luận tốt nghiệp", "8(0,16,16)", ()),
        ("2101622", "Lập trình hướng đối tượng", "3(2,2,6)", (("2101539", "a"),)),
        ("2101000", "Lập trình Java", "3(2,2,6)", ()),
        ("2113431", "Toán cao cấp 1 Calculus 1", "2(1,2,4)", ()),
    ]
    assert asked_courses([r.name for r in rows], None, ["khóa luận tốt nghiệp ngành cntt"]) == [
        "khoa luan tot nghiep"
    ]
