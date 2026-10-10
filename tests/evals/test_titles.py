from pathlib import Path

import pymupdf

from evals.titles import (
    apply_titles,
    clean_title,
    page_inputs,
    rename_files,
    title_problem,
    upload_name,
)


def _row(file_id: str, department: str = "PHONG_DAO_TAO", **extra: str) -> dict[str, str]:
    return {"file_id": file_id, "department_id": department, "selected": "true", **extra}


def test_clean_title_drops_uuid_prefix_and_joins_lines() -> None:
    raw = "5de416ca-e39b-44dd-9b35-1ae9caf7ebb0_Đơn đề nghị\n  xét học bổng "
    assert clean_title(raw) == "Đơn đề nghị xét học bổng"


def test_clean_title_normalizes_to_nfc() -> None:
    decomposed = "Học phí"
    assert clean_title(decomposed) == "Học phí"


def test_clean_title_cuts_long_titles_at_a_word() -> None:
    title = clean_title("Quy chế " + "đào tạo " * 40)
    assert len(title) <= 150
    assert not title.endswith(" ")


def test_upload_name_replaces_characters_no_os_allows() -> None:
    assert (
        upload_name("Quyết định 1035/QĐ-ĐHCN: học phí?") == "Quyết định 1035-QĐ-ĐHCN- học phí.pdf"
    )


def test_title_problem_flags_ids_and_short_codes() -> None:
    assert title_problem("a1b2c3d4e5f6", "a1b2c3d4e5f6") == "looks like an id"
    assert title_problem("CTDT", "x") == "too short"
    assert title_problem("", "x") == "empty"
    assert title_problem("Chuẩn đầu ra ngành Kế toán", "x") is None


def test_apply_titles_sets_title_and_source() -> None:
    rows = [_row("f1")]
    review = apply_titles(
        rows, [{"file_id": "f1", "title": "Kế hoạch đào tạo ngành Kế toán", "basis": "heading"}]
    )
    assert rows[0]["title"] == "Kế hoạch đào tạo ngành Kế toán"
    assert rows[0]["title_source"] == "heading"
    assert review == []


def test_apply_titles_never_overwrites_manual_rows() -> None:
    rows = [_row("f1", title="Tên sửa tay", title_source="manual")]
    apply_titles(rows, [{"file_id": "f1", "title": "Tên khác", "basis": "heading"}])
    assert rows[0]["title"] == "Tên sửa tay"


def test_apply_titles_reviews_low_confidence_and_same_title_in_department() -> None:
    rows = [_row("f1"), _row("f2"), _row("f3", department="KHOA_CNTT")]
    titles = [
        {"file_id": "f1", "title": "Thông báo tuyển sinh 2025", "basis": "content"},
        {"file_id": "f2", "title": "Thông báo tuyển sinh 2025", "basis": "content"},
        {"file_id": "f3", "title": "Thông báo tuyển sinh 2025", "basis": "x", "confidence": "low"},
    ]
    review = apply_titles(rows, titles)
    reasons = {(row["file_id"], row["reason"].split(":")[0]) for row in review}
    assert ("f1", "same title in department") in reasons
    assert ("f2", "same title in department") in reasons
    assert ("f3", "low confidence") in reasons
    assert rows[2]["title_source"] == "content"


def test_page_inputs_returns_largest_font_line_first(tmp_path: Path) -> None:
    path = tmp_path / "doc.pdf"
    with pymupdf.open() as document:
        page = document.new_page()
        page.insert_text((50, 80), "BIG TITLE", fontsize=24)
        page.insert_text((50, 120), "small body text", fontsize=9)
        document.save(path)

    inputs = page_inputs(path)
    assert inputs["headings"][0] == "BIG TITLE"
    assert "small body text" in inputs["text"]


def test_rename_files_moves_to_title_and_updates_local_path(tmp_path: Path) -> None:
    (tmp_path / "files" / "khoa").mkdir(parents=True)
    (tmp_path / "files" / "khoa" / "a1.pdf").write_bytes(b"1")
    rows = [{"file_id": "a1", "local_path": "files/khoa/a1.pdf", "title": "Học phí 1035/QĐ"}]

    moved = rename_files(tmp_path, rows)

    assert rows[0]["local_path"] == "files/khoa/Học phí 1035-QĐ.pdf"
    assert (tmp_path / rows[0]["local_path"]).read_bytes() == b"1"
    assert moved == [
        {"file_id": "a1", "old_path": "files/khoa/a1.pdf", "new_path": rows[0]["local_path"]}
    ]
    assert rename_files(tmp_path, rows) == []


def test_rename_files_keeps_names_unique_ignoring_case(tmp_path: Path) -> None:
    folder = tmp_path / "files" / "khoa"
    folder.mkdir(parents=True)
    for file_id in ("a1", "b2"):
        (folder / f"{file_id}.pdf").write_bytes(file_id.encode())
    rows = [
        {"file_id": "a1", "local_path": "files/khoa/a1.pdf", "title": "Thông báo học phí"},
        {"file_id": "b2", "local_path": "files/khoa/b2.pdf", "title": "THÔNG BÁO HỌC PHÍ"},
    ]

    rename_files(tmp_path, rows)

    assert rows[1]["local_path"] == "files/khoa/THÔNG BÁO HỌC PHÍ (b2).pdf"
    assert (tmp_path / rows[1]["local_path"]).read_bytes() == b"b2"
