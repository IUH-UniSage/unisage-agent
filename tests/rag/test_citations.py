from app.rag.prompting.citations import (
    build_citations,
    cited_indexes,
    keep_known_links,
    source_title,
    strip_source_list,
)
from app.schemas.retrieval import RetrievedChunk
from app.schemas.web_search import WebSearchResult

_KEY = "a63a8c6f-3715-4d03-a464-13116e18bed7_Quyet dinh 1035 QD DHCN Hoc phi 2025-2026.pdf"


def _chunk(chunk_id: str, **overrides: object) -> RetrievedChunk:
    values: dict[str, object] = {
        "chunk_id": chunk_id,
        "content": "Nội dung",
        "source": _KEY,
        "score": 1.0,
        "heading_path": ["QUYẾT ĐỊNH", "Khối Công nghệ"],
        "page_start": 3,
        "page_end": 4,
        "source_type": "PDF",
        "metadata": {"document_id": "doc-1"},
    }
    values.update(overrides)
    return RetrievedChunk.model_validate(values)


def test_source_title_drops_uuid_prefix_and_extension() -> None:
    assert source_title(_KEY) == "Quyet dinh 1035 QD DHCN Hoc phi 2025-2026"
    assert source_title("Quy chế A") == "Quy chế A"


def test_cited_indexes_keeps_only_valid_indexes_in_first_seen_order() -> None:
    text = "A [2]. B [1][2]. C [9]. D [1, 3]. E [0]."

    assert cited_indexes(text, 3) == [2, 1, 3]


def test_build_citations_uses_chunk_metadata_and_never_exposes_object_key() -> None:
    citations = build_citations("Học phí là 35 triệu [1].", [_chunk("c1"), _chunk("c2")])

    assert citations == [
        {
            "index": 1,
            "documentId": "doc-1",
            "title": "Quyet dinh 1035 QD DHCN Hoc phi 2025-2026",
            "section": "Khối Công nghệ",
            "pageStart": 3,
            "pageEnd": 4,
            "sourceType": "PDF",
        }
    ]
    assert "objectKey" not in citations[0]
    assert _KEY not in str(citations)


def test_build_citations_tolerates_chunk_without_document_id_or_headings() -> None:
    chunk = _chunk("c1", metadata={}, heading_path=[], page_start=None, page_end=None)

    (citation,) = build_citations("Nội dung [1]", [chunk])

    assert citation["documentId"] is None
    assert citation["section"] is None
    assert citation["pageStart"] is None


def test_build_citations_is_empty_without_markers_or_chunks() -> None:
    assert build_citations("Không có nguồn", [_chunk("c1")]) == []
    assert build_citations("Có [1]", []) == []


def _web(title: str) -> WebSearchResult:
    return WebSearchResult(
        title=title, url=f"https://pdt.iuh.edu.vn/{title}", content="...", score=0.8
    )


def test_markers_after_the_chunks_cite_web_pages_in_listed_order() -> None:
    chunks = [
        RetrievedChunk(chunk_id="c1", content="a", source="hoc-phi.pdf", score=0.9),
        RetrievedChunk(chunk_id="c2", content="b", source="quy-che.pdf", score=0.8),
    ]

    citations = build_citations(
        "Học phí [1]. Lịch thi [4]. Thêm [3].", chunks, [_web("thong-bao"), _web("lich-thi")]
    )

    assert [c["index"] for c in citations] == [1, 4, 3]
    assert citations[0]["title"] == "hoc-phi"
    assert "url" not in citations[0]
    assert citations[1] == {
        "index": 4,
        "documentId": None,
        "title": "lich-thi",
        "section": None,
        "pageStart": None,
        "pageEnd": None,
        "sourceType": "WEB",
        "url": "https://pdt.iuh.edu.vn/lich-thi",
    }
    assert citations[2]["url"] == "https://pdt.iuh.edu.vn/thong-bao"


def test_web_only_turn_numbers_pages_from_one_and_drops_out_of_range_markers() -> None:
    citations = build_citations("Theo website [1], còn [2] không có.", [], [_web("lich-thi")])

    assert [(c["index"], c["sourceType"]) for c in citations] == [(1, "WEB")]


def test_strip_source_list_drops_the_list_the_model_wrote_at_the_end() -> None:
    """The 14-10-2026 answer: the cited page's address repeated under the reply."""

    text = (
        "Viên chức được nghỉ lễ Quốc khánh trong 05 ngày [1].\n\n"
        "[1] https://iuh.edu.vn/upload/files/2026/08/TB-957-2026.pdf"
    )

    assert strip_source_list(text) == "Viên chức được nghỉ lễ Quốc khánh trong 05 ngày [1]."
    assert (
        strip_source_list(
            "Ý A [1]. Ý B [2].\n\n---\n**Nguồn:**\n- [1]: Thông báo A\n- [2]: Quy chế B"
        )
        == "Ý A [1]. Ý B [2]."
    )


def test_strip_source_list_keeps_an_answer_without_one() -> None:
    text = "Ý A [1].\n\n---\n\nNguồn\n\nBạn có thể tạo ticket để được hỗ trợ."
    assert strip_source_list(text) == text
    assert strip_source_list("Ý A [1].\n\n---") == "Ý A [1].\n\n---"


def test_a_link_the_model_wrote_must_be_one_of_the_pages_it_was_given() -> None:
    page = WebSearchResult(
        title="TB",
        url="https://iuh.edu.vn/thong-bao/nghi-le-quoc-khanh-2026",
        content="",
        score=0.5,
    )
    text = (
        "Xem [thông báo](https://iuh.edu.vn/thong-bao/nghi-le-quoc-khánh-2026), "
        "[trang lạ](https://iuh.edu.vn/khong-co) và https://iuh.edu.vn/THONG-BAO/nghi-le-quoc-khanh-2026."
    )

    assert keep_known_links(text, [page]) == (
        f"Xem [thông báo]({page.url}), trang lạ và {page.url}."
    )
