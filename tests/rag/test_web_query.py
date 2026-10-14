"""Web search policy (app/rag/web_query.py): query templates, page ordering and filtering,
the academic year of a page, cache TTLs."""

from datetime import date

from app.rag.web_query import (
    academic_year_of,
    build_web_query,
    cache_ttl_seconds,
    is_listing_page,
    order_candidates,
)
from app.schemas.web_search import WebSearchResult, WebSlots

_OCT_2026 = date(2026, 10, 14)


def _page(url: str, score: float, title: str = "t", content: str = "c") -> WebSearchResult:
    return WebSearchResult(title=title, url=url, content=content, score=score)


def test_fee_query_has_school_and_current_year_but_no_major() -> None:
    """The 14-10-2026 case: "học phí ngành cnnt". Fee schedules are split by Khối, so the
    major's name would only lower the right page's rank."""

    slots = WebSlots(
        topics=["hoc_phi"],
        subject="mức thu học phí",
        bac_dao_tao="dai_hoc",
        he_dao_tao="chính quy",
        nganh_hoc="CNTT",
    )

    assert build_web_query(slots, today=_OCT_2026) == (
        "mức thu học phí Đại học Chính quy năm học 2026-2027 IUH"
    )


def test_named_academic_year_wins_over_the_current_one() -> None:
    slots = WebSlots(topics=["hoc_phi"], subject="mức thu học phí", nam_hoc="2025 - 2026")

    assert build_web_query(slots, today=_OCT_2026) == "mức thu học phí năm học 2025-2026 IUH"


def test_admission_query_keeps_the_major_and_the_calendar_year() -> None:
    slots = WebSlots(
        topics=["tuyen_sinh"], subject="điểm chuẩn", nganh_hoc="it", bac_dao_tao="Đại học"
    )

    assert build_web_query(slots, today=_OCT_2026) == (
        "điểm chuẩn Công nghệ Thông tin Đại học năm 2026 IUH"
    )


def test_contact_query_has_the_campus_and_no_year() -> None:
    slots = WebSlots(topics=["lien_he_co_so"], subject="địa chỉ", co_so="Quảng Ngãi")

    assert build_web_query(slots, today=_OCT_2026) == "địa chỉ Quảng Ngãi IUH"


def test_school_name_is_not_repeated() -> None:
    slots = WebSlots(topics=["khac"], subject="xếp hạng IUH")

    assert build_web_query(slots, today=_OCT_2026) == "xếp hạng IUH"


def test_in_the_break_before_the_school_year_the_coming_year_is_searched() -> None:
    """Fees and notices for the year about to start are what a student asks for in August."""

    slots = WebSlots(topics=["hoc_phi"], subject="mức thu học phí")

    assert build_web_query(slots, today=date(2026, 9, 1)).endswith("năm học 2026-2027 IUH")


def test_academic_year_is_read_from_title_url_or_content() -> None:
    assert academic_year_of(_page("https://x/a", 0.5, title="Học phí năm học 2025-2026")) == (
        "2025-2026"
    )
    assert (
        academic_year_of(_page("https://pdt.iuh.edu.vn/TB_Dang20hoc20HK1-nam202026-2027.pdf", 0.5))
        == "2026-2027"
    )
    assert academic_year_of(
        _page("https://x/a", 0.5, content="học kỳ I năm học 2026 \u2013 2027")
    ) == ("2026-2027")
    # Two unrelated years are not an academic year.
    assert academic_year_of(_page("https://x/2019-2026", 0.5)) is None


def test_listing_pages_are_recognised() -> None:
    assert is_listing_page("https://fit.iuh.edu.vn")
    assert is_listing_page("https://iuh.edu.vn/")
    assert is_listing_page("https://ctsv.iuh.edu.vn/news.html@150@Thong-bao")
    assert is_listing_page("https://ipe.iuh.edu.vn/vi/author/namlt/page/26")
    assert not is_listing_page(
        "https://fit.iuh.edu.vn/news.html@detail@155@3548@Thong-bao-dang-ky-hoc-phan-HK1-2026-2027"
    )
    assert not is_listing_page("https://ptckt.iuh.edu.vn/upload/files/bieumau/1035-QD-DHCN.pdf")


def test_candidates_put_the_topics_own_sites_first() -> None:
    pages = [
        _page("https://iuh.edu.vn/vi/tin-tuc/hoc-bong", 0.9),
        _page("https://ptckt.iuh.edu.vn/upload/qd-hoc-phi.pdf", 0.3),
        _page("https://www.iuh.edu.vn/vi/thong-bao/hoc-phi", 0.2),
    ]

    ordered = order_candidates(pages, topic="hoc_phi", floor=0.05, limit=5)

    assert [page.url for page in ordered] == [
        "https://ptckt.iuh.edu.vn/upload/qd-hoc-phi.pdf",
        "https://www.iuh.edu.vn/vi/thong-bao/hoc-phi",
        "https://iuh.edu.vn/vi/tin-tuc/hoc-bong",
    ]


def test_cache_ttl_follows_the_topic_and_the_admission_season() -> None:
    assert cache_ttl_seconds("hoc_phi", today=_OCT_2026, empty=False) == 24 * 3600
    assert cache_ttl_seconds("lich_thong_bao", today=_OCT_2026, empty=False) == 3 * 3600
    assert cache_ttl_seconds("lien_he_co_so", today=_OCT_2026, empty=False) == 7 * 24 * 3600
    assert cache_ttl_seconds("tuyen_sinh", today=_OCT_2026, empty=False) == 6 * 3600
    assert cache_ttl_seconds("tuyen_sinh", today=date(2026, 7, 20), empty=False) == 3600
    assert cache_ttl_seconds(None, today=_OCT_2026, empty=False) == 24 * 3600
    assert cache_ttl_seconds("hoc_phi", today=_OCT_2026, empty=True) == 30 * 60
