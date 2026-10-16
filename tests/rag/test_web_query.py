"""Web search policy (app/rag/web_query.py): query templates, page ordering and filtering,
the academic year of a page, cache TTLs."""

from datetime import date

from app.rag.web_query import (
    ABBREVIATIONS,
    PHRASE_ALIASES,
    SPELLING,
    SUBJECT_ALIASES,
    academic_year_of,
    build_web_query,
    cache_ttl_seconds,
    campus_of,
    is_listing_page,
    is_postgraduate,
    normalize_phrases,
    order_candidates,
    relative_year,
)
from app.schemas.web_search import WebSearchResult, WebSlots

_OCT_2026 = date(2026, 10, 14)


def _page(url: str, score: float, title: str = "t", content: str = "c") -> WebSearchResult:
    return WebSearchResult(title=title, url=url, content=content, score=score)


def test_fee_query_has_the_school_but_no_major_and_no_guessed_year() -> None:
    """The 14-10-2026 case: "học phí ngành cnnt". Fee schedules are split by Khối, so the
    major's name would only lower the right page's rank; the user named no year, so none
    is added (the current one pushed the latest fee decision out of the top 10)."""

    slots = WebSlots(
        topics=["hoc_phi"],
        subject="mức thu học phí",
        bac_dao_tao="dai_hoc",
        he_dao_tao="chính quy",
        nganh_hoc="CNTT",
    )

    assert build_web_query(slots) == "mức thu học phí Đại học Chính quy IUH"


def test_a_year_the_user_named_is_kept() -> None:
    fee = WebSlots(topics=["hoc_phi"], subject="mức thu học phí", nam_hoc="2025 - 2026")
    notice = WebSlots(topics=["lich_thong_bao"], subject="lịch thi", nam_hoc="2026")

    assert build_web_query(fee) == "mức thu học phí năm học 2025-2026 IUH"
    assert build_web_query(notice) == "lịch thi năm 2026 IUH"


def test_admission_query_keeps_the_major_and_the_year_named() -> None:
    slots = WebSlots(
        topics=["tuyen_sinh"],
        subject="điểm chuẩn",
        nganh_hoc="it",
        bac_dao_tao="Đại học",
        nam_hoc="2026-2027",
    )

    assert build_web_query(slots) == "điểm chuẩn Công nghệ Thông tin Đại học năm 2026 IUH"
    assert build_web_query(slots.model_copy(update={"nam_hoc": None})) == (
        "điểm chuẩn Công nghệ Thông tin Đại học IUH"
    )


def test_contact_query_has_the_campus_and_no_year() -> None:
    slots = WebSlots(topics=["lien_he_co_so"], subject="địa chỉ", co_so="Quảng Ngãi")

    assert build_web_query(slots) == "địa chỉ Quảng Ngãi IUH"


def test_school_name_is_not_repeated() -> None:
    slots = WebSlots(topics=["khac"], subject="xếp hạng IUH")

    assert build_web_query(slots) == "xếp hạng IUH"


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
    assert is_listing_page("https://www.iuh.edu.vn/vi/thong-bao.html")
    assert is_listing_page("https://www.iuh.edu.vn/vi/thong-tin-tuyen-sinh.html")
    assert is_listing_page("https://iuh.edu.vn/vi/thong-bao.html/p=13")
    # A detail page with a page parameter is still a detail page.
    assert not is_listing_page(
        "https://itc.iuh.edu.vn/detail@91-1541-Thong-bao-ke-hoach-hoc-va-thi-cac-lop-chung-chi"
        "-ung-dung-CNTT-co-ban-khoa-thang-3,4,5,6,7,8.html@p=2"
    )
    assert not is_listing_page(
        "https://www.iuh.edu.vn/vi/thong-bao/thong-bao-v-v-dang-ky-hoc-phan-2188.html"
    )
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

    ordered = order_candidates(pages, topic="hoc_phi", floor=0.05)

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


def test_the_score_floor_never_drops_a_page_of_the_topics_own_sites() -> None:
    fee_decision = _page("https://ptckt.iuh.edu.vn/upload/files/bieumau/1035-QD-DHCN.pdf", 0.03)
    weak_news = _page("https://iuh.edu.vn/vi/tin-tuc/hoc-bong", 0.03)

    ordered = order_candidates([weak_news, fee_decision], topic="hoc_phi", floor=0.05)

    assert ordered == [fee_decision.model_copy(update={"academic_year": None})]
    # Another topic does not trust ptckt: the floor applies there.
    assert order_candidates([fee_decision], topic="tuyen_sinh", floor=0.05) == []


def test_a_campus_notice_is_kept_last_only_for_the_universitys_own_notices() -> None:
    """The 2025 National Day notice was found only on the Quảng Ngãi campus site."""

    campus = _page(
        "https://qn.iuh.edu.vn/thong-bao/thong-bao-v-v-nghi-le-quoc-khanh-nam-2025", 0.65
    )
    main = _page("https://www.iuh.edu.vn/vi/thong-bao/thong-bao-nghi-le-2232.html", 0.47)
    other = _page("https://fct.iuh.edu.vn/news.html@detail@264@682@Thong-bao", 0.19)

    notices = order_candidates([campus, main, other], topic="lich_thong_bao", floor=0.05)
    fees = order_candidates([campus, main, other], topic="hoc_phi", floor=0.05)

    assert [page.url for page in notices] == [main.url, other.url, campus.url]
    assert campus.url not in [page.url for page in fees]


def test_campus_sites_are_known_by_name() -> None:
    thanh_hoa = _page("https://th.iuh.edu.vn/thong-bao/tien-do-dao-tao-2026-2027", 0.6)

    assert campus_of(thanh_hoa.url) == "Cơ sở Thanh Hóa"
    assert campus_of("https://pdt.iuh.edu.vn/thong-bao") is None
    # Kept for a fee question only when it names that campus, whichever accent it uses.
    assert order_candidates([thanh_hoa], topic="hoc_phi", floor=0.05) == []
    assert order_candidates([thanh_hoa], topic="hoc_phi", co_so="Thanh Hoá", floor=0.05)


def test_a_year_already_in_the_subject_is_not_added_again() -> None:
    slots = WebSlots(
        topics=["lich_thong_bao"], subject="lịch nghỉ lễ Quốc khánh 2025", nam_hoc="2025"
    )

    assert build_web_query(slots) == "lịch nghỉ lễ Quốc khánh 2025 IUH"


def test_a_postgraduate_page_is_left_out_unless_the_question_is_about_that_level() -> None:
    """The 14-10-2026 turn: a master's admission notice gave its fee for an undergraduate
    major."""

    masters = _page(
        "https://faet.iuh.edu.vn/news.html@detail@272@667@Thong-bao-tuyen-sinh-dao-tao-trinh-do-Thac-si-nam-2026",
        0.72,
        title="Thông báo tuyển sinh đào tạo trình độ Thạc sĩ năm 2026 - Đợt 1",
    )
    fees = _page("https://ptckt.iuh.edu.vn/upload/hoc-phi-2026-2027.pdf", 0.3)

    undergraduate = order_candidates([masters, fees], topic="hoc_phi", floor=0.05)
    postgraduate = order_candidates([masters, fees], topic="hoc_phi", postgraduate=True, floor=0.05)

    assert [page.url for page in undergraduate] == [fees.url]
    assert masters.url in [page.url for page in postgraduate]
    assert is_postgraduate("Thạc sĩ") and is_postgraduate("chuong-trinh-cao-hoc")
    assert not is_postgraduate("Học phí đại học chính quy")


def test_a_relative_time_word_means_this_year_or_this_academic_year() -> None:
    """For when the classifier leaves the year out ("đợt gần nhất", 14-10-2026)."""

    assert relative_year("Lịch thi chứng chỉ CNTT đợt gần nhất là khi nào?", _OCT_2026) == "2026"
    assert relative_year("học phí kỳ này", _OCT_2026) == "2026-2027"
    assert relative_year("lịch nghỉ tết năm sau", _OCT_2026) == "2027"
    assert relative_year("ngành hiện tại của tôi", _OCT_2026) is None
    assert relative_year("điều kiện học bổng", _OCT_2026) is None


def test_the_same_fee_question_becomes_the_same_query_however_the_subject_is_written() -> None:
    """16-10-2026: "học phí ngành cntt" asked twice became "học phí IUH" and "mức thu học
    phí ngành Công nghệ thông tin IUH" - two cache entries, and the second's top page a
    study-abroad notice."""

    written = [
        WebSlots(topics=["hoc_phi"], subject="học phí", nganh_hoc="Công nghệ Thông tin"),
        WebSlots(
            topics=["hoc_phi"],
            subject="mức thu học phí ngành Công nghệ thông tin",
            nganh_hoc="Công nghệ thông tin",
        ),
        WebSlots(topics=["hoc_phi"], subject="mức thu học phí ngành CNTT", nganh_hoc="cntt"),
        # The major written into the subject only, its slot left empty.
        WebSlots(topics=["hoc_phi"], subject="mức thu học phí ngành Công nghệ thông tin"),
        WebSlots(topics=["hoc_phi"], subject="mức học phí"),
    ]

    assert {build_web_query(slots) for slots in written} == {"mức thu học phí IUH"}


def test_a_slot_written_into_the_subject_is_added_back_by_the_template_only() -> None:
    fee = WebSlots(
        topics=["hoc_phi"],
        subject="học phí hệ chính quy cơ sở Quảng Ngãi",
        he_dao_tao="chính quy",
        co_so="Quảng Ngãi",
    )
    admission = WebSlots(topics=["tuyen_sinh"], subject="điểm trúng tuyển ngành Kế toán")

    assert build_web_query(fee) == "mức thu học phí Chính quy Quảng Ngãi IUH"
    # Admission's template names the major: moved out of the subject, then put back.
    assert build_web_query(admission) == "điểm chuẩn Kế toán IUH"


def test_a_subject_about_something_else_keeps_its_words() -> None:
    exemption = WebSlots(topics=["hoc_phi"], subject="miễn giảm học phí", nganh_hoc="cntt")
    change = WebSlots(topics=["quy_che_thu_tuc"], subject="điều kiện chuyển ngành")
    only_major = WebSlots(
        topics=["chuong_trinh_dao_tao"], subject="Công nghệ thông tin", nganh_hoc="CNTT"
    )
    only_major_fee = WebSlots(topics=["hoc_phi"], subject="ngành CNTT", nganh_hoc="CNTT")

    assert build_web_query(exemption) == "miễn giảm học phí IUH"
    assert build_web_query(change) == "điều kiện chuyển ngành IUH"
    # The subject was only the major: the template names it once...
    assert build_web_query(only_major) == "Công nghệ Thông tin IUH"
    # ...or, leaving the major out, keeps the subject as written rather than search "IUH".
    assert build_web_query(only_major_fee) == "ngành CNTT IUH"


def test_vocabulary_keys_are_lowercase() -> None:
    """Every lookup lowercases first: a key in capitals would never match."""

    tables = {"spelling": SPELLING, "abbreviations": ABBREVIATIONS, "phrases": PHRASE_ALIASES}
    tables |= {f"subject_aliases.{topic}": table for topic, table in SUBJECT_ALIASES.items()}
    for name, table in tables.items():
        assert [key for key in table if key != key.lower()] == [], name


def test_a_canonical_wording_is_left_as_it_is() -> None:
    """A key that starts its own canonical form ("tạm dừng học" -> "tạm dừng học tập")
    rewrote an already canonical subject into "tạm dừng học tập tập" until the canonical
    form was added as a key of its own."""

    canonical = {*SPELLING.values(), *ABBREVIATIONS.values(), *PHRASE_ALIASES.values()}
    canonical |= {value for table in SUBJECT_ALIASES.values() for value in table.values()}
    assert {
        value: normalize_phrases(value) for value in canonical if normalize_phrases(value) != value
    } == {}


def test_subject_alias_keys_are_written_as_the_phrase_tables_leave_them() -> None:
    """Whole-subject aliases are looked up after spelling, abbreviations and phrases: a key
    those would rewrite ("tkb", "học kì") is never reached."""

    stale = {
        key: normalize_phrases(key)
        for table in SUBJECT_ALIASES.values()
        for key in table
        if normalize_phrases(key).lower() != key
    }
    assert stale == {}


def test_abbreviations_terms_and_synonyms_inside_a_longer_subject() -> None:
    def query(topic: str, subject: str) -> str:
        return build_web_query(WebSlots(topics=[topic], subject=subject))  # type: ignore[list-item]

    assert query("lich_thong_bao", "lịch thi cuối kì hk2") == ("lịch thi cuối kỳ học kỳ II IUH")
    # An abbreviation expands before the synonyms, which then see the full wording.
    assert query("lich_thong_bao", "thời gian đkhp kỳ 3") == "lịch đăng ký học phần học kỳ III IUH"
    assert query("hoc_phi", "hạn nộp học phí hk1") == "hạn đóng học phí học kỳ I IUH"
    assert query("lien_he_co_so", "sđt pđt") == "số điện thoại Phòng Đào tạo IUH"
    # A phrase is replaced as whole words only, never as the start of a longer compound.
    assert query("hoc_phi", "tiền học bổng") == "tiền học bổng IUH"
    assert query("quy_che_thu_tuc", "tạm dừng học tập") == "thủ tục tạm dừng học tập IUH"


def test_whole_subject_aliases_keep_different_procedures_apart() -> None:
    def query(topic: str, subject: str) -> str:
        return build_web_query(WebSlots(topics=[topic], subject=subject))  # type: ignore[list-item]

    assert query("quy_che_thu_tuc", "học lại") == "đăng ký học lại IUH"
    assert query("quy_che_thu_tuc", "học cải thiện") == "đăng ký học cải thiện IUH"
    assert query("quy_che_thu_tuc", "điều kiện chuyển ngành") == "điều kiện chuyển ngành IUH"
    assert query("quy_che_thu_tuc", "xin chuyển ngành") == "thủ tục chuyển ngành IUH"
    # `_all` applies whatever topic the classifier picked.
    assert query("khac", "giá ktx") == query("hoc_phi", "tiền ký túc xá") == "phí ký túc xá IUH"


def test_a_slot_spelled_with_the_old_accent_placement_is_still_taken_out() -> None:
    slots = WebSlots(topics=["lien_he_co_so"], subject="địa chỉ cơ sở Thanh Hoá", co_so="Thanh Hóa")

    assert build_web_query(slots) == "địa chỉ Thanh Hóa IUH"
