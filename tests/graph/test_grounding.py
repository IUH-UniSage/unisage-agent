"""Grounding checks: numbers and citations an answer uses versus what the model saw."""

from app.graph.grounding import (
    caution_line,
    check_answer,
    check_citations,
    numbers_in,
    ungrounded_numbers,
)

FEE = "Khối Công nghệ: 38.350.000 đồng/năm; học phí một tín chỉ lý thuyết 1.150.000 đồng."
CREDITS = "Học kỳ 5 có 18 tín chỉ, trong đó 12 tín chỉ bắt buộc; mỗi học phần tự chọn 3 tín chỉ."


def test_numbers_in_reads_vietnamese_formats_and_skips_dates_and_markers() -> None:
    found = numbers_in("Nộp 38.350.000 đồng trước 15/08/2026, điểm 7,9 [1][2] và 1,5 lần")

    assert found == {"38.350.000": 38350000.0, "7,9": 7.9, "1,5": 1.5}


def test_numbers_present_in_cited_sources_question_or_known_are_grounded() -> None:
    answer = (
        "Học phí là **38.350.000 đồng** [1], tức 1.150.000 đồng mỗi tín chỉ; bạn đã học 90 tín chỉ."
    )

    assert ungrounded_numbers(answer, [FEE], question="tôi đã học 90 tín chỉ") == []
    assert ungrounded_numbers(answer, [FEE], known={"tc_da_hoc": 90}) == []


def test_an_invented_figure_is_reported() -> None:
    answer = "Học phí là 42.500.000 đồng/năm [1]."

    assert ungrounded_numbers(answer, [FEE]) == ["42.500.000"]


def test_results_of_one_arithmetic_step_or_a_sum_are_accepted() -> None:
    # 18 - 12 = 6 ; 6 / 3 = 2 (small, skipped anyway) ; 3 + 4 + 2 + 3 = 12 (small) ; 15 x 1.150.000
    answer = "Cần thêm 6 tín chỉ, tức 2 học phần; 15 tín chỉ x 1.150.000 = 17.250.000 đồng."

    assert ungrounded_numbers(answer, [CREDITS, FEE], question="15 tín chỉ") == []
    sources = ["HK1 20, HK2 25, HK3 21, HK4 22, HK5 21; tổng 140"]
    # A total whose terms the answer lists is derivable; a bare total is not (the reader
    # cannot verify it either), and neither is a remainder built on it.
    assert (
        ungrounded_numbers(
            "Đã học 20 + 25 + 21 + 22 + 21 = 109 tín chỉ, còn 140 − 109 = 31.", sources
        )
        == []
    )
    assert ungrounded_numbers("Tổng 109 tín chỉ, còn 31.", sources) == ["109", "31"]


def test_small_integers_and_years_are_not_flagged() -> None:
    assert ungrounded_numbers("Có 3 bước, nộp trong năm 2026, mục 11.", ["không có số nào"]) == []


def test_check_answer_scopes_to_cited_sources_when_there_are_citations() -> None:
    sources = [FEE, CREDITS]

    cited = check_answer("Học kỳ 5 có 18 tín chỉ và học phí 38.350.000 đồng [2].", sources)
    uncited = check_answer("Học kỳ 5 có 18 tín chỉ và học phí 38.350.000 đồng.", sources)

    # [2] is CREDITS: the fee figure is not in the cited source.
    assert cited.ungrounded == ["38.350.000"] and cited.cited == [2] and cited.no_citation is False
    assert uncited.ungrounded == [] and uncited.no_citation is True


def test_markers_outside_the_sources_are_reported() -> None:
    check = check_citations("Theo [1] và [4], xem thêm [2, 9].", 3)

    assert check.cited == [1, 2] and check.invalid == [4, 9]
    assert check_answer("Theo [4].", [FEE]).invalid_markers == [4]


def test_caution_line_lists_the_numbers() -> None:
    assert "42.500.000" in caution_line(["42.500.000"]) and "kiểm tra lại" in caution_line(["1"])


def test_a_number_rich_table_does_not_make_invented_figures_derivable() -> None:
    import random

    random.seed(0)
    table = " ".join(
        f"{random.randint(1, 99)}.{random.randint(100, 999)}.000 đồng" for _ in range(300)
    )
    invented = ["487.213.000", "612.907.000", "73.456.000"]

    flagged = ungrounded_numbers("Phải nộp " + ", ".join(invented) + " đồng [1].", [table])

    # 73.456.000 happens to equal 71.748.000 + 1.708.000 in this table: an additive
    # collision the check accepts by design (one step from two source numbers). The two
    # multiplicative near-misses (51,006 x a fee) are caught since ratios must be exact.
    assert flagged == invented[:2]


def test_day_and_hour_spans_between_dates_and_times_in_the_sources_are_grounded() -> None:
    sources = [
        "Chứng chỉ có thời hạn 02 năm, cấp ngày 20/05/2024, nộp ngày 10/06/2026; "
        "hỗ trợ từ 06h30 đến 19h40; giá 250.000đ giảm 60.000đ."
    ]
    # 730 = 2 x 365; 21 = 10/06/2026 - (20/05/2024 + 2 năm); 13 = 19h40 - 06h30; 190.000 = 250.000 - 60.000
    answer = "Thời hạn 2 năm = 730 ngày, quá hạn 21 ngày; mỗi ngày 13 giờ 10 phút; trả 190.000đ."

    assert ungrounded_numbers(answer, sources) == []
    assert ungrounded_numbers("Quá hạn 1.234 ngày.", sources) == ["1.234"]


def test_phone_numbers_and_extensions_are_not_figures() -> None:
    assert (
        ungrounded_numbers("Gọi 028 3588 2679 hoặc máy lẻ 112, Ext: 3894.", ["không có số"]) == []
    )
    assert numbers_in("hotline 0283 894 0390, học phí 1.150.000") == {"1.150.000": 1150000.0}
