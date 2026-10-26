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
    assert (
        ungrounded_numbers(
            "Tổng 109 tín chỉ, còn 31.", ["HK1 20, HK2 25, HK3 21, HK4 22, HK5 21; tổng 140"]
        )
        == []
    )


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
