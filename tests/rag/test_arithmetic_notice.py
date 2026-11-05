"""`{arithmetic_notice}` (04/11/2026): a deferred calculation turn gets the strict notice;
a plain turn asking a derived quantity (total, difference, rounding, verdict) gets the
softer one, so the answer states it through a calc_request (q0050, q0054, q0071)."""

from app.rag.prompting.builder import (
    ARITHMETIC_NOTICE,
    CALC_TOOLS_NOTICE,
    DERIVED_QUANTITY_NOTICE,
    asks_derived_quantity,
    build_arithmetic_notice,
)


def test_derived_quantity_questions_are_detected() -> None:
    for question in (
        "tổng số tín chỉ của Anh văn 1 và Anh văn 2 là bao nhiêu?",
        "Tổng cộng bao nhiêu tín chỉ và còn được đăng ký thêm tối đa bao nhiêu?",
        "sau học kỳ 4 còn lại bao nhiêu tín chỉ?",
        "Các điểm 6,2; 7,6 sau khi làm tròn là bao nhiêu?",
        "điểm tổng kết là bao nhiêu và có đạt không?",
    ):
        assert asks_derived_quantity(question), question
    for question in (
        "Điều kiện học bổng là gì?",
        "Còn học phí thì sao?",
        "Hướng dẫn truy cập Wi-Fi",
    ):
        assert not asks_derived_quantity(question), question


def test_the_notice_follows_the_turn() -> None:
    assert build_arithmetic_notice(True) == ARITHMETIC_NOTICE + CALC_TOOLS_NOTICE
    derived = build_arithmetic_notice(False, "tổng số tín chỉ là bao nhiêu?")
    assert derived == DERIVED_QUANTITY_NOTICE + CALC_TOOLS_NOTICE
    assert build_arithmetic_notice(False, "Điều kiện học bổng là gì?") == ""
    assert '"condition": {"compare"' in CALC_TOOLS_NOTICE and "round(7.6; 0.5)" in CALC_TOOLS_NOTICE
