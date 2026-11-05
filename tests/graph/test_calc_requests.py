"""`calc_request` blocks of the advisory answer (SPEC-calculation-node §2): computed
by code only when the expression is plain arithmetic over figures the question or the
cited documents state."""

from decimal import Decimal

from app.graph.calc_requests import (
    REJECTED_LINE,
    CalcBlocks,
    arithmetic_blocks,
    missing_operands,
)
from app.schemas.retrieval import RetrievedChunk

_BOOKS = RetrievedChunk(
    chunk_id="s1",
    content="Life A2-B1 250.000đ (giảm 60.000đ); Oxford 320.000đ (giảm 80.000đ); Collins 200.000đ",
    source="CDIMEX.pdf",
    score=0.9,
)
_OTHER = RetrievedChunk(chunk_id="s2", content="Lịch thi học kỳ", source="Lich.pdf", score=0.5)


def test_a_request_over_cited_figures_is_computed_and_rendered() -> None:
    text, warnings = arithmetic_blocks(
        [
            {
                "type": "calc_request",
                "label": "Tổng tiền được giảm",
                "expression": "60000 + 80000",
                "unit": "đồng",
                "sources": [1],
            }
        ],
        question="mua 3 cuốn thì được giảm bao nhiêu?",
        chunks=[_BOOKS, _OTHER],
    )
    assert warnings == []
    assert text.startswith("\n\n**Phép tính số học: Tổng tiền được giảm**")
    assert "nguồn: số liệu trích từ tài liệu [1]" in text
    assert text.rstrip().endswith("Kết quả: Tổng tiền được giảm: **140.000 đồng**")


def test_an_operand_no_cited_document_states_is_refused() -> None:
    text, warnings = arithmetic_blocks(
        [
            {
                "type": "calc_request",
                "label": "Tổng tiền",
                "expression": "250000 + 999",
                "unit": "đồng",
            }
        ],
        question="tổng bao nhiêu?",
        chunks=[_BOOKS],
    )
    assert text == "\n\n" + REJECTED_LINE.format(label="Tổng tiền")
    [warning] = warnings
    assert warning.code == "CALC_REQUEST_REJECTED" and "999" in warning.message


def test_numbers_from_the_question_count_as_evidence() -> None:
    text, warnings = arithmetic_blocks(
        [
            {
                "type": "calc_request",
                "label": "Tổng tín chỉ",
                "expression": "18 + 15",
                "unit": "tín chỉ",
            }
        ],
        question="kỳ này 18 tín, kỳ sau 15 tín, tổng bao nhiêu?",
        chunks=[],
    )
    assert warnings == [] and "**33 tín chỉ**" in text


def test_bad_expressions_and_malformed_blocks_are_reported_not_computed() -> None:
    text, warnings = arithmetic_blocks(
        [
            {"type": "calc_request", "label": "Chia", "expression": "10 / 0"},
            {"type": "calc_request", "expression": "1 + 1"},
            {"type": "calc_request", "label": "Lũy thừa", "expression": "2 ** 3"},
        ],
        question="?",
        chunks=[],
    )
    assert text.count(REJECTED_LINE.format(label="Chia")) == 1
    assert [w.code for w in warnings] == ["CALC_REQUEST_REJECTED"] * 3


def test_at_most_three_requests_are_honoured() -> None:
    requests = [
        {"type": "calc_request", "label": f"T{i}", "expression": "18 + 15", "unit": "tín chỉ"}
        for i in range(5)
    ]
    text, _ = arithmetic_blocks(requests, question="18 và 15", chunks=[])
    assert text.count("**Phép tính số học") == 3


def test_missing_operands_compares_with_every_way_a_number_is_written() -> None:
    texts = ["Life A2-B1 250.000đ, giảm 60.000đ", "còn 1.200.000 đồng"]
    assert missing_operands([Decimal(250000), Decimal(60000), Decimal(1200000)], texts) == []
    assert missing_operands([Decimal(190000)], texts) == [Decimal(190000)]
    assert missing_operands([Decimal("7.5")], ["điểm 7,5 môn Toán"]) == []


_APPEALS = RetrievedChunk(
    chunk_id="s3",
    content=(
        "Học kỳ I 2020-2021: phúc khảo trắc nghiệm 16 đơn, tự luận 358 đơn; thay đổi điểm 45 đơn."
    ),
    source="PhucKhao.pdf",
    score=0.9,
)


def test_a_request_may_use_the_result_of_an_earlier_one() -> None:
    # Live 28/10/2026 (d0317): "374 - 45" was refused because 374 is the first block's
    # result, not a figure of the document.
    from app.graph.calc_requests import calc_blocks

    blocks = calc_blocks(
        [
            {"label": "Tổng số đơn", "expression": "16 + 358", "unit": "đơn", "sources": [1]},
            {"label": "Đơn không đổi điểm", "expression": "374 - 45", "unit": "đơn"},
        ],
        question="tổng bao nhiêu đơn và bao nhiêu đơn không đổi điểm?",
        chunks=[_APPEALS],
    )
    assert blocks.warnings == []
    assert blocks.values == [Decimal(374), Decimal(329)]
    assert "Đơn không đổi điểm: **329 đơn**" in blocks.markdown
    # Only a result code computed counts: an invented subtotal is still refused.
    refused = calc_blocks(
        [{"label": "Đơn không đổi điểm", "expression": "374 - 45", "unit": "đơn"}],
        question="bao nhiêu đơn không đổi điểm?",
        chunks=[_APPEALS],
    )
    assert refused.values == [] and refused.warnings[0].code == "CALC_REQUEST_REJECTED"


def _chunk(content: str) -> RetrievedChunk:
    return RetrievedChunk(chunk_id="e", content=content, source="Doc.pdf", score=0.9)


def _one(
    expression: str, question: str, content: str, unit: str = "", answer: str = ""
) -> CalcBlocks:
    from app.graph.calc_requests import calc_blocks

    request = {"label": "Kết quả", "expression": expression, "unit": unit, "sources": [1]}
    return calc_blocks([request], question=question, chunks=[_chunk(content)], answer=answer)


def test_operands_written_with_units_scales_and_percentages_are_evidence() -> None:
    # Eval 03/11/2026: each of these was refused as "chưa tính được".
    # q0076: "3,5cm" glued to its unit.
    margins = _one("3.5 + 2", "lề trái và lề phải tổng cộng?", "lề trái 3,5cm; lề phải 2cm", "cm")
    assert margins.warnings == [] and margins.values == [Decimal("5.5")]
    # q0075: "5 triệu" states 5000000.
    income = _one(
        "6 * 5000000",
        "tổng thu nhập tối thiểu của cả nhóm?",
        "thu nhập ( tối thiểu 5 triệu \u2013 7 triệu); nhóm từ 4 \u2013 6 thành viên",
        "đồng",
    )
    assert income.warnings == [] and income.values == [Decimal(30000000)]
    # q0073 / q0059: "80%" states 0.8, "(50%)" states 0.5.
    visits = _one("15 * 0.8", "hẹn gặp 15 lần", "gặp GVHD tối thiểu 80% số lần hẹn", "lần")
    assert visits.warnings == [] and visits.values == [Decimal(12)]
    score = _one(
        "6.0 * 0.5 + 4.5 * 0.5",
        "đơn vị thực tập chấm 6.0 và đơn vị đào tạo chấm 4.5",
        "Đơn vị thực tập (50%), Đơn vị đào tạo (50%).",
    )
    assert score.warnings == [] and score.values == [Decimal("5.25")]


def test_the_whole_part_of_a_stated_score_is_evidence() -> None:
    # Eval 03/11/2026 (d0314): "7.0 + 0.5" (phần nguyên + phần lẻ làm tròn) for 7,6.
    rule = "phần lẻ từ 0,25 đến dưới 0,75 thì làm tròn thành 0,5"
    blocks = _one("7.0 + 0.5", "Các điểm 6,2; 7,6 và 8,8 sau khi làm tròn?", rule, "điểm")
    assert blocks.warnings == [] and blocks.values == [Decimal("7.5")]
    # Not any whole number: 9 is no stated score's whole part.
    assert _one("9.0 + 0.5", "Điểm 7,6?", rule).warnings


def test_a_cohort_code_is_not_an_operand() -> None:
    # Eval 03/11/2026 (q0071): "khóa 14" is the cohort, not a number of credits.
    blocks = _one("14 + 4", "ngành Cơ điện tử khóa 14 học bao nhiêu tín chỉ?", "GDQP 1: 4 tín chỉ")
    assert blocks.values == [] and blocks.warnings


def test_dates_the_documents_state_can_be_counted_and_shifted() -> None:
    # Eval 03/11/2026 (d0329, q0055, d0335): day counts and "20 ngày trước" were written
    # as day-of-month arithmetic ("20 - 13 + 1", "2018/06/20 - 20") and refused.
    stay = _one(
        "20/08/2026 - 13/08/2026 + 1",
        "kéo dài bao nhiêu ngày (tính cả ngày bắt đầu và ngày kết thúc)?",
        "Thời gian đăng ký và đóng phí đợt 1 từ ngày 13/08/2026 đến 20/08/2026.",
        "ngày",
    )
    assert stay.warnings == [] and stay.values == [Decimal(8)]
    assert "**8 ngày**" in stay.markdown
    # Across a month end, where day-of-month arithmetic is wrong.
    across = _one("05/04/2019 - 25/03/2019 + 1", "?", "nộp từ 25/03/2019 \u2013 05/04/2019", "ngày")
    assert across.values == [Decimal(12)]
    deadline = _one(
        "20/6/2018 - 20",
        "Nếu bảo vệ ngày 20/6/2018 thì chậm nhất công bố ngày nào?",
        "công bố chậm nhất 20 ngày trước ngày bảo vệ",
    )
    assert deadline.warnings == [] and "**31/05/2018**" in deadline.markdown
    assert deadline.values == []  # a date is not a figure for the grounding check
    # Year-first, as the model also writes it.
    iso = _one("2018/06/20 - 20", "bảo vệ ngày 20/6/2018?", "chậm nhất 20 ngày trước")
    assert "**31/05/2018**" in iso.markdown
    # A date no text states is refused, and so is arithmetic dates do not have.
    assert _one("21/08/2026 - 13/08/2026", "?", "từ 13/08/2026 đến 20/08/2026").warnings
    assert _one("20/08/2026 * 2", "?", "từ 13/08/2026 đến 20/08/2026").warnings


def test_a_refusal_is_not_appended_when_the_answer_already_states_the_grounded_result() -> None:
    # Eval 03/11/2026 (q0055): the answer said "16 ngày (01/04/2019 đến 16/04/2019)" and
    # "Mình chưa tính được ..." contradicted it.
    content = "| 3 | Học viên nộp đề cương | 01/04/2019 \u2013 16/04/2019 | Nộp cho Khoa |"
    answer = "Học viên có tổng cộng **16** ngày, từ 01/04/2019 đến 16/04/2019 [1]."
    blocks = _one("16 - 1 + 1", "tổng cộng bao nhiêu ngày?", content, "ngày", answer=answer)
    assert blocks.markdown == ""
    [warning] = blocks.warnings
    assert warning.code == "CALC_REQUEST_REJECTED"
    # Without the answer stating it - or stating a figure no document grounds - the
    # refusal stays.
    assert REJECTED_LINE.format(label="Kết quả") in _one("16 - 1 + 1", "?", content).markdown
    invented = _one("250000 + 999", "?", "giá 250.000đ", "đồng", answer="Tổng 250.999 đồng [1].")
    assert REJECTED_LINE.format(label="Kết quả") in invented.markdown


def test_a_both_ends_one_is_dropped_from_a_date_shift() -> None:
    # Eval 04/11/2026 (d0335): "20/06/2018 - 20 + 1" gave 01/06/2018; 20 days before the
    # defence is 31/05/2018 - the "+ 1" belongs to day counts only.
    shifted = _one(
        "20/06/2018 - 20 + 1",
        "Nếu bảo vệ ngày 20/6/2018 thì chậm nhất công bố ngày nào?",
        "công bố chậm nhất 20 ngày trước ngày bảo vệ",
    )
    assert shifted.warnings == [] and "**31/05/2018**" in shifted.markdown
    assert "không cộng hay trừ thêm 1" in shifted.markdown
    # A day count keeps its "+ 1".
    stay = _one("20/08/2026 - 13/08/2026 + 1", "?", "từ 13/08/2026 đến 20/08/2026", "ngày")
    assert stay.values == [Decimal(8)] and "không cộng hay trừ thêm 1" not in stay.markdown


def test_rounding_to_a_step_the_question_states_is_computed() -> None:
    # Eval 04/11/2026 (d0314): the 0.5-point rounding rule given in the question.
    question = "phần lẻ dưới 0,25 → 0,0; từ 0,25 đến dưới 0,75 → 0,5. Các điểm 6,2; 7,6 và 8,8?"
    blocks = _one("round(7.6; 0.5)", question, "Sổ tay sinh viên")
    assert blocks.warnings == [] and blocks.values == [Decimal("7.5")]
    assert "làm tròn(7.6; 0.5) = 7.5" in blocks.markdown


def test_a_condition_with_a_grounded_threshold_adds_the_verdict() -> None:
    # Eval 04/11/2026 (q0059): 5.25 was computed but the answer never said "đạt".
    from app.graph.calc_requests import calc_blocks

    content = "Điểm đạt của học phần TTDN là điểm tổng kết từ 5.0 trở lên (theo thang điểm 10)."
    request = {
        "label": "Điểm tổng kết",
        "expression": "6 * 0.5 + 4.5 * 0.5",
        "unit": "điểm",
        "sources": [1],
        "condition": {"compare": ">=", "value": 5.0, "true": "Đạt", "false": "Không đạt"},
    }
    question = (
        "đơn vị thực tập chấm 6.0, đơn vị đào tạo chấm 4.5, phân bổ 50% mỗi bên: có đạt không?"
    )
    blocks = calc_blocks([request], question=question, chunks=[_chunk(content)])
    assert blocks.warnings == []
    assert "kết luận: 5.25 ≥ 5 → **Đạt**" in blocks.markdown
    failing = {**request, "expression": "4 * 0.5 + 4.5 * 0.5"}
    question_low = "chấm 4 và 4.5, phân bổ 50%: có đạt không?"
    low = calc_blocks([failing], question=question_low, chunks=[_chunk(content)])
    assert "4.25 < 5 → **Không đạt**" in low.markdown
    # A threshold nothing states: the result stays, the verdict is dropped with a warning.
    made_up = {**request, "condition": {**request["condition"], "value": 6.5}}  # type: ignore[dict-item]
    blocks = calc_blocks([made_up], question=question, chunks=[_chunk(content)])
    assert "**5.25 điểm**" in blocks.markdown and "kết luận" not in blocks.markdown
    assert len(blocks.warnings) == 1


def test_a_bare_number_is_not_a_calculation() -> None:
    # Eval 04/11/2026 (d0314): the model sent its own rounded "6" and the block read "6 = 6".
    blocks = _one("6", "6,2 làm tròn?", "điểm 6")
    assert blocks.markdown == "" and blocks.values == []
    [warning] = blocks.warnings
    assert "chỉ là một số" in warning.message
