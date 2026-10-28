"""`calc_request` blocks of the advisory answer (SPEC-calculation-node §2): computed
by code only when the expression is plain arithmetic over figures the question or the
cited documents state."""

from decimal import Decimal

from app.graph.calc_requests import REJECTED_LINE, arithmetic_blocks, missing_operands
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
