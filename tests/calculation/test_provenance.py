import ast
from decimal import Decimal

from app.calculation.expression import FormulaVariable, RetrievedFormula
from app.calculation.provenance import constants_anchored, quote_is_in_chunk, variables_anchored

QUOTE = (
    "Học phí học kỳ = số tín chỉ đăng ký \u00d7 đơn giá 420.000 đồng/tín chỉ, "
    "giảm 20% cho sinh viên diện chính sách"
)


def _tree(expression: str) -> ast.expr:
    return ast.parse(expression, mode="eval").body


def test_quote_must_really_be_in_the_chunk() -> None:
    chunk = "Điều 8.  " + QUOTE.upper() + "\nĐiều 9 ..."
    assert quote_is_in_chunk(QUOTE, chunk)
    assert not quote_is_in_chunk("Học phí = số tín chỉ \u00d7 500.000", chunk)
    assert not quote_is_in_chunk("", chunk)


def test_constants_must_come_from_the_quote() -> None:
    assert constants_anchored(_tree("so_tc * 420000 * (1 - 0.2)"), QUOTE) == []
    assert constants_anchored(_tree("so_tc * 420000 * 0.4"), QUOTE) == [Decimal("0.4")]
    # 0, 1 and round()'s place count never need anchoring
    assert constants_anchored(_tree("round(so_tc * 1, 2) + 0"), QUOTE) == []


def test_variables_must_be_named_in_the_quote() -> None:
    formula = RetrievedFormula(
        expression="so_tc * don_gia + phi_bh",
        variables=(
            FormulaVariable("so_tc", "Số tín chỉ đăng ký"),
            FormulaVariable("don_gia", "Đơn giá tín chỉ"),
            FormulaVariable("phi_bh", "Phí bảo hiểm y tế"),
        ),
        result_label="Học phí",
    )
    assert variables_anchored(formula, QUOTE) == ["phi_bh"]
