"""Pure arithmetic (SPEC-calc-engine, số học thuần): whitelisted expressions parsed and
evaluated by code, operands checked against the texts that state them."""

from decimal import Decimal

import pytest

from app.calculation.arithmetic import (
    ArithmeticError,
    ArithmeticRequest,
    arithmetic_result,
    display,
    evaluate,
    parse_request,
)
from app.calculation.render import render_markdown


@pytest.mark.parametrize(
    ("expression", "value", "shown"),
    [
        ("18 + 15", "33", "18 + 15"),
        ("1,5 + 2", "3.5", "1.5 + 2"),
        ("250000 + 30000", "280000", "250000 + 30000"),
        ("190.000 + 240.000 + 170.000", "600000", "190000 + 240000 + 170000"),
        ("3 × 4 − 2", "10", "3 × 4 - 2"),  # noqa: RUF001
        ("(13 + 10/60) * 2 + 11 + 10/60", "37.5", "(13 + 10 ÷ 60) × 2 + 11 + 10 ÷ 60"),  # noqa: RUF001
        ("-5 + 8", "3", "- 5 + 8"),
        ("100 - (6 + 94)", "0", "100 - (6 + 94)"),
    ],
)
def test_whitelisted_expressions_evaluate_in_decimal(
    expression: str, value: str, shown: str
) -> None:
    evaluation = evaluate(expression)
    assert evaluation.value == Decimal(value)
    assert evaluation.expression == shown


def test_operands_are_collected_in_order() -> None:
    assert evaluate("1.200.000 - 865 * 2").operands == (
        Decimal(1200000),
        Decimal(865),
        Decimal(2),
    )


@pytest.mark.parametrize(
    "expression",
    [
        "10 / 0",
        "2 ** 3",
        "__import__('os')",
        "sum(1, 2)",
        "7 +",
        "(1 + 2",
        "1 + 2)",
        "1.2.3,4 + 1",
        "a + 1",
        "",
        "1 + " * 30 + "1",
        "(" * 9 + "1" + ")" * 9,
    ],
)
def test_anything_outside_the_whitelist_is_refused(expression: str) -> None:
    with pytest.raises(ArithmeticError):
        evaluate(expression)


def test_parse_request_keeps_only_well_formed_blocks() -> None:
    request = parse_request(
        {
            "expression": " 18+15 ",
            "label": "**Tổng** tín chỉ",
            "unit": "tín chỉ",
            "sources": [1, "x", 3],
        }
    )
    assert request == ArithmeticRequest("18+15", "Tổng tín chỉ", "tín chỉ", (1, 3))
    assert parse_request({"expression": "1+1"}) is None
    assert parse_request({"label": "x", "expression": ""}) is None
    assert parse_request("1+1") is None


def test_display_formats_money_and_hours() -> None:
    assert display(Decimal(600000), "đồng") == "600.000 đồng"
    assert display(Decimal("37.5"), "giờ") == "37.5 giờ (= 37 giờ 30 phút)"
    assert display(Decimal(33), "tín chỉ") == "33 tín chỉ"
    assert display(Decimal("2.3333"), None) == "≈ 2.33"


def test_arithmetic_result_renders_like_a_registry_formula() -> None:
    request = ArithmeticRequest("190000 + 240000 + 170000", "Tổng tiền phải trả", "đồng", (1,))
    source = "số liệu trích từ tài liệu [1]"
    result = arithmetic_result(request, evaluate(request.expression), source=source)
    rendered = render_markdown(result)
    assert rendered.startswith("**Phép tính số học: Tổng tiền phải trả**")
    assert "- _Phiên bản công thức 1 · nguồn: số liệu trích từ tài liệu [1]_" in rendered
    assert "1. Tổng tiền phải trả: 190000 + 240000 + 170000 = 600000" in rendered
    assert rendered.endswith("Kết quả: Tổng tiền phải trả: **600.000 đồng**")
    assert result.values == {"ket_qua": Decimal(600000)}
