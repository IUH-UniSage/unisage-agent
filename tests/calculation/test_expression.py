from decimal import Decimal

import pytest

from app.calculation.expression import (
    FormulaRejected,
    FormulaVariable,
    RetrievedFormula,
    evaluate,
    validate,
)
from app.calculation.formulas import TIMES
from app.calculation.result import CalculationInputError


def _formula(expression: str, *names: str, label: str = "Kết quả") -> RetrievedFormula:
    variables = tuple(FormulaVariable(name=name, label=name.upper()) for name in names or ("a",))
    return RetrievedFormula(expression=expression, variables=variables, result_label=label)


ATTACKS = [
    "__import__('os').system('id')",
    "().__class__.__bases__[0]",
    "a.real",
    "a ** 2",
    "9 ** 9 ** 9",
    "[1][0] + a",
    "a if a else 1",
    "'x' * a",
    "True + a",
    "1e400 + a",
    "a % 2",
    "a // 2",
    "a < 2",
    "a and 1",
    "(lambda: a)()",
    "[a for _ in range(9)]",
    "f'{a}'",
    "exec('1') + a",
    "round(a)",
    "round(a, 5)",
    "round(a, n=2)",
    "round(a, True)",
    "min(a)",
    "min(*[a, a])",
    "max(" + ", ".join(["a"] * 11) + ")",
    "a" + " + a" * 30,
    "-" * 11 + "a",  # 12 levels of real nesting (redundant parentheses add no AST node)
    "a + 1234567890123",
    "a + 0.1234567",
    "a +",
    "a = 1",
    "a; 1",
]


@pytest.mark.parametrize("expression", ATTACKS)
def test_attack_strings_are_rejected(expression: str) -> None:
    with pytest.raises(FormulaRejected):
        validate(_formula(expression, "a"))


def test_expression_length_is_checked_before_parsing() -> None:
    with pytest.raises(FormulaRejected, match="quá dài"):
        validate(_formula("a + " * 60 + "a", "a"))


def test_undeclared_variable_is_rejected() -> None:
    with pytest.raises(FormulaRejected, match="không được khai báo"):
        validate(_formula("a + b", "a"))


def test_declared_but_unused_variable_is_rejected() -> None:
    with pytest.raises(FormulaRejected, match="không dùng"):
        validate(_formula("a * 2", "a", "b"))


@pytest.mark.parametrize("name", ["A", "1a", "a-b", "round", "x" * 33])
def test_bad_variable_names_are_rejected(name: str) -> None:
    with pytest.raises(FormulaRejected):
        validate(
            RetrievedFormula(
                expression=name, variables=(FormulaVariable(name, "x"),), result_label="r"
            )
        )


def test_too_many_variables_are_rejected() -> None:
    names = [f"v{i}" for i in range(11)]
    with pytest.raises(FormulaRejected):
        validate(_formula(" + ".join(names), *names))


def test_tuition_formula_is_evaluated_with_steps() -> None:
    formula = RetrievedFormula(
        expression="so_tc * don_gia + phi_khac",
        variables=(
            FormulaVariable("so_tc", "Số tín chỉ đăng ký", "TC", Decimal(1), Decimal(40)),
            FormulaVariable(
                "don_gia", "Đơn giá một tín chỉ", "đồng", Decimal(0), Decimal(10_000_000)
            ),
            FormulaVariable("phi_khac", "Phí khác", "đồng"),
        ),
        result_label="Học phí",
    )
    result = evaluate(formula, {"so_tc": 20, "don_gia": "420000", "phi_khac": 150000})
    assert dict(result.outputs) == {"Học phí": "8550000"}
    assert result.steps[0].symbolic == f"Học phí = so_tc {TIMES} don_gia + phi_khac"
    assert result.steps[0].substituted == f"Học phí = 20 {TIMES} 420000 + 150000"
    assert result.formula_text[1] == "so_tc: Số tín chỉ đăng ký (TC)"


def test_division_at_root_shows_numerator_and_denominator() -> None:
    result = evaluate(_formula("(a + b) / c", "a", "b", "c"), {"a": 10, "b": 12.4, "c": 3})
    assert [step.label for step in result.steps] == ["Thay số", "Tử số", "Mẫu số", "Kết quả"]
    assert result.steps[0].substituted == "Kết quả = (10 + 12.4) / 3"
    assert result.steps[-1].substituted == "Kết quả ≈ 7.47"


def test_parentheses_are_kept_where_needed() -> None:
    result = evaluate(_formula("a - (b - c)", "a", "b", "c"), {"a": 10, "b": 4, "c": 1})
    assert result.steps[0].symbolic == "Kết quả = a - (b - c)"
    assert dict(result.outputs)["Kết quả"] == "7"


def test_round_min_max_use_half_up() -> None:
    result = evaluate(
        _formula("round(a, 1) + min(b, 2) + max(b, 2)", "a", "b"), {"a": "7.45", "b": 5}
    )
    assert result.steps[-1].value == Decimal("7.5") + 2 + 5


def test_negative_literals_are_allowed() -> None:
    assert evaluate(_formula("-a + 3", "a"), {"a": 1}).steps[-1].value == 2


def test_division_by_zero_names_the_variable() -> None:
    with pytest.raises(CalculationInputError) as error:
        evaluate(_formula("a / b", "a", "b"), {"a": 1, "b": 0})
    assert error.value.errors[0].field == "b"
    assert "mẫu số bằng 0" in error.value.errors[0].reason


def test_huge_intermediate_value_is_rejected() -> None:
    with pytest.raises(CalculationInputError, match="quá lớn"):
        evaluate(_formula("a * 1000000000 * 1000000000", "a"), {"a": 5})


def test_missing_and_out_of_range_values_are_all_reported() -> None:
    formula = RetrievedFormula(
        expression="a + b + c",
        variables=(
            FormulaVariable("a", "A", max=Decimal(10)),
            FormulaVariable("b", "B"),
            FormulaVariable("c", "C"),
        ),
        result_label="r",
    )
    with pytest.raises(CalculationInputError) as error:
        evaluate(formula, {"a": 11, "c": "x"})
    assert {item.field for item in error.value.errors} == {"a", "b", "c"}
