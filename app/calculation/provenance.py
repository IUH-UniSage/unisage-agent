"""Deterministic provenance checks for a formula an LLM copied from a regulation.

None of these prove the formula is right - they catch a quote that isn't in
the chunk, coefficients or variables the quote doesn't contain. The semantic
check (an independent verifier LLM) runs after these in the calculation node.
Spec: docs/specs/SPEC-calculation-node.md §2, checks 2, 5 and 6.
"""

import ast
import re
import unicodedata
from decimal import Decimal, InvalidOperation

from app.calculation.expression import RetrievedFormula

# Short function words that say nothing about which quantity a variable is.
_STOPWORDS = frozenset(
    {
        "của", "các", "cho", "theo", "trong", "với", "được", "một", "những", "này",
        "đó", "là", "và", "hoặc", "khi", "thì", "mỗi", "số", "tổng", "giá", "trị",
    }
)  # fmt: skip
_NUMBER = re.compile(r"\d+(?:[.,]\d+)*\s*%?")


def normalize_text(text: str) -> str:
    """NFC, lower-case, single spaces - what the quote check compares."""

    return " ".join(unicodedata.normalize("NFC", text).lower().split())


def _strip_accents(text: str) -> str:
    text = text.replace("đ", "d")
    return "".join(
        char for char in unicodedata.normalize("NFD", text) if unicodedata.category(char) != "Mn"
    )


def quote_is_in_chunk(quote: str, chunk_content: str) -> bool:
    """Check 2: the quote really appears in the chunk (whitespace/case-insensitive)."""

    needle = normalize_text(quote)
    return bool(needle) and needle in normalize_text(chunk_content)


def _numbers_in(text: str) -> set[Decimal]:
    """Every reading of every number in `text`: 20% -> {20, 0.2}; 420.000 -> {420000,
    420}; 6,5 -> {6.5, 65}. Generous on purpose - the check only rejects constants
    that appear in NO reading."""

    found: set[Decimal] = set()
    for match in _NUMBER.finditer(text):
        token = match.group(0).replace(" ", "")
        percent = token.endswith("%")
        digits = token.rstrip("%")
        candidates = {
            digits.replace(",", "."),
            digits.replace(".", "").replace(",", "."),
            digits.replace(",", "").replace(".", ""),
        }
        for candidate in candidates:
            try:
                value = Decimal(candidate)
            except InvalidOperation:
                continue
            found.add(value.normalize())
            if percent:
                found.add((value / 100).normalize())
    return found


def _constants(tree: ast.expr) -> list[Decimal]:
    """Numeric literals in the expression, minus 0/1 and round()'s place count."""

    skip: set[int] = set()
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "round"
            and len(node.args) == 2
        ):
            skip.add(id(node.args[1]))
    values: list[Decimal] = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Constant)
            and id(node) not in skip
            and isinstance(node.value, int | float)
            and not isinstance(node.value, bool)
        ):
            value = Decimal(str(node.value)).normalize()
            if value not in (Decimal(0), Decimal(1)):
                values.append(value)
    return values


def constants_anchored(tree: ast.expr, quote: str) -> list[Decimal]:
    """Check 5: the constants missing from the quote (empty = pass)."""

    available = _numbers_in(quote)
    return [value for value in _constants(tree) if value not in available]


def _content_words(label: str) -> list[str]:
    return [
        word
        for word in re.findall(r"\w+", label.lower())
        if len(word) >= 3 and word not in _STOPWORDS
    ]


def variables_anchored(formula: RetrievedFormula, quote: str) -> list[str]:
    """Check 6: names of variables whose label is not grounded in the quote - at least
    half of the label's content words must appear as whole words in it."""

    quote_words = {_strip_accents(word) for word in re.findall(r"\w+", normalize_text(quote))}
    loose: list[str] = []
    for variable in formula.variables:
        words = [_strip_accents(word) for word in _content_words(variable.label)]
        present = sum(1 for word in words if word in quote_words)
        if not words or present * 2 < len(words):
            loose.append(variable.name)
    return loose
