"""Redaction tests against the shared vector file — plan.md "Secret redaction".

`redaction-vectors.json` is vendored by hand for now at contracts/vendor/ (same
pattern as `ssrf-url-vectors.json` — see tests/core/test_ssrf_guard.py) — Task
0.8's `contracts:sync` tool will take over keeping it in sync with backend-java.
Every vector here is also exercised by Java's `SecretRedactorTest` against the
exact same file, so a passing run on both sides is what proves parity.
"""

import json
from pathlib import Path

import pytest

from app.core.redaction import redact

_VECTORS = json.loads(
    (Path(__file__).resolve().parents[2] / "contracts" / "vendor" / "redaction-vectors.json").read_text(
        encoding="utf-8"
    )
)["vectors"]


@pytest.mark.parametrize("vector", _VECTORS, ids=lambda v: v["description"])
def test_redaction_vectors(vector: dict) -> None:
    known_secret = vector.get("knownSecret")
    assert redact(vector["input"], known_secret) == vector["expected"]


def test_none_input_returns_empty_string() -> None:
    assert redact(None) == ""


def test_empty_input_returns_empty_string() -> None:
    assert redact("") == ""


def test_truncates_to_500_chars_when_no_secret_involved() -> None:
    text = "y" * 600
    assert redact(text) == "y" * 500
