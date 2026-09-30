import re
from pathlib import Path

import pytest

from evals.questions.build import build, evidence_found, is_visible
from evals.questions.plan import excerpt, proportional_quota


def test_proportional_quota_respects_minimum_size_and_total():
    quota = proportional_quota({"A": 100, "B": 10, "C": 1, "D": 3}, total=20)
    assert sum(quota.values()) == 20
    assert quota["C"] == 1  # never above the department's size
    assert quota["D"] >= 2
    assert quota["A"] > quota["B"]


def test_proportional_quota_when_total_exceeds_supply():
    assert proportional_quota({"A": 2, "B": 1}, total=10) == {"A": 2, "B": 1}


def test_excerpt_centres_on_focus_and_is_bounded():
    text = "mở đầu " * 2000 + "Học phí là 500.000 đồng/tín chỉ" + " kết thúc" * 2000
    result = excerpt(text, re.compile("học phí", re.IGNORECASE))
    assert "Học phí là 500.000 đồng/tín chỉ" in result
    assert len(result) <= 5000


def test_evidence_found_ignores_whitespace_and_case():
    assert evidence_found("Thời hạn  nộp\nhồ sơ", "... thời hạn nộp hồ sơ là 30/10 ...")
    assert not evidence_found("không có", "thời hạn nộp hồ sơ")
    assert not evidence_found("  ", "anything")
    assert evidence_found("Kiểm toán", "Kie\u0302\u0309m toa\u0301n cơ bản")


@pytest.mark.parametrize(
    ("asker", "visible"),
    [
        ([{"department_id": "D", "access_level": 3}], True),
        ([{"department_id": "D", "access_level": 2}], False),
        ([{"department_id": "E", "access_level": 5}], False),
        ([{"department_id": "*", "access_level": 5}], True),
        ([], False),
    ],
)
def test_is_visible_matches_qdrant_rule(asker, visible):
    assert is_visible(asker, "D", 3) is visible


def test_public_documents_are_visible_to_everyone():
    assert is_visible([], "D", None)


def _slot(tmp_path: Path, slot_id: str, category: str, **extra: str) -> dict[str, str]:
    path = tmp_path / f"{slot_id}.txt"
    path.write_text("Sinh viên nộp hồ sơ trước ngày 30/10/2026 tại phòng A1.", encoding="utf-8")
    return {
        "slot_id": slot_id,
        "category": category,
        "expected_intent": "academic_advisory",
        "department_id": "D",
        "file_id": "f1" if category != "off_topic" else "",
        "is_public": "true",
        "access_level": "",
        "excerpt_path": str(path) if category != "off_topic" else "",
        **extra,
    }


def test_build_expands_access_personas_and_checks_evidence(tmp_path):
    slots = [
        _slot(tmp_path, "acc-001", "access", is_public="false", access_level="3"),
        _slot(tmp_path, "nor-001", "normal"),
        _slot(tmp_path, "nor-002", "normal"),
        _slot(tmp_path, "off-001", "off_topic"),
        _slot(tmp_path, "nor-003", "normal"),
    ]
    good = {
        "question": "Hạn nộp hồ sơ?",
        "expected_answer": "30/10/2026",
        "evidence": "trước ngày 30/10/2026",
    }
    answers = {
        "acc-001": {"slot_id": "acc-001", **good},
        "nor-001": {"slot_id": "nor-001", **good},
        "nor-002": {"slot_id": "nor-002", **good, "evidence": "invented quote"},
        "off-001": {
            "slot_id": "off-001",
            "question": "Thời tiết?",
            "expected_answer": "Từ chối",
            "evidence": "",
        },
        "nor-003": {"slot_id": "nor-003", "skip": True, "reason": "blank form"},
    }
    rows, problems = build(slots, answers, ["D", "E"], seed=1, work_dir=tmp_path)

    access = [row for row in rows if row["category"] == "access"]
    assert [(r["persona"], r["expect_visible"]) for r in access] == [
        ("same_department_exact_level", True),
        ("same_department_one_level_below", False),
        ("other_department_max_level", False),
        ("wildcard_max_level", True),
    ]
    assert access[2]["asker"]["department_access"][0]["department_id"] == "E"
    normal = {row["slot_id"]: row for row in rows if row["category"] == "normal"}
    assert normal["nor-001"]["evidence_found"] is True
    assert normal["nor-001"]["asker"] == {"department_access": []}
    assert normal["nor-002"]["evidence_found"] is False
    off = next(row for row in rows if row["category"] == "off_topic")
    assert off["expected_doc_ids"] == [] and off["expect_visible"] is None
    assert problems == {"evidence_not_in_excerpt": 1, "skipped_by_author": 1}
    assert [row["id"] for row in rows] == [f"q{i:04d}" for i in range(1, len(rows) + 1)]
    assert next(iter(rows[0])) == "id"
