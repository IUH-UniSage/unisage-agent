from typing import Any

import pytest

from evals.metrics import (
    can_see,
    leaked_documents,
    numbers_in,
    numbers_match,
    outcome_of,
    recall_at_k,
    score_row,
    summarize,
    wilson,
)

DEPT = "uuid-khoa-luat"
PRIVATE_L3 = {"document_id": "doc-p", "department": DEPT, "access_level": 3, "is_public": False}
PUBLIC = {"document_id": "doc-pub", "department": "other", "access_level": 0, "is_public": True}


def _row(**extra: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "id": "q1",
        "variant": "exact",
        "category": "normal",
        "nodes": ["03_MessageClassificationNode", "08_RetrievalFilteringNode"],
        "context": [PUBLIC],
        "retrievals": [{"query": "q", "chunks": [PUBLIC]}],
        "citations": [],
        "access": [],
        "response": "Có, theo quy chế…",
        "expected_document_ids": ["doc-pub"],
        "total_ms": 1000.0,
    }
    row.update(extra)
    return row


@pytest.mark.parametrize(
    ("access", "visible"),
    [
        ([{"department_id": DEPT, "access_level": 3}], True),  # exact level
        ([{"department_id": DEPT, "access_level": 2}], False),  # one level short
        ([{"department_id": "other", "access_level": 5}], False),  # other department
        ([{"department_id": "*", "access_level": 5}], True),  # wildcard
        ([], False),  # guest
    ],
)
def test_can_see_mirrors_the_qdrant_filter(access: list[dict[str, Any]], visible: bool) -> None:
    assert can_see(PRIVATE_L3, access) is visible
    assert can_see(PUBLIC, access) is True


def test_leak_is_found_in_retrieval_context_or_citations() -> None:
    guest_sees_private = _row(
        context=[PUBLIC], retrievals=[{"query": "q", "chunks": [PRIVATE_L3]}], access=[]
    )
    assert leaked_documents(guest_sees_private) == ["doc-p"]
    allowed = _row(
        context=[PRIVATE_L3],
        retrievals=[{"query": "q", "chunks": [PRIVATE_L3]}],
        access=[{"department_id": DEPT, "access_level": 3}],
    )
    assert leaked_documents(allowed) == []


@pytest.mark.parametrize(
    ("row", "expected"),
    [
        (_row(), ["rag"]),
        (_row(nodes=["01_GreetingDetectionNode"], context=[]), ["social"]),
        (_row(nodes=["03_MessageClassificationNode", "05_OffTopicRejectNode"]), ["off_topic"]),
        (_row(used_ticket_fallback=True, context=[]), ["ticket"]),
        (_row(nodes=["07_CalculationNode"], context=[]), ["calculation"]),
        (_row(error="TimeoutError"), ["error"]),
    ],
)
def test_outcome_of(row: dict[str, Any], expected: list[str]) -> None:
    assert outcome_of(row) == expected


def test_recall_uses_the_first_five_context_chunks() -> None:
    late = [dict(PUBLIC, document_id=f"d{i}") for i in range(5)] + [PUBLIC]
    assert recall_at_k(_row()) is True
    assert recall_at_k(_row(context=late)) is False
    assert recall_at_k(_row(expected_document_ids=[])) is None


def test_numbers_accept_vietnamese_and_english_separators() -> None:
    assert {1200000.0, 7.5, 12.0} <= numbers_in("Học phí 1.200.000đ, điểm 7,5, tổng 12 tín chỉ")
    assert numbers_match([600000, 170000], "trả 600.000đ, giảm 170 000 đ")
    assert numbers_match([25.4], "ĐXT = 25,4 điểm")
    assert not numbers_match([9.5], "cần 9,1 điểm")


def test_score_row_per_category() -> None:
    assert score_row(_row())["pass"] is None  # normal: correctness left to the judge
    assert score_row(_row(context=[dict(PUBLIC, document_id="x")]))["pass"] is False  # recall miss
    hidden = _row(
        category="access",
        expect_visible=False,
        used_ticket_fallback=True,
        context=[],
        retrievals=[{"query": "q", "chunks": [PRIVATE_L3]}],
    )
    assert score_row(hidden)["pass"] is False  # ticket, but the private chunk leaked
    refused = _row(category="unanswerable", response="Mình không tìm thấy thông tin này.")
    assert score_row(refused)["pass"] is True
    calc = _row(
        category="calculation",
        nodes=["07_CalculationNode"],
        context=[],
        expected_numbers=[12],
        response="Tổng 12 tín chỉ.",
    )
    assert score_row(calc)["pass"] is True
    assert score_row(dict(calc, asked_back=True))["pass"] is False
    off = _row(category="off_topic", nodes=["05_OffTopicRejectNode"], retrievals=[], context=[])
    assert score_row(off)["pass"] is True


def test_summary_counts_and_wilson() -> None:
    rows = [dict(_row(id=f"q{i}"), checks=score_row(_row())) for i in range(3)]
    summary = summarize(rows)
    assert summary["table"]["exact/normal"]["recall_hit"] == 3
    assert summary["latency_ms"]["n"] == 3
    rate, low, high = wilson(0, 30) or (0, 0, 0)
    assert rate == 0 and low == 0 and 0.10 < high < 0.12
