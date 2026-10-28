from typing import Any

import pytest

from evals.run.metrics import (
    can_see,
    leaked_documents,
    numbers_from_answer,
    numbers_in,
    numbers_match,
    outcome_of,
    recall_at_k,
    score_row,
    summarize,
    web_scope_ok,
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


def test_recall_skips_hidden_personas_and_uses_retrieval_for_calculation() -> None:
    assert recall_at_k(_row(expect_visible=False)) is None
    calculation = _row(
        nodes=["07_CalculationNode"],
        context=[],
        retrievals=[{"query": "q", "chunks": [PUBLIC]}],
    )
    assert recall_at_k(calculation) is True


def test_calculation_falls_back_to_the_number_after_the_last_equals() -> None:
    assert numbers_from_answer("12 tín chỉ (3 + 4 + 2 + 3 = 12).") == [12.0]
    assert numbers_from_answer("ĐXT = 25,4 điểm") == [25.4]
    assert numbers_from_answer("Không có phép tính") == []
    calc = _row(
        category="calculation",
        nodes=["07_CalculationNode"],
        context=[],
        expected_answer="total = 60 (num1 = 25, num2 = 35; total = 25 + 35 = 60).",
        response="Giá trị in ra là 60.",
    )
    checks = score_row(calc)
    assert checks["pass"] is True and checks["numbers_source"] == "answer"


def test_academic_calculation_label_also_accepts_a_rag_answer() -> None:
    row = _row(
        category="access",
        expect_visible=True,
        expected_intent="academic_calculation",
        expected_answer="mỗi học phần 3 tín chỉ nên tổng là 4 x 3 = 12",
        response="Bạn học 4 học phần, tổng 12 tín chỉ [1].",
    )
    assert score_row(row)["pass"] is True
    assert score_row(dict(row, category="calculation"))["pass"] is False  # must take calculation


def test_hidden_access_passes_when_neither_id_nor_title_reaches_the_context() -> None:
    hidden: dict[str, Any] = {
        "category": "access",
        "expect_visible": False,
        "expected_document_ids": ["doc-p"],
        "expected_titles": ["Phụ lục 1/2026: học phần bổ sung"],
        "retrievals": [],
    }
    answered_from_other_doc = _row(**hidden, context=[dict(PUBLIC, title="Quy chế khác")])
    assert score_row(answered_from_other_doc)["pass"] is True  # an answer is fine

    by_id = _row(**hidden, context=[dict(PUBLIC, document_id="doc-p", title="x")])
    assert score_row(by_id)["pass"] is False

    # Same document ingested again under another id: caught by its title (as the chip shows it).
    by_title = _row(**hidden, context=[dict(PUBLIC, title="Phụ lục 1-2026: học phần bổ sung")])
    assert score_row(by_title)["hidden_document_used"] is True
    assert score_row(by_title)["pass"] is False

    cited = _row(**hidden, context=[], citations=[{"documentId": "doc-p", "title": "x"}])
    assert score_row(cited)["pass"] is False


# --- web search -------------------------------------------------------------------------------

_FEE_PAGE = {"url": "https://ptckt.iuh.edu.vn/qd-hoc-phi.pdf", "title": "QĐ học phí"}
_MASTER_PAGE = {
    "url": "https://faet.iuh.edu.vn/news.html@detail@1@Thong-bao-tuyen-sinh-Thac-si",
    "title": "Tuyển sinh thạc sĩ",
}


def _web_row(**extra: Any) -> dict[str, Any]:
    return _row(category="web_search", web="on", **extra)


def test_a_linked_official_page_is_a_web_source_outcome() -> None:
    row = _web_row(used_ticket_fallback=True, context=[], web_sources=[_FEE_PAGE])

    assert outcome_of(row) == ["ticket", "web_source"]
    assert score_row(row)["pass"] is True


def test_older_web_questions_pass_on_an_answer_from_the_web() -> None:
    row = _web_row(
        used_web_search=True,
        context=[],
        citations=[{"sourceType": "WEB"}],
        web_results=[_FEE_PAGE],
    )

    assert score_row(row)["pass"] is True
    assert score_row(_web_row(used_ticket_fallback=True, context=[]))["pass"] is False


def test_a_page_of_the_wrong_scope_fails() -> None:
    row = _web_row(
        used_web_search=True,
        context=[],
        citations=[{"sourceType": "WEB"}],
        web_results=[_MASTER_PAGE],
        expected_web_outcome="web_any",
        forbidden_source_patterns=["thac-si", "thạc sĩ"],
    )

    checks = score_row(row)
    assert checks["web_scope_ok"] is False
    assert checks["pass"] is False


def test_pages_must_be_on_the_expected_sites() -> None:
    on_site = _web_row(web_sources=[_FEE_PAGE], expected_source_hosts=["ptckt.iuh.edu.vn"])
    off_site = _web_row(web_sources=[_MASTER_PAGE], expected_source_hosts=["ptckt.iuh.edu.vn"])
    parent = _web_row(web_sources=[_MASTER_PAGE], expected_source_hosts=["iuh.edu.vn"])

    assert web_scope_ok(on_site) is True
    assert web_scope_ok(off_site) is False
    assert web_scope_ok(parent) is True
    assert web_scope_ok(_web_row()) is None


def test_a_knowledge_base_question_fails_when_the_web_is_used() -> None:
    answered_by_kb = _web_row(expected_web_outcome="kb")
    used_web = _web_row(
        expected_web_outcome="kb", used_ticket_fallback=True, context=[], web_sources=[_FEE_PAGE]
    )

    assert score_row(answered_by_kb)["pass"] is True
    assert score_row(used_web)["pass"] is False


def test_summary_counts_web_use_and_cache_hits() -> None:
    rows = [
        _web_row(
            web_queries=["a", "b"],
            web_cache_hits=1,
            used_ticket_fallback=True,
            context=[],
            web_sources=[_FEE_PAGE],
        ),
        _web_row(web_queries=["c"], web_cache_hits=0),
    ]
    for row in rows:
        row["checks"] = score_row(row)

    assert summarize(rows)["web"] == {
        "searches": 3,
        "cache_hits": 1,
        "answered_from_web": 0,
        "linked_sources": 1,
        "wrong_scope": 0,
    }


def _unanswerable(kind: str, response: str, **extra) -> dict:
    row = {
        "id": "u1",
        "category": "unanswerable",
        "unanswerable_kind": kind,
        "variant": "exact",
        "response": response,
        "nodes": ["08_RetrievalFilteringNode", "10_GenerationSynthesisNode"],
        "retrievals": [{"query": "q", "chunks": []}],
        "citations": [],
        "warning_codes": [],
        "expected_doc_ids": [],
        "expected_document_ids": [],
    }
    row.update(extra)
    return row


def test_unanswerable_kinds_pass_on_a_stated_limit_or_a_question_back() -> None:
    from evals.run.metrics import score_row

    refused = score_row(
        _unanswerable("not_in_corpus", "Hiện mình chưa có thông tin về mức phạt này.")
    )
    confident = score_row(_unanswerable("not_in_corpus", "Mức phạt là 5% số tiền còn nợ [1]."))
    asked = score_row(
        _unanswerable("calc_missing_input", "Cho mình xin số tín chỉ đã học nhé.", asked_back=True)
    )
    computed = score_row(_unanswerable("calc_missing_input", "Bạn còn thiếu 30 tín chỉ."))

    assert refused["pass"] is True and refused["fabricated"] is False
    assert confident["pass"] is False and confident["fabricated"] is True
    assert asked["pass"] is True
    assert computed["pass"] is False and computed["fabricated"] is True


def test_near_miss_fails_when_the_nearly_right_document_is_cited_as_the_answer() -> None:
    from evals.run.metrics import score_row

    wrong = score_row(
        _unanswerable(
            "near_miss",
            "Học phí Khối Công nghệ là 38.350.000 đồng/năm [1].",
            forbidden_source_patterns=["2025-2026"],
            citations=[{"title": "Quyet Dinh 1035 - Hoc phi 2025-2026", "document_id": "x"}],
        )
    )
    honest = score_row(
        _unanswerable(
            "near_miss",
            "Chưa có quyết định học phí 2026-2027; mức 2025-2026 là 38.350.000 đồng [1].",
            forbidden_source_patterns=["2025-2026"],
            citations=[{"title": "Quyet Dinh 1035 - Hoc phi 2025-2026", "document_id": "x"}],
        )
    )

    assert wrong["wrong_source"] is True and wrong["pass"] is False and wrong["fabricated"] is True
    # A stated limit passes even though it cites the nearly-right document for context.
    assert honest["wrong_source"] is True and honest["pass"] is True
    assert honest["fabricated"] is False


def test_a_tagged_calculation_row_must_take_its_expected_path() -> None:
    base: dict[str, Any] = {
        "category": "calculation",
        "expected_numbers": [8],
        "response": "Kéo dài **8 ngày**.",
        "context": [],
    }
    by_formula = _row(
        **base,
        nodes=["07_CalculationNode"],
        calculation_items=[{"mode": "builtin", "status": "computed"}],
        calculation_formulas=["date_span"],
        expected_calculation="formula:date_span",
    )
    assert score_row(by_formula)["pass"] is True
    wrong_formula = dict(by_formula, calculation_formulas=["arithmetic"])
    assert score_row(wrong_formula)["pass"] is False
    assert score_row(wrong_formula)["calc_path"] == "arithmetic"
    # An advisory tag: a quoted answer passes, so does a code-computed calc_request block;
    # a sum the model worked out itself (strict grounding) does not.
    quoted = _row(**base, nodes=["08_RetrievalFilteringNode"], expected_calculation="advisory")
    assert score_row(quoted)["pass"] is True
    block = dict(
        quoted, response="Kết quả ở dưới.\n\n**Phép tính số học: X**\n\nKết quả: **8 ngày**"
    )
    assert score_row(block)["calc_path"] == "calc_request" and score_row(block)["pass"] is True
    self_computed = dict(quoted, warning_codes=["ANSWER_UNGROUNDED_NUMBERS"])
    assert score_row(self_computed)["pass"] is False
    assert (
        score_row(dict(quoted, calculation_formulas=["date_span"], calculation_items=[{}]))["pass"]
        is False
    )
    # An arithmetic tag: the calculation node or a calc_request block (both code-computed);
    # an answer with no computed block does not pass.
    arithmetic = dict(block, expected_calculation="arithmetic")
    assert score_row(arithmetic)["calc_path_ok"] is True and score_row(arithmetic)["pass"] is True
    assert score_row(dict(quoted, expected_calculation="arithmetic"))["pass"] is False
