from datetime import datetime
from pathlib import Path
from unittest.mock import patch

from app.rag.prompting import (
    build_json_repair_prompt,
    build_multi_intent_prompt,
    build_system_prompt,
    get_templates,
)
from app.schemas.chat_history import HistoryMessage
from app.schemas.retrieval import RetrievedChunk
from app.schemas.security import AcademicSecurityContext, DepartmentAccessEntry
from app.schemas.web_search import WebSearchResult

_SNAPSHOT_PATH = Path(__file__).parent.parent / "fixtures" / "advisory_prompt_snapshot.txt"


def test_templates_load_without_error() -> None:
    templates = get_templates()

    assert templates.chat_academic_advisory
    assert "{academic_metadata}" not in templates.header  # loader returns raw text, not re-parsed


def test_design_only_templates_are_loaded() -> None:
    templates = get_templates()

    assert "{sub_queries_list}" in templates.chat_multi_intent_synthesis
    assert "{history_message}" in templates.chat_multi_intent_synthesis
    assert "{ticket_fallback}" in templates.chat_ticket_fallback
    assert "{prepared_context}" not in templates.chat_ticket_fallback
    assert templates.ticket_fallback
    assert templates.agent_calculation_extractor
    assert templates.agent_reranker_compressor


def test_agent_templates_keep_literal_json_braces() -> None:
    """Agent prompts are used verbatim as `system_prompt`, never `.format()`-ed -
    their JSON examples must survive loading with braces intact."""

    templates = get_templates()

    assert '"sub_queries": [' in templates.agent_multi_query_decomposer
    assert '"formula_id": "course_score", "params": {' in templates.agent_calculation_extractor
    assert '"formula_id": "llm", "params": {' in templates.agent_calculation_extractor


def test_task_2_no_longer_mentions_the_pending_block_or_confirmed_metadata() -> None:
    task_2 = get_templates().task_2
    assert "missing_metadata_to_confirm" not in task_2
    assert "confirmed_metadata" not in task_2
    assert "{ask_user_form_guide}" in task_2


def test_calculation_titles_reach_the_advisory_prompt_without_numbers() -> None:
    prompt = build_system_prompt(
        user_query="Điều kiện học bổng?",
        security=AcademicSecurityContext(),
        confirmed_metadata={},
        chunks=[],
        calculation_titles=["Điểm tổng kết học phần"],
    )
    assert "- Điểm tổng kết học phần (đã tính ở trên)" in prompt


def test_template_versions_change_with_content() -> None:
    versions = get_templates().versions
    assert len(versions["chat_calculation_llm"]) == 12
    assert versions["chat_calculation_llm"] != versions["agent_calculation_extractor"]


def test_ticket_fallback_templates_do_not_mention_rerank_score_or_a_button() -> None:
    templates = get_templates()

    for text in (templates.ticket_fallback, templates.chat_ticket_fallback):
        assert "rerank_score" not in text
        assert "nhấn nút" not in text


def test_security_access_control_uses_department_access_not_stale_terms() -> None:
    templates = get_templates()

    assert "max_access_level" not in templates.security_access_control
    assert "organization_scopes" not in templates.security_access_control
    assert "department_access" in templates.security_access_control


def test_advisory_prompt_includes_the_security_access_control_block() -> None:
    prompt = build_system_prompt(
        user_query="Điều kiện học bổng là gì?",
        security=AcademicSecurityContext(),
        confirmed_metadata={},
        chunks=[],
    )

    assert "Quy Tắc Bảo Mật & Phân Quyền Thông Tin" in prompt


def test_prompt_renders_two_separate_tags_never_merged() -> None:
    security = AcademicSecurityContext(
        user_id="u1",
        role="SINH_VIEN",
        department_access=[DepartmentAccessEntry(department_id="KHOA_CNTT", access_level=2)],
    )
    confirmed_metadata = {"training_type": "chinh_quy"}

    prompt = build_system_prompt(
        user_query="Điều kiện học bổng là gì?",
        security=security,
        confirmed_metadata=confirmed_metadata,
        chunks=[],
    )

    assert "<academic_user_context>" in prompt
    assert "</academic_user_context>" in prompt
    assert "<student_declared_attributes>" in prompt
    assert "</student_declared_attributes>" in prompt

    # The two tags never share a body: KHOA_CNTT must not appear inside
    # student_declared_attributes, and training_type must not appear inside
    # academic_user_context.
    user_ctx_start = prompt.index("<academic_user_context>")
    user_ctx_end = prompt.index("</academic_user_context>")
    declared_start = prompt.index("<student_declared_attributes>")
    declared_end = prompt.index("</student_declared_attributes>")

    academic_block = prompt[user_ctx_start:user_ctx_end]
    declared_block = prompt[declared_start:declared_end]

    assert "training_type" not in academic_block
    assert "KHOA_CNTT" not in declared_block


def test_prompt_never_puts_confirmed_metadata_in_qdrant_filter_shape() -> None:
    """Regression guard: confirmed_metadata must only ever end up as prose text,
    never as anything resembling a retrieval filter object."""

    prompt = build_system_prompt(
        user_query="Điều kiện học bổng là gì?",
        security=AcademicSecurityContext(),
        confirmed_metadata={"training_type": "chinh_quy"},
        chunks=[],
    )

    assert '"training_type"' not in prompt  # never JSON-shaped, only prose


def test_prepared_context_renders_chunks_with_citation_index() -> None:
    prompt = build_system_prompt(
        user_query="Điều kiện học bổng là gì?",
        security=AcademicSecurityContext(),
        confirmed_metadata={},
        chunks=[
            RetrievedChunk(
                chunk_id="c1", content="Nội dung 1", source="Quy chế A", faculty="GLOBAL", score=1.0
            )
        ],
    )

    assert "[1] (Quy chế A) Nội dung 1" in prompt


def test_prepared_context_shows_page_suffix_for_pdf_chunks_with_a_single_page() -> None:
    prompt = build_system_prompt(
        user_query="Điều kiện học bổng là gì?",
        security=AcademicSecurityContext(),
        confirmed_metadata={},
        chunks=[
            RetrievedChunk(
                chunk_id="c1",
                content="Nội dung 1",
                source="Quy chế A",
                score=1.0,
                page_start=5,
            )
        ],
    )

    assert "[1] (Quy chế A, tr. 5) Nội dung 1" in prompt


def test_prepared_context_shows_page_range_suffix_when_start_and_end_differ() -> None:
    prompt = build_system_prompt(
        user_query="Điều kiện học bổng là gì?",
        security=AcademicSecurityContext(),
        confirmed_metadata={},
        chunks=[
            RetrievedChunk(
                chunk_id="c1",
                content="Nội dung 1",
                source="Quy chế A",
                score=1.0,
                page_start=5,
                page_end=6,
            )
        ],
    )

    assert "[1] (Quy chế A, tr. 5-6) Nội dung 1" in prompt


def test_prepared_context_has_no_page_suffix_for_non_pdf_chunks() -> None:
    prompt = build_system_prompt(
        user_query="Điều kiện học bổng là gì?",
        security=AcademicSecurityContext(),
        confirmed_metadata={},
        chunks=[
            RetrievedChunk(
                chunk_id="c1",
                content="Nội dung 1",
                source="Quy chế A",
                score=1.0,
            )
        ],
    )

    assert "[1] (Quy chế A) Nội dung 1" in prompt
    assert "tr. None" not in prompt
    # The chunk citation line itself must have no page suffix - this is a
    # narrower check than searching the whole prompt for ", tr." (that
    # substring legitimately appears in citation_rules.yaml's own
    # instructions to the LLM about how to use a page suffix WHEN present).
    assert "[1] (Quy chế A, tr." not in prompt


def test_json_repair_prompt_includes_academic_context_for_verbatim_options() -> None:
    """Without the source chunks, ask_user_form_guide's "copy the branch label
    verbatim" rule has nothing to copy from - the repair prompt must carry
    the same <academic_context> the main prompt saw."""

    prompt = build_json_repair_prompt(
        "Bạn vui lòng cho biết ngành học của bạn nhé!",
        [
            RetrievedChunk(
                chunk_id="c1",
                content="Ngành đào tạo: Công nghệ Thông tin, Logistics, Kế toán.",
                source="Biểu học phí",
                score=0.9,
            )
        ],
        security=AcademicSecurityContext(),
        confirmed_metadata={},
    )

    assert "Bạn vui lòng cho biết ngành học của bạn nhé!" in prompt
    assert "[1] (Biểu học phí) Ngành đào tạo: Công nghệ Thông tin" in prompt
    assert "ASK_USER_FORM_GUIDE" in prompt


def test_no_pending_clarification_renders_khong_co() -> None:
    prompt = build_system_prompt(
        user_query="Điều kiện học bổng là gì?",
        security=AcademicSecurityContext(),
        confirmed_metadata={},
        chunks=[],
    )

    assert "Không có" in prompt


def test_prompt_embeds_the_user_query() -> None:
    prompt = build_system_prompt(
        user_query="Điều kiện học bổng loại giỏi là gì?",
        security=AcademicSecurityContext(),
        confirmed_metadata={},
        chunks=[],
    )

    assert "Điều kiện học bổng loại giỏi là gì?" in prompt


def test_guest_security_context_renders_khach_role_and_no_department_access() -> None:
    prompt = build_system_prompt(
        user_query="Điều kiện học bổng là gì?",
        security=AcademicSecurityContext(),  # default: role=KHACH, no department_access
        confirmed_metadata={},
        chunks=[],
    )

    assert "Vai trò (Role): KHACH" in prompt


def test_known_metadata_fields_vocabulary_is_embedded() -> None:
    """The common-concept-to-canonical-field-name table (known_metadata_fields.json)
    must reach the model in every system prompt (via ask_user_form_guide, always
    embedded regardless of whether a clarification round is active this turn) -
    this is what stops the model inventing a second field name for a concept
    already covered (observed live: 'nganh_hoc' vs 'nganh_dao_tao')."""

    prompt = build_system_prompt(
        user_query="Điều kiện học bổng là gì?",
        security=AcademicSecurityContext(),
        confirmed_metadata={},
        chunks=[],
    )

    assert "he_dao_tao" in prompt
    assert "nganh_hoc" in prompt


def test_history_message_renders_prior_turns_with_role_labels() -> None:
    prompt = build_system_prompt(
        user_query="Còn học phí thì sao?",
        security=AcademicSecurityContext(),
        confirmed_metadata={},
        chunks=[],
        history=[
            HistoryMessage(role="USER", content="Tôi học ngành CNTT"),
            HistoryMessage(role="ASSISTANT", content="Ngành CNTT có mã 7480201."),
        ],
    )

    assert "<history_message>" in prompt
    assert "Người dùng: Tôi học ngành CNTT" in prompt
    assert "Trợ lý: Ngành CNTT có mã 7480201." in prompt


def test_history_message_empty_renders_sentinel_not_a_crash() -> None:
    prompt = build_system_prompt(
        user_query="hello",
        security=AcademicSecurityContext(),
        confirmed_metadata={},
        chunks=[],
    )

    assert "<history_message>" in prompt
    assert "chưa có lịch sử" in prompt


def test_json_repair_prompt_includes_student_declared_attributes() -> None:
    """Regression guard for the live bug: repair must see confirmed_metadata
    so it doesn't re-ask a field the student already answered."""

    prompt = build_json_repair_prompt(
        "Bạn cho mình biết hệ đào tạo nhé!",
        [],
        security=AcademicSecurityContext(),
        confirmed_metadata={"he_dao_tao": "chinh_quy"},
    )

    assert "<student_declared_attributes>" in prompt
    assert "he_dao_tao: chinh_quy" in prompt


def test_citation_rules_ask_for_inline_markers_without_trailing_source_block() -> None:
    prompt = build_system_prompt(
        user_query="Học phí là bao nhiêu?",
        security=AcademicSecurityContext(),
        confirmed_metadata={},
        chunks=[RetrievedChunk(chunk_id="c1", content="Nội dung 1", source="Quy chế A", score=1.0)],
    )

    assert "SAU câu hoặc đoạn dùng nguồn" in prompt
    assert "MỖI hàng dữ liệu của bảng" in prompt
    assert "Văn bản tham chiếu chính thức" not in prompt


def test_resolved_query_prepended_when_it_differs_from_raw_user_query() -> None:
    security = AcademicSecurityContext(user_id="u1", role="SINH_VIEN", department_access=[])

    prompt = build_system_prompt(
        user_query="còn Khóa tuyển sinh năm học 2023-2024 thì sao",
        resolved_query=(
            "Học phí đại học chính quy khóa tuyển sinh năm học 2023-2024 ngành Công nghệ "
            "là bao nhiêu?"
        ),
        security=security,
        confirmed_metadata={},
        chunks=[],
    )

    assert "Học phí đại học chính quy khóa tuyển sinh năm học 2023-2024" in prompt
    assert "còn Khóa tuyển sinh năm học 2023-2024 thì sao" in prompt


def test_resolved_query_omitted_when_same_as_raw_user_query() -> None:
    security = AcademicSecurityContext(user_id="u1", role="SINH_VIEN", department_access=[])
    raw_query = "Điều kiện học bổng là gì?"

    with_resolved = build_system_prompt(
        user_query=raw_query,
        resolved_query=raw_query,
        security=security,
        confirmed_metadata={},
        chunks=[],
    )
    without_resolved = build_system_prompt(
        user_query=raw_query,
        security=security,
        confirmed_metadata={},
        chunks=[],
    )

    assert with_resolved == without_resolved


def test_build_system_prompt_unchanged_by_the_multi_intent_frame_refactor() -> None:
    """Snapshot captured before `_base_params` was factored out of
    `build_system_prompt` - a single-question advisory prompt must render
    byte-for-byte the same after the refactor."""

    with patch("app.rag.prompting.builder.now_ict", return_value=datetime(2026, 9, 24, 9, 0)):
        prompt = build_system_prompt(
            user_query="còn khóa 2024 thì sao?",
            resolved_query="Học phí khóa 2024 ngành CNTT là bao nhiêu?",
            security=AcademicSecurityContext(
                user_id="u1",
                role="SINH_VIEN",
                department_access=[
                    DepartmentAccessEntry(department_id="KHOA_CNTT", access_level=2)
                ],
            ),
            confirmed_metadata={"he_dao_tao": "chinh_quy"},
            chunks=[
                RetrievedChunk(
                    chunk_id="c1",
                    content="Học phí khóa 2024...",
                    source="hoc-phi.pdf",
                    score=0.9,
                    page_start=3,
                ),
                RetrievedChunk(
                    chunk_id="c2", content="Điều 5...", source="quy-che.docx", score=0.8
                ),
            ],
            history=[
                HistoryMessage(role="USER", content="Học phí khóa 2023?"),
                HistoryMessage(role="ASSISTANT", content="Học phí khóa 2023 là ... [1]."),
            ],
        )

    assert prompt == _SNAPSHOT_PATH.read_text(encoding="utf-8")


def test_build_multi_intent_prompt_renders_sub_queries_and_shares_base_params() -> None:
    prompt = build_multi_intent_prompt(
        user_query="Học phí ngành CNTT bao nhiêu, với lại điều kiện học bổng là gì?",
        sub_queries=["Học phí ngành CNTT bao nhiêu?", "Điều kiện học bổng là gì?"],
        security=AcademicSecurityContext(),
        confirmed_metadata={},
        chunks=[RetrievedChunk(chunk_id="c1", content="Nội dung", source="s", score=0.9)],
    )

    assert "SQ1. Học phí ngành CNTT bao nhiêu?" in prompt
    assert "SQ2. Điều kiện học bổng là gì?" in prompt
    assert "<academic_context>" in prompt
    assert "Nội dung" in prompt


def test_build_multi_intent_prompt_never_shows_the_resolved_query_marker() -> None:
    """The multi-intent frame has no `resolved_query` concept - it always
    renders the plain `user_query`, unlike the advisory frame."""

    prompt = build_multi_intent_prompt(
        user_query="Câu hỏi gốc",
        sub_queries=["SQ một", "SQ hai"],
        security=AcademicSecurityContext(),
        confirmed_metadata={},
        chunks=[],
    )

    assert "Câu hỏi gốc" in prompt
    assert "Nguyên văn người dùng vừa nhắn" not in prompt


def _web_page(title: str) -> WebSearchResult:
    return WebSearchResult(
        title=title, url=f"https://pdt.iuh.edu.vn/{title}", content=f"Nội dung {title}", score=0.8
    )


def test_websearch_block_sits_below_academic_context_numbered_after_the_chunks() -> None:
    prompt = build_multi_intent_prompt(
        user_query="Học phí và lịch thi?",
        sub_queries=["Học phí?", "Lịch thi?"],
        security=AcademicSecurityContext(),
        confirmed_metadata={},
        chunks=[
            RetrievedChunk(chunk_id="c1", content="Học phí...", source="hoc-phi.pdf", score=0.9),
            RetrievedChunk(chunk_id="c2", content="Miễn giảm...", source="mien.pdf", score=0.8),
        ],
        web_results=[_web_page("lich-thi")],
    )

    assert "[3] (lich-thi — https://pdt.iuh.edu.vn/lich-thi) Nội dung lich-thi" in prompt
    # Tag names also appear inside the rules; compare the blocks' own lines.
    assert prompt.index("\n</academic_context>") < prompt.index("\n<websearch>\n")
    assert prompt.index("\n</websearch>") < prompt.index("\n<current_date>\n")
    assert "luôn theo `<academic_context>`" in prompt


def test_websearch_only_turn_has_an_empty_academic_context() -> None:
    prompt = build_system_prompt(
        user_query="Lịch thi học kỳ 1?",
        security=AcademicSecurityContext(),
        confirmed_metadata={},
        chunks=[],
        web_results=[_web_page("lich-thi")],
    )

    assert "(không có tài liệu liên quan)" in prompt
    assert "[1] (lich-thi — https://pdt.iuh.edu.vn/lich-thi)" in prompt


def test_no_web_results_renders_no_websearch_block() -> None:
    prompt = build_system_prompt(
        user_query="Học phí?",
        security=AcademicSecurityContext(),
        confirmed_metadata={},
        chunks=[RetrievedChunk(chunk_id="c1", content="x", source="s", score=0.9)],
    )

    assert "<websearch>\n" not in prompt
    assert "## Thông Tin Bổ Sung Từ Website" not in prompt


def test_json_repair_prompt_sees_the_same_web_results() -> None:
    prompt = build_json_repair_prompt(
        "Bạn cho mình biết thêm hệ đào tạo nhé.",
        [],
        security=AcademicSecurityContext(),
        confirmed_metadata={},
        web_results=[_web_page("lich-thi")],
    )

    assert "[1] (lich-thi — https://pdt.iuh.edu.vn/lich-thi)" in prompt


def test_calculation_llm_prompt_keeps_its_ask_form_json_after_formatting() -> None:
    from app.rag.prompting import build_calculation_llm_prompt

    prompt = build_calculation_llm_prompt(
        user_query="q", builtin_rules="R", documents="", known_values="{}", history=[]
    )
    assert '{"type": "ask_user_form", "fields": [{"field":' in prompt
    assert "(không có tài liệu)" in prompt
