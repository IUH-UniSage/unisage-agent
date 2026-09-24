from app.rag.prompting import (
    build_json_repair_prompt,
    build_system_prompt,
    get_templates,
)
from app.schemas.chat_history import HistoryMessage
from app.schemas.clarification import PendingClarification
from app.schemas.retrieval import RetrievedChunk
from app.schemas.security import AcademicSecurityContext, DepartmentAccessEntry


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
    assert '"missing_params": [' in templates.agent_calculation_extractor


def test_ticket_fallback_templates_do_not_mention_rerank_score_or_a_button() -> None:
    templates = get_templates()

    for text in (templates.ticket_fallback, templates.chat_ticket_fallback):
        assert "rerank_score" not in text
        assert "nhấn nút" not in text


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
        pending_clarification=None,
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
        pending_clarification=None,
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
        pending_clarification=None,
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
        pending_clarification=None,
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
        pending_clarification=None,
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
        pending_clarification=None,
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


def test_pending_clarification_renders_as_ask_user_form_json() -> None:
    pending = PendingClarification(
        origin_node="QueryTransformationNode",
        missing_fields=["training_type"],
        options=[["chinh_quy", "lien_thong"]],
        retry_count=0,
    )

    prompt = build_system_prompt(
        user_query="Điều kiện học bổng là gì?",
        security=AcademicSecurityContext(),
        confirmed_metadata={},
        chunks=[],
        pending_clarification=pending,
    )

    assert '"type": "ask_user_form"' in prompt
    assert '"field": "training_type"' in prompt


def test_no_pending_clarification_renders_khong_co() -> None:
    prompt = build_system_prompt(
        user_query="Điều kiện học bổng là gì?",
        security=AcademicSecurityContext(),
        confirmed_metadata={},
        chunks=[],
        pending_clarification=None,
    )

    assert "Không có" in prompt


def test_prompt_embeds_the_user_query() -> None:
    prompt = build_system_prompt(
        user_query="Điều kiện học bổng loại giỏi là gì?",
        security=AcademicSecurityContext(),
        confirmed_metadata={},
        chunks=[],
        pending_clarification=None,
    )

    assert "Điều kiện học bổng loại giỏi là gì?" in prompt


def test_guest_security_context_renders_khach_role_and_no_department_access() -> None:
    prompt = build_system_prompt(
        user_query="Điều kiện học bổng là gì?",
        security=AcademicSecurityContext(),  # default: role=KHACH, no department_access
        confirmed_metadata={},
        chunks=[],
        pending_clarification=None,
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
        pending_clarification=None,
    )

    assert "he_dao_tao" in prompt
    assert "nganh_hoc" in prompt


def test_history_message_renders_prior_turns_with_role_labels() -> None:
    prompt = build_system_prompt(
        user_query="Còn học phí thì sao?",
        security=AcademicSecurityContext(),
        confirmed_metadata={},
        chunks=[],
        pending_clarification=None,
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
        pending_clarification=None,
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
        pending_clarification=None,
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
        pending_clarification=None,
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
        pending_clarification=None,
    )
    without_resolved = build_system_prompt(
        user_query=raw_query,
        security=security,
        confirmed_metadata={},
        chunks=[],
        pending_clarification=None,
    )

    assert with_resolved == without_resolved
