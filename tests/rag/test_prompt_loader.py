from app.rag.prompting import (
    build_direct_llm_prompt,
    build_json_repair_prompt,
    build_system_prompt,
    get_templates,
)
from app.schemas.clarification import PendingClarification
from app.schemas.retrieval import RetrievedChunk
from app.schemas.security import AcademicSecurityContext, DepartmentAccessEntry


def test_templates_load_without_error() -> None:
    templates = get_templates()

    assert templates.chat_academic_advisory
    assert templates.chat_direct_llm
    assert "{academic_metadata}" not in templates.header  # loader returns raw text, not re-parsed


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


def test_direct_llm_prompt_has_no_academic_context_or_task_sections() -> None:
    prompt = build_direct_llm_prompt(
        user_query="1 + 1 bằng mấy?",
        security=AcademicSecurityContext(),
        confirmed_metadata={},
    )

    assert "1 + 1 bằng mấy?" in prompt
    assert "<academic_user_context>" in prompt  # still identity-aware
    assert "<academic_context>" not in prompt  # no retrieved context in this flow
    # response_style.yaml (shared) legitimately references "NHIỆM VỤ 1" in one
    # of its formatting rules - task_1/task_2's own headings are the real signal.
    assert "NHIỆM VỤ 1: XÁC NHẬN" not in prompt
    assert "NHIỆM VỤ 2: THU THẬP" not in prompt


def test_guest_security_context_renders_khach_role_and_no_department_access() -> None:
    prompt = build_system_prompt(
        user_query="Điều kiện học bổng là gì?",
        security=AcademicSecurityContext(),  # default: role=KHACH, no department_access
        confirmed_metadata={},
        chunks=[],
        pending_clarification=None,
    )

    assert "Vai trò (Role): KHACH" in prompt
