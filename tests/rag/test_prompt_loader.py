from app.rag.prompting.loader import build_system_prompt
from app.schemas.clarification import PendingClarification
from app.schemas.retrieval import RetrievedChunk
from app.schemas.security import AcademicSecurityContext, DepartmentAccessEntry


def test_prompt_renders_two_separate_tags_never_merged() -> None:
    security = AcademicSecurityContext(
        user_id="u1",
        role="SINH_VIEN",
        department_access=[DepartmentAccessEntry(department_id="KHOA_CNTT", access_level=2)],
    )
    confirmed_metadata = {"training_type": "chinh_quy"}

    prompt = build_system_prompt(
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
        security=AcademicSecurityContext(),
        confirmed_metadata={"training_type": "chinh_quy"},
        chunks=[],
        pending_clarification=None,
    )

    assert '"training_type"' not in prompt  # never JSON-shaped, only prose


def test_prepared_context_renders_chunks_with_citation_index() -> None:
    prompt = build_system_prompt(
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


def test_pending_clarification_renders_as_ask_user_form_json() -> None:
    pending = PendingClarification(
        origin_node="QueryTransformationNode",
        missing_fields=["training_type"],
        options=[["chinh_quy", "lien_thong"]],
        retry_count=0,
    )

    prompt = build_system_prompt(
        security=AcademicSecurityContext(),
        confirmed_metadata={},
        chunks=[],
        pending_clarification=pending,
    )

    assert '"type": "ask_user_form"' in prompt
    assert '"field": "training_type"' in prompt


def test_no_pending_clarification_renders_khong_co() -> None:
    prompt = build_system_prompt(
        security=AcademicSecurityContext(),
        confirmed_metadata={},
        chunks=[],
        pending_clarification=None,
    )

    assert "Không có" in prompt
