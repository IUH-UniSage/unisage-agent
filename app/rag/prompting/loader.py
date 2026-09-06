"""Prompt assembly for `GenerationSynthesisNode`/`DirectLLMNode`.

Builds the system prompt as plain Python string assembly rather than a
YAML-file template loader. Enforces two invariants: the two identity tags
(`academic_user_context` vs `student_declared_attributes`) are always
separate and never merged, and the `ask_user_form`/`missing_metadata` block
feeds the clarification flow. The system-prompt copywriting here is a
placeholder, not tuned prompt text; a follow-up could load it from an
external template tree instead, keeping the same function signatures.
"""

import json
from collections.abc import Sequence

from app.schemas.clarification import PendingClarification
from app.schemas.retrieval import RetrievedChunk
from app.schemas.security import AcademicSecurityContext

_NO_DECLARED_ATTRIBUTES = "Chưa có thuộc tính nào được sinh viên xác nhận."
_NO_PENDING_CLARIFICATION = "Không có"


def render_academic_user_context(security: AcademicSecurityContext) -> str:
    """`<academic_user_context>` — JWT-derived, verified identity.

    Never receives data from `confirmed_metadata` — that is a hard boundary:
    this tag is the only one an authorization decision may ever be based on.
    """

    department_lines = (
        "\n".join(
            f"    - {entry.department_id} (access_level={entry.access_level})"
            for entry in security.department_access
        )
        or "    - (không có phòng ban nào được cấp quyền)"
    )
    return (
        "<academic_user_context>\n"
        f"  user_id: {security.user_id or '(khách vãng lai)'}\n"
        f"  role: {security.role}\n"
        "  department_access:\n"
        f"{department_lines}\n"
        "</academic_user_context>"
    )


def render_student_declared_attributes(confirmed_metadata: dict[str, str]) -> str:
    """`<student_declared_attributes>` — self-declared, UNVERIFIED.

    Deliberately named and rendered separately from
    `render_academic_user_context` so an LLM (or a future refactor) cannot
    conflate "trusted" with "self-declared".
    """

    body = (
        "\n".join(f"    - {field}: {value}" for field, value in confirmed_metadata.items())
        or f"    {_NO_DECLARED_ATTRIBUTES}"
    )
    return f"<student_declared_attributes>\n{body}\n</student_declared_attributes>"


def render_prepared_context(chunks: Sequence[RetrievedChunk]) -> str:
    if not chunks:
        return "<academic_context>\n  (không có tài liệu liên quan)\n</academic_context>"
    lines = [
        f"  [{index}] ({chunk.source}) {chunk.content}" for index, chunk in enumerate(chunks, 1)
    ]
    return "<academic_context>\n" + "\n".join(lines) + "\n</academic_context>"


def render_missing_metadata_block(pending: PendingClarification | None) -> str:
    """`{missing_metadata_to_confirm}` — JSON `ask_user_form` shape, or "Không có".

    Source is always `pending_clarification` — never re-derived here.
    """

    if pending is None:
        return _NO_PENDING_CLARIFICATION
    fields = [
        {
            "field": field,
            "options": [{"id": option_id, "label": option_id} for option_id in options]
            if options is not None
            else None,
        }
        for field, options in zip(pending.missing_fields, pending.options, strict=True)
    ]
    return json.dumps({"type": "ask_user_form", "fields": fields}, ensure_ascii=False)


def build_system_prompt(
    *,
    security: AcademicSecurityContext,
    confirmed_metadata: dict[str, str],
    chunks: Sequence[RetrievedChunk],
    pending_clarification: PendingClarification | None,
) -> str:
    """Assemble the full system prompt handed to `GenerationSynthesisNode`'s Agent.

    Order matters only for readability; the two identity tags are always
    both present and always separate blocks.
    """

    return "\n\n".join(
        [
            "Bạn là Trợ Lý AI Học Vụ của trường Đại học. Chỉ trả lời dựa trên "
            "<academic_context> bên dưới, luôn gắn trích dẫn [1][2] cho mỗi khẳng định.",
            render_academic_user_context(security),
            render_student_declared_attributes(confirmed_metadata),
            render_prepared_context(chunks),
            "Nếu văn bản chia nhánh theo một thuộc tính sinh viên chưa biết (không có trong "
            "academic_user_context lẫn student_declared_attributes), hỏi lại đúng MỘT lần bằng "
            "cách kết thúc câu trả lời với một khối ```json ask_user_form``` duy nhất, ví dụ:\n"
            '{"type": "ask_user_form", "fields": [{"field": "...", "options": [...]}]}',
            f"missing_metadata_to_confirm (lượt trước, nếu có): "
            f"{render_missing_metadata_block(pending_clarification)}",
        ]
    )
