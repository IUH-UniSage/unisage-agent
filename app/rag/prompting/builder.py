"""Prompt building functions - assemble YAML template sections with graph state."""

from __future__ import annotations

import json
from collections.abc import Sequence

from app.schemas.clarification import PendingClarification
from app.schemas.retrieval import RetrievedChunk
from app.schemas.security import AcademicSecurityContext

from .loader import get_templates

_NO_DECLARED_ATTRIBUTES = "Chưa có thuộc tính nào được sinh viên xác nhận."
_NO_DEPARTMENT_ACCESS = "    - (không có phòng ban nào được cấp quyền)"
_NO_RETRIEVED_CONTEXT = "  (không có tài liệu liên quan)"
NO_PENDING_CLARIFICATION = "Không có"


def build_metadata_section(
    security: AcademicSecurityContext,
    confirmed_metadata: dict[str, str],
) -> str:
    """Build `{academic_metadata}` - two separate XML tags, JWT-verified identity
    (`academic_user_context`) vs self-declared attributes (`student_declared_attributes`).
    Never merged: authorization must only ever be based on the former."""

    department_access_lines = (
        "\n".join(
            f"    - {entry.department_id} (access_level={entry.access_level})"
            for entry in security.department_access
        )
        or _NO_DEPARTMENT_ACCESS
    )
    confirmed_metadata_text = (
        "\n".join(f"    - {field}: {value}" for field, value in confirmed_metadata.items())
        or f"    {_NO_DECLARED_ATTRIBUTES}"
    )
    return get_templates().academic_metadata.format(
        user_id=security.user_id or "(khách vãng lai)",
        role=security.role,
        department_access_lines=department_access_lines,
        confirmed_metadata=confirmed_metadata_text,
    )


def build_prepared_context_section(chunks: Sequence[RetrievedChunk]) -> str:
    """Build `{prepared_context}` - the `<academic_context>` block, chunks numbered
    to match the `[1][2]` citation indices the generation prompt asks the model to use."""

    context_chunks = (
        "\n".join(
            f"  [{index}] ({chunk.source}) {chunk.content}" for index, chunk in enumerate(chunks, 1)
        )
        or _NO_RETRIEVED_CONTEXT
    )
    return get_templates().prepared_context.format(context_chunks=context_chunks)


def build_missing_metadata_block(pending: PendingClarification | None) -> str:
    """`<missing_metadata_to_confirm>` content - JSON `ask_user_form` shape sourced
    from `pending_clarification`, or the `"Không có"` sentinel that keeps
    `{task_2}` silent for this turn."""

    if pending is None:
        return NO_PENDING_CLARIFICATION
    fields = [
        {
            "field": field,
            "options": (
                [{"id": option_id, "label": option_id} for option_id in options]
                if options is not None
                else None
            ),
        }
        for field, options in zip(pending.missing_fields, pending.options, strict=True)
    ]
    return json.dumps({"type": "ask_user_form", "fields": fields}, ensure_ascii=False)


def build_task_2_section(pending: PendingClarification | None) -> str:
    """Build `{task_2}` - nested format: `task_2.yaml` embeds the static
    `ask_user_form_guide.yaml` plus the dynamic missing-metadata block."""

    templates = get_templates()
    return templates.task_2.format(
        missing_metadata_to_confirm=build_missing_metadata_block(pending),
        ask_user_form_guide=templates.ask_user_form_guide,
    )
