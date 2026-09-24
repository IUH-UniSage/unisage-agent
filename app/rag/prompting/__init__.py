"""Chat prompt assembly - YAML-based.

Prompt content lives in `prompt_templates/` (`common/` building blocks,
`main/` per-node frames); this package only contains loading (`loader.py`)
and dynamic assembly (`builder.py`) logic. Enforces one invariant across
every prompt built here: the two identity blocks inside `{academic_metadata}`
(`academic_user_context`, JWT-verified, vs `student_declared_attributes`,
self-declared) are always rendered separately and never merged - only the
former may ever be used to make an authorization decision.
"""

from __future__ import annotations

from collections.abc import Sequence

from app.schemas.chat_history import HistoryMessage
from app.schemas.clarification import PendingClarification
from app.schemas.retrieval import RetrievedChunk
from app.schemas.security import AcademicSecurityContext

from .builder import (
    append_recent_history,
    build_ask_user_form_guide,
    build_history_section,
    build_json_repair_prompt,
    build_known_metadata_fields_section,
    build_metadata_section,
    build_missing_metadata_block,
    build_prepared_context_section,
    build_task_2_section,
    render_resolved_user_query,
)
from .loader import get_templates, reset_templates_cache
from .schema import PromptTemplates

__all__ = [
    "PromptTemplates",
    "append_recent_history",
    "build_ask_user_form_guide",
    "build_history_section",
    "build_json_repair_prompt",
    "build_known_metadata_fields_section",
    "build_missing_metadata_block",
    "build_system_prompt",
    "get_templates",
    "render_resolved_user_query",
    "reset_templates_cache",
]


def build_system_prompt(
    *,
    user_query: str,
    resolved_query: str | None = None,
    security: AcademicSecurityContext,
    confirmed_metadata: dict[str, str],
    chunks: Sequence[RetrievedChunk],
    pending_clarification: PendingClarification | None,
    history: Sequence[HistoryMessage] = (),
) -> str:
    """Assemble the full prompt for `GenerationSynthesisNode`'s unified advisory
    flow (advisory/procedure/document/calendar - one frame, `academic_domain_rules`
    picks the response format per question type).

    `resolved_query` is `QueryTransformationNode`'s self-contained rewrite of
    a follow-up turn (see `extract_standalone_question`) - e.g. turn 3's raw
    "còn ĐHCQ ngành CN thì sao" carries no khóa on its own. Retrieval already
    benefits from it (it drove the HyDE search), but without it here too,
    generation has to re-derive the same topic from raw `<history_message>`
    text on its own - which is exactly the step observed to drop the khóa
    and answer the wrong row. `build_render_user_query` folds it into
    `{user_query}` so no template placeholder needs to change for this."""

    templates = get_templates()
    return templates.chat_academic_advisory.format(
        header=templates.header,
        academic_metadata=build_metadata_section(security, confirmed_metadata),
        history_message=build_history_section(history),
        security_access_control=templates.security_access_control,
        academic_domain_rules=templates.academic_domain_rules,
        response_style=templates.response_style,
        citation_rules=templates.citation_rules,
        prepared_context=build_prepared_context_section(chunks),
        task_1=templates.task_1,
        task_2=build_task_2_section(pending_clarification),
        user_query=render_resolved_user_query(user_query, resolved_query),
    )
