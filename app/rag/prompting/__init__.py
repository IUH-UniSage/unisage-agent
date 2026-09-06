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

from app.schemas.clarification import PendingClarification
from app.schemas.retrieval import RetrievedChunk
from app.schemas.security import AcademicSecurityContext

from .builder import (
    build_json_repair_prompt,
    build_metadata_section,
    build_missing_metadata_block,
    build_prepared_context_section,
    build_task_2_section,
)
from .loader import get_templates, reset_templates_cache
from .schema import PromptTemplates

__all__ = [
    "PromptTemplates",
    "build_direct_llm_prompt",
    "build_json_repair_prompt",
    "build_missing_metadata_block",
    "build_system_prompt",
    "get_templates",
    "reset_templates_cache",
]

# `security_access_control.yaml` asserts that `<academic_context>` has already
# passed a department_access/access_level filter - not true yet (retrieval
# doesn't filter by permission this phase), so wiring it in now would tell the
# model something false about its own input. Passed as an empty string until
# that filter exists; the placeholder in the copied main templates is kept
# so no template edit is needed to turn it on later.
_SECURITY_ACCESS_CONTROL_DEFERRED = ""


def build_system_prompt(
    *,
    user_query: str,
    security: AcademicSecurityContext,
    confirmed_metadata: dict[str, str],
    chunks: Sequence[RetrievedChunk],
    pending_clarification: PendingClarification | None,
) -> str:
    """Assemble the full prompt for `GenerationSynthesisNode`'s unified advisory
    flow (advisory/procedure/document/calendar - one frame, `academic_domain_rules`
    picks the response format per question type)."""

    templates = get_templates()
    return templates.chat_academic_advisory.format(
        header=templates.header,
        academic_metadata=build_metadata_section(security, confirmed_metadata),
        security_access_control=_SECURITY_ACCESS_CONTROL_DEFERRED,
        academic_domain_rules=templates.academic_domain_rules,
        response_style=templates.response_style,
        citation_rules=templates.citation_rules,
        prepared_context=build_prepared_context_section(chunks),
        task_1=templates.task_1,
        task_2=build_task_2_section(pending_clarification),
        user_query=user_query,
    )


def build_direct_llm_prompt(
    *,
    user_query: str,
    security: AcademicSecurityContext,
    confirmed_metadata: dict[str, str],
) -> str:
    """Assemble the full prompt for `DirectLLMNode` (general-knowledge questions,
    no `<academic_context>` - no `task_1`/`task_2`, no retrieved chunks)."""

    templates = get_templates()
    return templates.chat_direct_llm.format(
        header=templates.header,
        academic_metadata=build_metadata_section(security, confirmed_metadata),
        security_access_control=_SECURITY_ACCESS_CONTROL_DEFERRED,
        response_style=templates.response_style,
        user_query=user_query,
    )
