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

from collections.abc import Mapping, Sequence

from app.schemas.chat_history import HistoryMessage
from app.schemas.clarification import LastCalculation
from app.schemas.retrieval import RetrievedChunk
from app.schemas.security import AcademicSecurityContext
from app.schemas.web_search import WebSearchResult

from .builder import (
    append_current_time,
    append_recent_history,
    build_ask_user_form_guide,
    build_calculation_results_section,
    build_current_time_section,
    build_history_section,
    build_json_repair_prompt,
    build_known_metadata_fields_section,
    build_metadata_section,
    build_prepared_context_section,
    build_recent_calculations_section,
    build_task_2_section,
    build_web_sources_section,
    render_resolved_user_query,
)
from .loader import get_templates, reset_templates_cache
from .schema import PromptTemplates

__all__ = [
    "PromptTemplates",
    "append_current_time",
    "append_recent_history",
    "build_ask_user_form_guide",
    "build_calculation_llm_prompt",
    "build_calculation_review_prompt",
    "build_current_time_section",
    "build_history_section",
    "build_json_repair_prompt",
    "build_known_metadata_fields_section",
    "build_multi_intent_prompt",
    "build_system_prompt",
    "build_ticket_fallback_prompt",
    "get_templates",
    "render_resolved_user_query",
    "render_sub_queries_list",
    "reset_templates_cache",
]


def _base_params(
    *,
    security: AcademicSecurityContext,
    confirmed_metadata: dict[str, str],
    chunks: Sequence[RetrievedChunk],
    web_results: Sequence[WebSearchResult],
    history: Sequence[HistoryMessage],
) -> dict[str, str]:
    """Blocks shared by every GenerationSynthesisNode frame."""

    templates = get_templates()
    return {
        "header": templates.header,
        "academic_metadata": build_metadata_section(security, confirmed_metadata),
        "history_message": build_history_section(history),
        "security_access_control": templates.security_access_control,
        "academic_domain_rules": templates.academic_domain_rules,
        "response_style": templates.response_style,
        "citation_rules": templates.citation_rules,
        "prepared_context": build_prepared_context_section(chunks, web_results, confirmed_metadata),
        "task_1": templates.task_1,
        "task_2": build_task_2_section(),
    }


def build_system_prompt(
    *,
    user_query: str,
    resolved_query: str | None = None,
    security: AcademicSecurityContext,
    confirmed_metadata: dict[str, str],
    chunks: Sequence[RetrievedChunk],
    history: Sequence[HistoryMessage] = (),
    web_results: Sequence[WebSearchResult] = (),
    calculation_titles: Sequence[str] = (),
) -> str:
    """Advisory frame for a single question. `resolved_query` (the standalone
    rewrite of a follow-up) is shown ahead of the raw message."""

    return get_templates().chat_academic_advisory.format(
        **_base_params(
            security=security,
            confirmed_metadata=confirmed_metadata,
            chunks=chunks,
            web_results=web_results,
            history=history,
        ),
        user_query=render_resolved_user_query(user_query, resolved_query),
        calculation_results=build_calculation_results_section(calculation_titles),
    )


def build_multi_intent_prompt(
    *,
    user_query: str,
    sub_queries: Sequence[str],
    security: AcademicSecurityContext,
    confirmed_metadata: dict[str, str],
    chunks: Sequence[RetrievedChunk],
    history: Sequence[HistoryMessage] = (),
    web_results: Sequence[WebSearchResult] = (),
    calculation_titles: Sequence[str] = (),
) -> str:
    """Multi-intent frame for several sub-queries, listed as `SQk. ...`."""

    return get_templates().chat_multi_intent_synthesis.format(
        **_base_params(
            security=security,
            confirmed_metadata=confirmed_metadata,
            chunks=chunks,
            web_results=web_results,
            history=history,
        ),
        sub_queries_list=render_sub_queries_list(sub_queries),
        user_query=user_query,
        calculation_results=build_calculation_results_section(calculation_titles),
    )


def build_ticket_fallback_prompt(
    *,
    user_query: str,
    security: AcademicSecurityContext,
    confirmed_metadata: dict[str, str],
    history: Sequence[HistoryMessage] = (),
    web_sources: Sequence[WebSearchResult] = (),
) -> str:
    """TicketFallbackNode frame: no `{prepared_context}`/`task_1`/`task_2` -
    the model gets no chunks and no clarification machinery, so it can only
    write the fallback message, never cite a regulation it has no source for.
    `web_sources` (official pages the web judge kept whose snippet lacks the answer) are
    only linked, with their title and academic year - never quoted for figures."""

    templates = get_templates()
    return templates.chat_ticket_fallback.format(
        header=templates.header,
        academic_metadata=build_metadata_section(security, confirmed_metadata),
        history_message=build_history_section(history),
        security_access_control=templates.security_access_control,
        response_style=templates.response_style,
        ticket_fallback=templates.ticket_fallback,
        web_sources=build_web_sources_section(web_sources),
        current_time=build_current_time_section(confirmed_metadata),
        user_query=user_query,
    )


def render_sub_queries_list(sub_queries: Sequence[str]) -> str:
    return "\n".join(f"SQ{index}. {query}" for index, query in enumerate(sub_queries, 1))


def build_calculation_review_prompt(
    *,
    user_query: str,
    calculation_payload: str,
    history: Sequence[HistoryMessage],
    confirmed_metadata: Mapping[str, str] | None = None,
) -> str:
    """`main/chat_calculation_review.yaml`: after a calculation-only turn, check the
    built-in results answer what was asked and write the opening answer."""

    templates = get_templates()
    return templates.chat_calculation_review.format(
        header=templates.header,
        calculation_payload=calculation_payload,
        history_message=build_history_section(history),
        current_time=build_current_time_section(confirmed_metadata),
        user_query=user_query,
    )


def build_calculation_llm_prompt(
    *,
    user_query: str,
    builtin_rules: str,
    documents: str,
    known_values: str,
    history: Sequence[HistoryMessage],
    security: AcademicSecurityContext,
    confirmed_metadata: dict[str, str],
    original_message: str | None = None,
    recent_calculations: Sequence[LastCalculation] = (),
) -> str:
    """`main/chat_calculation_llm.yaml`: a calculation the LLM does itself (target
    questions, formulas from documents, follow-ups) - everything but the forward
    calculation of the built-in formulas. `original_message`: what the student typed
    this turn, shown under the classifier's normalised question when they differ.
    It reads the profile and the recent calculations before asking anything, and asks
    under the registry's field names."""

    templates = get_templates()
    return templates.chat_calculation_llm.format(
        header=templates.header,
        academic_metadata=build_metadata_section(security, confirmed_metadata),
        security_access_control=templates.security_access_control,
        academic_domain_rules=templates.academic_domain_rules,
        response_style=templates.response_style,
        citation_rules=templates.citation_rules,
        builtin_rules=builtin_rules,
        documents=documents or "(không có tài liệu)",
        known_values=known_values,
        recent_calculations=build_recent_calculations_section(recent_calculations),
        metadata_fields=build_known_metadata_fields_section(every_scope=True),
        history=build_history_section(history),
        current_time=build_current_time_section(confirmed_metadata),
        user_query=user_query,
        original_message=_original_message_line(original_message, user_query),
    )


def _original_message_line(original: str | None, query: str) -> str:
    if not original or original.strip() == query.strip():
        return ""
    return (
        "\nTin nhắn gốc sinh viên gõ ở lượt này (nguyên văn, có thể gồm cả câu hỏi khác; câu "
        "hỏi ở trên là bản đã chuẩn hoá - khác nghĩa thì tin theo tin nhắn gốc): "
        f"{original.strip()}"
    )
