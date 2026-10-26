"""Prompt building functions - assemble YAML template sections with graph state."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence

from app.core.academic_calendar import (
    INTAKE_FIELD,
    cohort_label,
    describe_current_term,
    describe_student_term,
    parse_intake_year,
)
from app.core.timezone import now_ict
from app.graph.address import address_for
from app.schemas.chat_history import HistoryMessage
from app.schemas.retrieval import RetrievedChunk
from app.schemas.security import AcademicSecurityContext
from app.schemas.web_search import WebSearchResult

from .loader import get_known_metadata_fields, get_templates

_NO_DECLARED_ATTRIBUTES = "Chưa có thuộc tính nào được sinh viên xác nhận."
_NO_DEPARTMENT_ACCESS = "    - (không có phòng ban nào được cấp quyền)"
_NO_RETRIEVED_CONTEXT = "  (không có tài liệu liên quan)"
_NO_HISTORY = "  (đây là lượt đầu tiên, chưa có lịch sử)"
_HISTORY_ROLE_LABELS = {"USER": "Người dùng", "ASSISTANT": "Trợ lý"}
NOTHING = "Không có"

RECENT_HISTORY_LIMIT = 4
_RECENT_ASSISTANT_MAX_CHARS = 300
_CITATION_MARKER_PATTERN = re.compile(r"\[\d+(?:\s*,\s*\d+)*\]")
_JSON_BLOCK_PATTERN = re.compile(r"```json.*?```", re.DOTALL)
# Amounts like "60.000.000" - the previous turn's figures belong to a
# different question, and would otherwise be copied into this turn's output.
_AMOUNT_PATTERN = re.compile(r"\d{1,3}(?:[.,]\d{3})+")
# The tags that frame retrieved text in `prepared_context.yaml` /
# `web_search_context.yaml` / `web_candidate_sources.yaml`. Text from a document
# or a web page that contains one of them could close the frame early and have
# what follows read as prompt rather than data.
_CONTEXT_FRAME_TAG_PATTERN = re.compile(
    r"<\s*(/?)\s*(academic_context|websearch|web_sources|current_date|current_time)\s*>",
    re.IGNORECASE,
)


def build_metadata_section(
    security: AcademicSecurityContext,
    confirmed_metadata: dict[str, str],
) -> str:
    """Build `{academic_metadata}` - two separate XML tags, JWT-verified identity
    (`academic_user_context`) vs self-declared attributes (`declared_attributes`).
    Never merged: authorization must only ever be based on the former. The form of
    address is worked out here from the role, not left to the model: told only the
    role-to-address rule, a model reading a student-like profile under SUPER_ADMIN
    went back to "mình/bạn" (16-10-2026)."""

    address = address_for(security)

    department_access_lines = (
        "\n".join(
            f"    - {entry.department_id} (access_level={entry.access_level})"
            for entry in security.department_access
        )
        or _NO_DEPARTMENT_ACCESS
    )
    return get_templates().academic_metadata.format(
        user_id=security.user_id or "(khách vãng lai)",
        role=security.role,
        address_me=address.me,
        address_you=address.you,
        department_access_lines=department_access_lines,
        declared_attributes=build_declared_attributes_section(confirmed_metadata),
    )


def build_declared_attributes_section(confirmed_metadata: Mapping[str, str]) -> str:
    """`<declared_attributes>`: what the student declared about themselves. The
    same block in the answering prompts (inside `{academic_metadata}`) and in the message
    to the classifier, so both read the profile under one tag and in one format."""

    declared_lines = (
        "\n".join(
            f"- {field}: {_declared_value(field, value)}"
            for field, value in confirmed_metadata.items()
        )
        or _NO_DECLARED_ATTRIBUTES
    )
    return get_templates().declared_attributes.format(declared_lines=declared_lines).strip()


def _declared_value(field: str, value: str) -> str:
    """The intake year carries its cohort ("2022 (khóa K18)"), so the model never
    names the cohort after the year ("K2022")."""

    if field != INTAKE_FIELD:
        return value
    year = parse_intake_year(value, now_ict().date())
    return f"{value} (khóa {cohort_label(year)})" if year is not None else value


def build_known_metadata_fields_section(*, every_scope: bool = False) -> str:
    """Render `known_metadata_fields.json` as the markdown table
    `ask_user_form_guide.yaml` embeds via `{known_metadata_fields}` - see
    that file for the field-name-reuse rule this backs. Static data (doesn't
    vary per conversation), but kept as a JSON file rather than hardcoded in
    the YAML prose so it stays easy to extend/audit as its own artifact.
    Node 10 only asks profile fields; the LLM calculation (`every_scope`) also
    names a calculation's numbers."""

    rows = "\n".join(
        f"    | {entry['concept']} | `{entry['field']}` | {entry['description']} |"
        for entry in get_known_metadata_fields()
        if every_scope or entry.get("scope", "profile") == "profile"
    )
    return (
        "| Khái niệm (mọi cách diễn đạt tương đương) | Tên field bắt buộc dùng lại | Ghi chú |\n"
        "    |---|---|---|\n" + rows
    )


def build_ask_user_form_guide() -> str:
    """`ask_user_form_guide.yaml`'s raw template still has one placeholder
    of its own (`{known_metadata_fields}`) that a plain `str.format()` on
    whatever embeds it (`task_2.yaml`, `json_repair.yaml`) would NOT fill in
    automatically - `.format()` doesn't recurse into an already-substituted
    value. Resolve it here once, so every caller gets the fully-rendered
    guide text.

    Uses `.replace(...)`, NOT `.format(...)`: this file's own content is full
    of literal ```json {"type": ...} example blocks, and `.format()` would
    try to parse every one of those braces as a placeholder too, raising
    `KeyError` on the first one it hits."""

    return get_templates().ask_user_form_guide.replace(
        "{known_metadata_fields}", build_known_metadata_fields_section()
    )


def build_history_section(history: Sequence[HistoryMessage]) -> str:
    """Build `{history_message}` - up to the last 15 raw messages (capped by
    the caller, see `app/api/v1/chat.py`), rendered as plain role: content
    lines. Independent of `confirmed_metadata`: this is the model's only
    view of the raw conversational flow (what was actually said, in order),
    since no other prompt section carries prior turns' text at all."""

    history_lines = (
        "\n".join(
            f"  - {_HISTORY_ROLE_LABELS.get(message.role, message.role)}: {message.content}"
            for message in history
        )
        or _NO_HISTORY
    )
    return get_templates().history_message.format(history_lines=history_lines)


def _recent_history_line(message: HistoryMessage) -> str:
    content = message.content
    if message.role == "ASSISTANT":
        # Only the topic matters here - citation markers, ask_user_form JSON
        # and amounts are noise, and a long answer is truncated to its opening.
        content = _JSON_BLOCK_PATTERN.sub("", content)
        content = _CITATION_MARKER_PATTERN.sub("", content)
        content = _AMOUNT_PATTERN.sub("...", content)
        content = content[:_RECENT_ASSISTANT_MAX_CHARS]
    content = " ".join(content.split())
    return f"- {_HISTORY_ROLE_LABELS.get(message.role, message.role)}: {content}"


def append_recent_history(query: str, history: Sequence[HistoryMessage]) -> str:
    """User prompt for the small agents: the message first, then the last
    `RECENT_HISTORY_LIMIT` history messages. The message comes first so the
    prompt is just the message itself when there is no history (first turn)."""

    recent = list(history)[-RECENT_HISTORY_LIMIT:]
    if not recent:
        return query
    history_lines = "\n".join(_recent_history_line(message) for message in recent)
    return (
        f"{query}\n\n"
        "Lịch sử hội thoại gần đây (chỉ dùng để hiểu tin nhắn nối tiếp ở trên):\n"
        f"{history_lines}"
    )


def _neutralize_frame_tags(text: str) -> str:
    """Escape any context-frame tag inside retrieved text (`<websearch>` ->
    `&lt;websearch&gt;`) so the only real frame tags are the template's own."""

    return _CONTEXT_FRAME_TAG_PATTERN.sub(r"&lt;\1\2&gt;", text)


def _page_suffix(chunk: RetrievedChunk) -> str:
    """`", tr. X"` / `", tr. X-Y"` when the chunk carries a real page number
    (PDF only - `page_start` stays `None` for HTML/DOCX/TXT/XLSX), else `""`.
    `content` already carries its heading prefix, so this is the only
    structural metadata the builder adds."""

    if chunk.page_start is None:
        return ""
    if chunk.page_end is not None and chunk.page_end != chunk.page_start:
        return f", tr. {chunk.page_start}-{chunk.page_end}"
    return f", tr. {chunk.page_start}"


def build_prepared_context_section(
    chunks: Sequence[RetrievedChunk],
    web_results: Sequence[WebSearchResult] = (),
    confirmed_metadata: Mapping[str, str] | None = None,
) -> str:
    """Build `{prepared_context}` - the `<academic_context>` block, chunks numbered
    to match the `[1][2]` citation indices the generation prompt asks the model to use,
    plus the `<current_time>` block (`build_current_time_section`).

    Each chunk's source is suffixed with its page number(s) when available
    (`_page_suffix`); `citation_rules.yaml` tells the LLM to cite them.

    `web_results` (WebSearchNode, only for sub-queries that found no chunk)
    fill the `<websearch>` block right below, numbered on from the last chunk
    so one `[n]` sequence covers both - `build_citations` resolves it the
    same way."""

    context_chunks = (
        "\n".join(
            f"  [{index}] ({_neutralize_frame_tags(chunk.source)}{_page_suffix(chunk)}) "
            f"{_neutralize_frame_tags(chunk.content)}"
            for index, chunk in enumerate(chunks, 1)
        )
        or _NO_RETRIEVED_CONTEXT
    )
    return get_templates().prepared_context.format(
        context_chunks=context_chunks,
        web_search_context=_build_web_search_context(web_results, first_index=len(chunks) + 1),
        current_time=build_current_time_section(confirmed_metadata),
    )


def build_current_time_section(confirmed_metadata: Mapping[str, str] | None = None) -> str:
    """`<current_time>`: today (ICT), the current term of the provisional school
    calendar and the student's own term number (from `khoa_nhap_hoc`) - one block,
    given to every chat flow."""

    today = now_ict().date()
    intake = parse_intake_year((confirmed_metadata or {}).get(INTAKE_FIELD), today)
    return get_templates().current_time.format(
        current_date=today.strftime("%d/%m/%Y"),
        current_term=describe_current_term(today),
        student_term=describe_student_term(today, intake),
    )


def append_current_time(prompt: str, confirmed_metadata: Mapping[str, str] | None = None) -> str:
    """For agents whose system prompt is static: the block goes into the message."""

    return f"{prompt}\n\n{build_current_time_section(confirmed_metadata)}"


def _build_web_search_context(web_results: Sequence[WebSearchResult], *, first_index: int) -> str:
    """The `<websearch>` block, or nothing at all - a turn without web
    results doesn't pay for the web-source rules in its prompt."""

    if not web_results:
        return ""
    entries = "\n".join(
        f"  [{index}] ({_page_label(result)}) {_neutralize_frame_tags(result.content)}"
        for index, result in enumerate(web_results, first_index)
    )
    return "\n" + get_templates().web_search_context.format(web_results=entries) + "\n"


def build_web_sources_section(sources: Sequence[WebSearchResult]) -> str:
    """The `<web_sources>` block of the ticket fallback, or nothing: official pages on the
    asked topic the web judge kept although their snippet lacks the answer."""

    if not sources:
        return ""
    entries = "\n".join(
        f"  - {_page_label(source)}: {_neutralize_frame_tags(source.content)}" for source in sources
    )
    return get_templates().web_candidate_sources.format(web_sources=entries)


def _page_label(result: WebSearchResult) -> str:
    year = f"; năm học {result.academic_year}" if result.academic_year else ""
    return f"{_neutralize_frame_tags(result.title)} — {_neutralize_frame_tags(result.url)}{year}"


def build_json_repair_prompt(
    previous_response: str,
    chunks: Sequence[RetrievedChunk],
    *,
    security: AcademicSecurityContext,
    confirmed_metadata: dict[str, str],
    web_results: Sequence[WebSearchResult] = (),
) -> str:
    """Build the standalone prompt for GenerationSynthesisNode's JSON-repair
    follow-up call (see `_repair_missing_ask_form` there) - no header needed,
    but `chunks` (the same `<academic_context>` the main prompt saw) IS
    needed: without it, `ask_user_form_guide`'s "copy the branch label
    verbatim from the text" rule has no text to copy from, and the repaired
    field falls back to `options: null` (free-text) even when the source
    document actually lists a finite set of options. `security`/
    `confirmed_metadata` are ALSO needed - without `<declared_attributes>`
    this call has no way to know a field was already answered in an earlier
    turn, and can re-ask it (observed live: repair re-emitted a form for
    `he_dao_tao` already present in `confirmed_metadata`). Reuses
    `build_prepared_context_section`/`build_metadata_section` so these blocks
    are built identically to the main prompt's."""

    templates = get_templates()
    return templates.json_repair.format(
        previous_response=previous_response,
        academic_context=build_prepared_context_section(chunks, web_results, confirmed_metadata),
        academic_metadata=build_metadata_section(security, confirmed_metadata),
        ask_user_form_guide=build_ask_user_form_guide(),
    )


def render_resolved_user_query(user_query: str, resolved_query: str | None) -> str:
    """Fills `{user_query}` in `chat_academic_advisory.yaml`/`json_repair.yaml`.

    When `resolved_query` (QueryTransformationNode's self-contained rewrite
    of this turn, see `extract_standalone_question`) differs from the raw
    `user_query`, both are shown: the rewrite first, since it is what the
    model should treat as the actual question (carrying forward whatever
    khóa/hệ/khối a follow-up like "còn ĐHCQ ngành CN thì sao" left implicit),
    then the verbatim text the student typed, so nothing said this turn is
    lost either. Identical strings (first turn, or a question that was
    already self-contained) render as plain `user_query` - no placeholder
    change needed in the YAML for either case."""

    resolved = (resolved_query or "").strip()
    if not resolved or resolved == user_query.strip():
        return user_query
    return f'{resolved}\n\n(Nguyên văn người dùng vừa nhắn ở lượt này: "{user_query}")'


def build_task_2_section() -> str:
    """Build `{task_2}` - `task_2.yaml` embeds `ask_user_form_guide.yaml`. Answers to a
    clarification panel arrive structured and are already folded into
    `confirmed_metadata`, so the model never re-reads a pending form."""

    return get_templates().task_2.format(ask_user_form_guide=build_ask_user_form_guide())


def build_calculation_results_section(titles: Sequence[str]) -> str:
    """`{calculation_results}`: titles only - never a number - of the calculations
    already shown above the advisory answer."""

    if not titles:
        return NOTHING
    return "\n".join(f"- {title} (đã tính ở trên)" for title in titles)
