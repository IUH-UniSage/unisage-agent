"""Prompt building functions - assemble YAML template sections with graph state."""

from __future__ import annotations

import json
import re
from collections.abc import Sequence

from app.core.timezone import now_ict
from app.schemas.chat_history import HistoryMessage
from app.schemas.clarification import PendingClarification
from app.schemas.retrieval import RetrievedChunk
from app.schemas.security import AcademicSecurityContext

from .loader import get_known_metadata_fields, get_templates

_NO_DECLARED_ATTRIBUTES = "Chưa có thuộc tính nào được sinh viên xác nhận."
_NO_DEPARTMENT_ACCESS = "    - (không có phòng ban nào được cấp quyền)"
_NO_RETRIEVED_CONTEXT = "  (không có tài liệu liên quan)"
_NO_HISTORY = "  (đây là lượt đầu tiên, chưa có lịch sử)"
_HISTORY_ROLE_LABELS = {"USER": "Người dùng", "ASSISTANT": "Trợ lý"}
NO_PENDING_CLARIFICATION = "Không có"

RECENT_HISTORY_LIMIT = 4
_RECENT_ASSISTANT_MAX_CHARS = 300
_CITATION_MARKER_PATTERN = re.compile(r"\[\d+(?:\s*,\s*\d+)*\]")
_JSON_BLOCK_PATTERN = re.compile(r"```json.*?```", re.DOTALL)
# Amounts like "60.000.000" - the previous turn's figures belong to a
# different question, and would otherwise be copied into this turn's output.
_AMOUNT_PATTERN = re.compile(r"\d{1,3}(?:[.,]\d{3})+")


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


def build_known_metadata_fields_section() -> str:
    """Render `known_metadata_fields.json` as the markdown table
    `ask_user_form_guide.yaml` embeds via `{known_metadata_fields}` - see
    that file for the field-name-reuse rule this backs. Static data (doesn't
    vary per conversation), but kept as a JSON file rather than hardcoded in
    the YAML prose so it stays easy to extend/audit as its own artifact."""

    rows = "\n".join(
        f"    | {entry['concept']} | `{entry['field']}` | {entry['description']} |"
        for entry in get_known_metadata_fields()
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


def _page_suffix(chunk: RetrievedChunk) -> str:
    """`", tr. X"` / `", tr. X-Y"` when the chunk carries a real page number
    (PDF only - `page_start` stays `None` for HTML/DOCX/TXT/XLSX), else `""`.
    `content` already carries its own heading prefix (chunker-side, Phase 3),
    so this is the only structural metadata the builder itself needs to add."""

    if chunk.page_start is None:
        return ""
    if chunk.page_end is not None and chunk.page_end != chunk.page_start:
        return f", tr. {chunk.page_start}-{chunk.page_end}"
    return f", tr. {chunk.page_start}"


def build_prepared_context_section(chunks: Sequence[RetrievedChunk]) -> str:
    """Build `{prepared_context}` - the `<academic_context>` block, chunks numbered
    to match the `[1][2]` citation indices the generation prompt asks the model to use,
    plus the `<current_date>` block (today, ICT/GMT+7) so the model has a real-world
    time anchor for questions like "học phí năm 2025-2026" without nêu rõ mốc thời gian,
    or for checking whether a document's stated effective date has passed.

    Each chunk's source is suffixed with its page number(s) when available
    (`_page_suffix`) - this is the mechanism that finally gets `page_start`/
    `page_end` in front of the LLM (Phase 6's main goal); `citation_rules.yaml`
    (Task 6.5) is what then instructs the LLM to copy it into its answer."""

    context_chunks = (
        "\n".join(
            f"  [{index}] ({chunk.source}{_page_suffix(chunk)}) {chunk.content}"
            for index, chunk in enumerate(chunks, 1)
        )
        or _NO_RETRIEVED_CONTEXT
    )
    current_date = now_ict().strftime("%d/%m/%Y")
    return get_templates().prepared_context.format(
        context_chunks=context_chunks, current_date=current_date
    )


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


def build_json_repair_prompt(
    previous_response: str,
    chunks: Sequence[RetrievedChunk],
    *,
    security: AcademicSecurityContext,
    confirmed_metadata: dict[str, str],
) -> str:
    """Build the standalone prompt for GenerationSynthesisNode's JSON-repair
    follow-up call (see `_repair_missing_ask_form` there) - no header needed,
    but `chunks` (the same `<academic_context>` the main prompt saw) IS
    needed: without it, `ask_user_form_guide`'s "copy the branch label
    verbatim from the text" rule has no text to copy from, and the repaired
    field falls back to `options: null` (free-text) even when the source
    document actually lists a finite set of options. `security`/
    `confirmed_metadata` are ALSO needed - without `<student_declared_attributes>`
    this call has no way to know a field was already answered in an earlier
    turn, and can re-ask it (observed live: repair re-emitted a form for
    `he_dao_tao` already present in `confirmed_metadata`). Reuses
    `build_prepared_context_section`/`build_metadata_section` so these blocks
    are built identically to the main prompt's."""

    templates = get_templates()
    return templates.json_repair.format(
        previous_response=previous_response,
        academic_context=build_prepared_context_section(chunks),
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


def build_task_2_section(pending: PendingClarification | None) -> str:
    """Build `{task_2}` - nested format: `task_2.yaml` embeds the static
    `ask_user_form_guide.yaml`/`confirmed_metadata_guide.yaml` plus the
    dynamic missing-metadata block."""

    templates = get_templates()
    return templates.task_2.format(
        missing_metadata_to_confirm=build_missing_metadata_block(pending),
        ask_user_form_guide=build_ask_user_form_guide(),
        confirmed_metadata_guide=templates.confirmed_metadata_guide,
    )
