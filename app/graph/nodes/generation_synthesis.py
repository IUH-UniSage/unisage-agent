"""Generation synthesis node - streams the final response and extracts any
`pending_clarification`/`confirmed_metadata` updates embedded in it.

After the full response text has streamed, a deterministic (no extra LLM
call) step scans every ```json fenced block in the response for two shapes:

- `{"type": "ask_user_form", "fields": [...]}` - rebuilt into a
  `PendingClarification` (last such block wins if the model emits more than
  one) — keeping `retry_count` if the field set is unchanged from the
  previous turn's pending clarification, resetting to 0 if it's a new field
  set. `origin_node` (where the clarification should resume) is passed in
  by the caller rather than derived from where the JSON was found, since
  the detection point and the resume point can differ.
- `{"type": "confirmed_metadata", "fields": {...}}` - a fallback for when
  the Clarification Guard's deterministic matcher (security_context.py)
  couldn't map a free-form reply to a pending option itself; the model
  reads the same `<missing_metadata_to_confirm>` block and, if it can
  confidently map the user's reply to one of the listed option ids, says so
  here. Only fields the model was actually asked about (i.e. present in the
  turn's pending clarification) are accepted - anything else is dropped, so
  a model that misreads the instruction can't inject arbitrary metadata.
"""

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from pydantic_ai import Agent
from pydantic_ai.models import Model

from app.core.config import settings
from app.core.graph_trace import GraphTrace
from app.core.model_registry import CredentialConfig
from app.graph.streaming import TokenSink, stream_agent_text
from app.rag.prompting import (
    build_json_repair_prompt,
    build_multi_intent_prompt,
    build_system_prompt,
)
from app.rag.prompting.builder import build_metadata_section, build_prepared_context_section
from app.schemas.chat_history import HistoryMessage
from app.schemas.clarification import PendingClarification
from app.schemas.intent import ClassifiedTask
from app.schemas.retrieval import RetrievedChunk
from app.schemas.security import AcademicSecurityContext

_JSON_BLOCK_PATTERN = re.compile(r"```json\s*(\{.*?\})\s*```", re.DOTALL)
_SUB_QUERY_ID_PATTERN = re.compile(r"SQ(\d+)")
_CLARIFICATION_PHRASE_PATTERN = re.compile(
    r"cho\s+.{0,15}?bi[eế]t\b"
    r"|cung\s+c[aấ]p\b.{0,25}?th[oô]ng\s+tin\b"
    r"|xin\s+.{0,15}?bi[eế]t\b",
    re.IGNORECASE,
)

_OFFER_CONDITION_PATTERN = re.compile(
    r"n[eế]u\s+(b[aạ]n|em)?\s*(c[aầ]n|mu[oố]n|c[oó]\s+nhu\s+c[aầ]u)\b", re.IGNORECASE
)

# Splits on sentence-ending punctuation followed by whitespace - good enough
# for finding "which sentence contains this match", not a general-purpose
# sentence tokenizer.
_SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[.!?])\s+")


def build_generation_agent(model: Model | str) -> Agent[None, str]:
    return Agent(model=model)


def _is_conditional_offer(full_text: str) -> bool:
    """True if the sentence containing `_CLARIFICATION_PHRASE_PATTERN`'s
    match - OR the sentence immediately before it - contains
    `_OFFER_CONDITION_PATTERN` anywhere in it (either order, see that
    pattern's comment).

    Checking only the matching sentence itself is NOT enough: Vietnamese
    routinely splits a conditional offer across two sentences - "Nếu bạn
    cần thêm thông tin cụ thể hơn (...), mình có thể giúp bạn tìm kiếm
    thông tin đó. Hãy cho mình biết nhé!" - where the `_CLARIFICATION_PHRASE
    _PATTERN` match ("cho mình biết") lands in the SECOND sentence, which on
    its own contains no "nếu" at all. Observed live (twice): a
    single-sentence-only version of this check missed exactly this split and
    let a hallucinated `ask_user_form` for an unrelated topic through. Only
    looking one sentence back keeps this from also swallowing a genuine
    question that happens to follow an unrelated "Nếu..." sentence earlier
    in a longer reply."""

    without_json = _JSON_BLOCK_PATTERN.sub("", full_text)
    sentences = _SENTENCE_SPLIT_PATTERN.split(without_json)
    for index, sentence in enumerate(sentences):
        if not _CLARIFICATION_PHRASE_PATTERN.search(sentence):
            continue
        window = sentences[max(0, index - 1) : index + 1]
        return any(_OFFER_CONDITION_PATTERN.search(candidate) for candidate in window)
    return False


async def _repair_missing_ask_form(
    agent: Agent[None, str],
    full_text: str,
    *,
    chunks: list[RetrievedChunk],
    security: AcademicSecurityContext,
    confirmed_metadata: dict[str, str],
    token_sink: TokenSink,
) -> str:
    """Second-chance fixup for a known model failure mode: the response reads
    like a clarification request in prose but the mandatory
    ```json ask_user_form``` block never showed up (see chat_academic_advisory.yaml
    / task_2.yaml's own instructions - the model sometimes just doesn't
    follow them). Only fires when heuristically worth the extra LLM call;
    a false negative here just means the old prose-only behavior, so the
    heuristic is fine erring toward "didn't detect it".

    Requires `_CLARIFICATION_PHRASE_PATTERN` to match AND that match's
    sentence to NOT be a conditional offer (`_is_conditional_offer`) - the
    phrase alone also matches a conditional closing offer ("Nếu bạn cần
    thêm thông tin..., vui lòng cho biết thêm thông tin."), which is exactly
    what produced a hallucinated `ask_user_form` for an unrelated topic in a
    live observation. `security`/`confirmed_metadata` are threaded through
    to the repair prompt so it can see `<student_declared_attributes>` and
    skip re-asking a field already answered in an earlier turn.

    Streams nothing to `token_sink` until the repair call itself has
    produced a valid JSON block - a partial/garbage repair attempt (or a
    literal "NONE") never reaches the client mid-stream.
    """

    if _extract_json_blocks(full_text):
        return full_text
    if not _CLARIFICATION_PHRASE_PATTERN.search(full_text):
        return full_text
    if _is_conditional_offer(full_text):
        return full_text

    repair_prompt = build_json_repair_prompt(
        full_text, chunks, security=security, confirmed_metadata=confirmed_metadata
    )

    async def _capture_sink(_token: str) -> None:
        return None

    repaired = await stream_agent_text(agent, repair_prompt, _capture_sink)
    match = _JSON_BLOCK_PATTERN.search(repaired)
    if match is None:
        return full_text

    appended = f"\n\n```json\n{match.group(1)}\n```"
    await token_sink(appended)
    return full_text + appended


@dataclass(frozen=True)
class GenerationResult:
    response_text: str
    pending_clarification: PendingClarification | None
    confirmed_metadata: dict[str, str]


async def run_generation_synthesis(
    agent: Agent[None, str],
    *,
    user_query: str,
    resolved_query: str | None = None,
    security: AcademicSecurityContext,
    confirmed_metadata: dict[str, str],
    chunks: list[RetrievedChunk],
    previous_pending: PendingClarification | None,
    origin_node: str,
    token_sink: TokenSink,
    trace: GraphTrace,
    history: Sequence[HistoryMessage] = (),
    advisory_tasks: Sequence[ClassifiedTask] | None = None,
    sub_queries: Sequence[str] | None = None,
    purpose: str | None = None,
    credential: CredentialConfig | None = None,
    snapshot_version: int | None = None,
) -> GenerationResult:
    # A single HyDE question uses the advisory frame; several sub-queries
    # (a decomposed comparison, or several different questions in one
    # message) use the multi-intent frame instead.
    if sub_queries and len(sub_queries) > 1:
        full_prompt = build_multi_intent_prompt(
            user_query=user_query,
            sub_queries=sub_queries,
            security=security,
            confirmed_metadata=confirmed_metadata,
            chunks=chunks,
            pending_clarification=previous_pending,
            history=history,
        )
    else:
        full_prompt = build_system_prompt(
            user_query=user_query,
            resolved_query=resolved_query,
            security=security,
            confirmed_metadata=confirmed_metadata,
            chunks=chunks,
            pending_clarification=previous_pending,
            history=history,
        )
    # Only the two per-request blocks are worth dumping - the rest of the
    # prompt is static YAML that can be read from the templates directly.
    trace.prompt(
        "10_GenerationSynthesisNode",
        f"{build_metadata_section(security, confirmed_metadata)}\n"
        f"{build_prepared_context_section(chunks)}",
    )
    full_text = await stream_agent_text(
        agent,
        full_prompt,
        token_sink,
        purpose=purpose,
        credential=credential,
        snapshot_version=snapshot_version,
        agent_factory=build_generation_agent,
    )
    if settings.CHAT_ALLOW_REPAIR_JSON:
        full_text = await _repair_missing_ask_form(
            agent,
            full_text,
            chunks=chunks,
            security=security,
            confirmed_metadata=confirmed_metadata,
            token_sink=token_sink,
        )
    confirmed_updates = collect_confirmed_metadata_updates(full_text, previous=previous_pending)
    updated_confirmed_metadata = (
        {**confirmed_metadata, **confirmed_updates} if confirmed_updates else confirmed_metadata
    )
    new_pending = collect_pending_clarification(
        full_text,
        origin_node=origin_node,
        previous=previous_pending,
        user_query=user_query,
        confirmed_metadata=updated_confirmed_metadata,
        origin_tasks=list(advisory_tasks) if advisory_tasks else None,
        sub_query_count=len(sub_queries) if sub_queries else 1,
    )
    if new_pending is None:
        new_pending = _carry_forward_unanswered(
            previous_pending, confirmed_metadata=updated_confirmed_metadata
        )
    return GenerationResult(
        response_text=full_text,
        pending_clarification=new_pending,
        confirmed_metadata=updated_confirmed_metadata,
    )


def _carry_forward_unanswered(
    previous: PendingClarification | None,
    *,
    confirmed_metadata: dict[str, str],
) -> PendingClarification | None:
    """Keep a round alive when this turn's response dropped it silently.

    A round only ends when its fields are answered - but the round lives in
    the model's own JSON output, so if the model simply forgets to re-emit
    `ask_user_form` while fields are still unanswered, the round (and with
    it `original_query`) evaporates: the next turn sees no pending round,
    treats the student's reply as a brand-new question, and the topic that
    started it all (e.g. "học phí") is gone. Re-deriving from
    `confirmed_metadata` instead of trusting the model's silence keeps the
    state machine's own bookkeeping authoritative: fields answered by now
    drop off, whatever is left stays pending.
    """

    if previous is None:
        return None
    labels = previous.option_labels or [None] * len(previous.missing_fields)
    remaining = [
        (field, options, label)
        for field, options, label in zip(
            previous.missing_fields,
            previous.options,
            list(labels[: len(previous.missing_fields)])
            + [None] * max(0, len(previous.missing_fields) - len(labels)),
            strict=True,
        )
        if field not in confirmed_metadata and options is not None
    ]
    if not remaining:
        return None
    return previous.model_copy(
        update={
            "missing_fields": [field for field, _options, _label in remaining],
            "options": [options for _field, options, _label in remaining],
            "option_labels": [label for _field, _options, label in remaining],
        }
    )


def _extract_json_blocks(text: str) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    for match in _JSON_BLOCK_PATTERN.finditer(text):
        try:
            parsed = json.loads(match.group(1))
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            blocks.append(parsed)
    return blocks


def collect_confirmed_metadata_updates(
    full_text: str,
    *,
    previous: PendingClarification | None,
) -> dict[str, str]:
    """Fallback confirmation from the model's own reading of the reply -
    only accepts fields the user was actually asked about this turn."""

    if previous is None:
        return {}
    allowed_fields = set(previous.missing_fields)

    merged: dict[str, str] = {}
    for block in _extract_json_blocks(full_text):
        if block.get("type") != "confirmed_metadata":
            continue
        fields = block.get("fields")
        if not isinstance(fields, dict):
            continue
        for field, value in fields.items():
            if isinstance(field, str) and isinstance(value, str) and field in allowed_fields:
                merged[field] = value
    return merged


def collect_pending_clarification(
    full_text: str,
    *,
    origin_node: str,
    previous: PendingClarification | None,
    user_query: str = "",
    confirmed_metadata: dict[str, str] | None = None,
    origin_tasks: list[ClassifiedTask] | None = None,
    sub_query_count: int = 1,
) -> PendingClarification | None:
    """Rebuild the pending round from the model's `ask_user_form` block.

    Two classes of field are dropped here rather than trusted, because both
    produce a round that can never close:

    - **Already answered** - a field whose value is in `confirmed_metadata`.
      The prompt tells the model not to re-ask these, but when it does
      anyway, persisting it strands the conversation: the student already
      said "chính quy", so nothing they type next will read as new
      information. (Asking a NARROWER follow-up is still fine - that carries
      a different field name, see task_1.yaml "HỎI SÂU THÊM MỘT CẤP".)
    - **Free-text (`options: null`)** - the Clarification Guard matches
      against option ids/labels, so a typed value can never be matched, and
      only the model volunteering a `confirmed_metadata` block would ever
      record it. Values like a GPA are self-declared numbers the assistant
      is forbidden to draw conclusions from anyway (task_1.yaml, "CẤM TỰ
      KẾT LUẬN TÌNH TRẠNG HỌC VỤ"), so they are stated as thresholds in
      prose for the student to check themselves - never collected as form
      state.

    A third case is dropped at the very top, before any of the above: the
    whole block is discarded if its lead-in sentence is a conditional offer
    (`_is_conditional_offer`) - same check `_repair_missing_ask_form` uses,
    but needed here too because this function also runs when the model
    attaches `ask_user_form` directly in its PRIMARY response (no repair
    call involved at all) - observed live: a fully-answered, unrelated
    question followed by "Nếu bạn có nhu cầu tìm hiểu thêm..., vui lòng cho
    biết nhé!" plus a populated (hallucinated) form, all in one pass.
    """
    ask_form_blocks = [
        block for block in _extract_json_blocks(full_text) if block.get("type") == "ask_user_form"
    ]
    if not ask_form_blocks:
        return None
    if _is_conditional_offer(full_text):
        return None
    parsed = ask_form_blocks[-1]

    fields_spec = parsed.get("fields") or []
    missing_fields: list[str] = []
    options: list[list[str] | None] = []
    option_labels: list[list[str] | None] = []
    for field_spec in fields_spec:
        missing_fields.append(field_spec["field"])
        raw_options = field_spec.get("options")
        options.append([option["id"] for option in raw_options] if raw_options else None)
        # Keep the labels too, not just the ids: the user sees (and often
        # types back) the label - "Công nghệ Thông tin", not "cntt" - so the
        # deterministic Guard needs both to resolve a hand-typed reply
        # without burning a retry and falling through to the LLM.
        option_labels.append(
            [str(option.get("label") or option["id"]) for option in raw_options]
            if raw_options
            else None
        )

    already_confirmed = confirmed_metadata or {}
    keep = [
        index
        for index, field in enumerate(missing_fields)
        if field not in already_confirmed and options[index] is not None
    ]
    missing_fields = [missing_fields[index] for index in keep]
    options = [options[index] for index in keep]
    option_labels = [option_labels[index] for index in keep]

    if not missing_fields:
        # A known model slip: it sometimes appends `{"type": "ask_user_form",
        # "fields": []}` after a complete answer (often triggered by an
        # innocuous closing courtesy line like "let me know if you need
        # anything else", misread as a clarification request). An empty
        # `fields` array means nothing to ask - treat exactly like no block
        # at all, rather than persisting a hollow PendingClarification that
        # would make every future turn think a round is still open.
        return None

    retry_count = (
        previous.retry_count
        if previous is not None and previous.missing_fields == missing_fields
        else 0
    )
    original_query = previous.original_query if previous is not None else user_query

    # Only trust an ask_user_form's sub_query_id when it names one of THIS
    # turn's sub-queries - a stale or made-up SQk would misroute the resume.
    raw_sub_query_id = parsed.get("sub_query_id")
    sub_query_match = (
        _SUB_QUERY_ID_PATTERN.fullmatch(raw_sub_query_id)
        if isinstance(raw_sub_query_id, str)
        else None
    )
    pending_sub_query_id = (
        raw_sub_query_id
        if sub_query_match and 1 <= int(sub_query_match.group(1)) <= sub_query_count
        else None
    )

    return PendingClarification(
        origin_node=origin_node,
        missing_fields=missing_fields,
        options=options,
        option_labels=option_labels,
        retry_count=retry_count,
        original_query=original_query,
        pending_sub_query_id=pending_sub_query_id,
        origin_tasks=origin_tasks,
    )
