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

Known model failure mode this node also guards against: sometimes the model
asks for a missing attribute in prose (any phrasing, any punctuation) but
forgets the mandatory ```json ask_user_form``` block the prompt requires.
`_repair_missing_ask_form` is a second, small follow-up LLM call - fired only
when a cheap regex heuristic thinks the response reads like a clarification
request AND no JSON block is present at all - that asks the model to look at
its own just-written response and emit only the JSON block it forgot. This
keeps the primary response streaming in true real time (the heuristic/repair
only run after the main stream is done) at the cost of one extra call in the
rare case a violation is actually detected.
"""

import json
import re
from dataclasses import dataclass
from typing import Any

from pydantic_ai import Agent
from pydantic_ai.models import Model

from app.core.graph_trace import GraphTrace
from app.graph.streaming import TokenSink, stream_agent_text
from app.rag.prompting import build_json_repair_prompt, build_system_prompt
from app.schemas.clarification import PendingClarification
from app.schemas.retrieval import RetrievedChunk
from app.schemas.security import AcademicSecurityContext

_JSON_BLOCK_PATTERN = re.compile(r"```json\s*(\{.*?\})\s*```", re.DOTALL)

# Heuristic-only (no LLM call): does the response READ like it's asking the
# user to supply a missing attribute, in prose, regardless of punctuation?
# Cheap to evaluate on every turn - only turns that both match this AND have
# no JSON block at all pay for the repair call below.
_CLARIFICATION_PHRASE_PATTERN = re.compile(
    r"cho\s+(t[oô]i|m[ìi]nh)?\s*bi[eế]t\b"
    r"|vui\s+l[oò]ng\s+cho\s+bi[eế]t\b"
    r"|cung\s+c[aấ]p\s+(th[eê]m\s+)?th[oô]ng\s+tin\b"
    r"|xin\s+(h[aã]y\s+)?cho\s+bi[eế]t\b",
    re.IGNORECASE,
)


def build_generation_agent(model: Model | str) -> Agent[None, str]:
    return Agent(model=model)


async def _repair_missing_ask_form(
    agent: Agent[None, str],
    full_text: str,
    *,
    chunks: list[RetrievedChunk],
    token_sink: TokenSink,
    trace: GraphTrace,
) -> str:
    """Second-chance fixup for a known model failure mode: the response reads
    like a clarification request in prose but the mandatory
    ```json ask_user_form``` block never showed up (see chat_academic_advisory.yaml
    / task_2.yaml's own instructions - the model sometimes just doesn't
    follow them). Only fires when heuristically worth the extra LLM call;
    a false negative here just means the old prose-only behavior, so the
    heuristic is fine erring toward "didn't detect it".

    Streams nothing to `token_sink` until the repair call itself has
    produced a valid JSON block - a partial/garbage repair attempt (or a
    literal "NONE") never reaches the client mid-stream.
    """

    if _extract_json_blocks(full_text):
        return full_text
    if not _CLARIFICATION_PHRASE_PATTERN.search(full_text):
        return full_text

    repair_prompt = build_json_repair_prompt(full_text, chunks)
    trace.prompt("12b_GenerationSynthesisNode_JsonRepair", repair_prompt)

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
    security: AcademicSecurityContext,
    confirmed_metadata: dict[str, str],
    chunks: list[RetrievedChunk],
    previous_pending: PendingClarification | None,
    origin_node: str,
    token_sink: TokenSink,
    trace: GraphTrace,
) -> GenerationResult:
    full_prompt = build_system_prompt(
        user_query=user_query,
        security=security,
        confirmed_metadata=confirmed_metadata,
        chunks=chunks,
        pending_clarification=previous_pending,
    )
    trace.prompt("12_GenerationSynthesisNode", full_prompt)
    full_text = await stream_agent_text(agent, full_prompt, token_sink)
    full_text = await _repair_missing_ask_form(
        agent, full_text, chunks=chunks, token_sink=token_sink, trace=trace
    )
    new_pending = collect_pending_clarification(
        full_text, origin_node=origin_node, previous=previous_pending
    )
    confirmed_updates = collect_confirmed_metadata_updates(full_text, previous=previous_pending)
    updated_confirmed_metadata = (
        {**confirmed_metadata, **confirmed_updates} if confirmed_updates else confirmed_metadata
    )
    return GenerationResult(
        response_text=full_text,
        pending_clarification=new_pending,
        confirmed_metadata=updated_confirmed_metadata,
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
) -> PendingClarification | None:
    ask_form_blocks = [
        block for block in _extract_json_blocks(full_text) if block.get("type") == "ask_user_form"
    ]
    if not ask_form_blocks:
        return None
    parsed = ask_form_blocks[-1]

    fields_spec = parsed.get("fields") or []
    missing_fields: list[str] = []
    options: list[list[str] | None] = []
    for field_spec in fields_spec:
        missing_fields.append(field_spec["field"])
        raw_options = field_spec.get("options")
        options.append([option["id"] for option in raw_options] if raw_options else None)

    retry_count = (
        previous.retry_count
        if previous is not None and previous.missing_fields == missing_fields
        else 0
    )

    return PendingClarification(
        origin_node=origin_node,
        missing_fields=missing_fields,
        options=options,
        retry_count=retry_count,
    )
