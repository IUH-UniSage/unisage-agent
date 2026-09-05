"""Node 12: `GenerationSynthesisNode` (T1.11) — streaming fan-in + JSON
`pending_clarification` extraction.

After the full response text has streamed, a deterministic (no extra LLM
call) step extracts the LAST ```json fenced block in the response and, if
it has shape `{"type": "ask_user_form", "fields": [...]}`, rebuilds it into
a `PendingClarification` — keeping `retry_count` if the field set is
unchanged from the previous turn's pending clarification, resetting to 0 if
it's a new field set. This matches
missing_metadata_clarification_design.md section 5's "điểm phát hiện != điểm
quay lại": `origin_node` is passed in by the caller (always
`QueryTransformationNode` for the Type B flow this phase implements), not
derived from where the JSON was found.
"""

import json
import re
from dataclasses import dataclass
from typing import Any

from pydantic_ai import Agent
from pydantic_ai.models import Model

from app.core.graph_trace import GraphTrace
from app.graph.streaming import TokenSink, stream_agent_text
from app.rag.prompting.loader import build_system_prompt
from app.schemas.clarification import PendingClarification
from app.schemas.retrieval import RetrievedChunk
from app.schemas.security import AcademicSecurityContext

_JSON_BLOCK_PATTERN = re.compile(r"```json\s*(\{.*?\})\s*```", re.DOTALL)


def build_generation_agent(model: Model | str) -> Agent[None, str]:
    return Agent(model=model)


@dataclass(frozen=True)
class GenerationResult:
    response_text: str
    pending_clarification: PendingClarification | None


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
    system_prompt = build_system_prompt(
        security=security,
        confirmed_metadata=confirmed_metadata,
        chunks=chunks,
        pending_clarification=previous_pending,
    )
    full_prompt = f"{system_prompt}\n\nCâu hỏi của sinh viên: {user_query}"
    trace.prompt("12_GenerationSynthesisNode", full_prompt)
    full_text = await stream_agent_text(agent, full_prompt, token_sink)
    new_pending = collect_pending_clarification(
        full_text, origin_node=origin_node, previous=previous_pending
    )
    return GenerationResult(response_text=full_text, pending_clarification=new_pending)


def _extract_last_json_block(text: str) -> dict[str, Any] | None:
    matches = list(_JSON_BLOCK_PATTERN.finditer(text))
    if not matches:
        return None
    try:
        parsed = json.loads(matches[-1].group(1))
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def collect_pending_clarification(
    full_text: str,
    *,
    origin_node: str,
    previous: PendingClarification | None,
) -> PendingClarification | None:
    parsed = _extract_last_json_block(full_text)
    if parsed is None or parsed.get("type") != "ask_user_form":
        return None

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
