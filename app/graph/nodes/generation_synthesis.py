"""Generation synthesis node - streams the final answer.

The prompt may make the model end with a ```json {"type": "ask_user_form"}```
block when it needs more information. `FenceRedactor` keeps every such block
out of the stream and out of `response_text`; the captured forms are returned
in `ask_forms` and become choice questions of the clarification panel
(app/graph/clarification_round.py). A prose-only clarification request gets a
second-chance repair call that supplies the missing block for the graph.
"""

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from pydantic_ai import Agent
from pydantic_ai.models import Model

from app.core.config import settings
from app.core.observability.graph_trace import GraphTrace
from app.core.registry.model_registry import CredentialConfig
from app.graph.fence_redactor import FenceRedactor
from app.graph.streaming import (
    AttemptRecorder,
    BudgetContext,
    FailoverCallback,
    TokenSink,
    generation_model_settings,
    stream_agent_text,
)
from app.rag.prompting import (
    build_json_repair_prompt,
    build_multi_intent_prompt,
    build_system_prompt,
)
from app.rag.prompting.builder import build_metadata_section, build_prepared_context_section
from app.schemas.chat_history import HistoryMessage
from app.schemas.retrieval import RetrievedChunk
from app.schemas.security import AcademicSecurityContext
from app.schemas.web_search import WebSearchResult

_JSON_BLOCK_PATTERN = re.compile(r"```json\s*(\{.*?\})\s*```", re.DOTALL)
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
    return Agent(model=model, model_settings=generation_model_settings(model))


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
    web_results: Sequence[WebSearchResult] = (),
    security: AcademicSecurityContext,
    confirmed_metadata: dict[str, str],
    token_sink: TokenSink,
    credential: CredentialConfig | None = None,
    on_attempt: AttemptRecorder | None = None,
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

    `credential`/`on_attempt`: this repair
    call has no failover wiring of its own (no retry, no `purpose`), so
    `credential` is whatever the caller's primary call started with - if THAT
    call failed over mid-flight, this snapshot is stale (a pre-existing gap,
    not introduced by usage recording: the repair call already reused the
    same possibly-stale local `agent` variable before this field existed).
    """

    if _extract_json_blocks(full_text):
        return full_text
    if not _CLARIFICATION_PHRASE_PATTERN.search(full_text):
        return full_text
    if _is_conditional_offer(full_text):
        return full_text

    repair_prompt = build_json_repair_prompt(
        full_text,
        chunks,
        security=security,
        confirmed_metadata=confirmed_metadata,
        web_results=web_results,
    )

    async def _capture_sink(_token: str) -> None:
        return None

    repaired = await stream_agent_text(
        agent, repair_prompt, _capture_sink, credential=credential, on_attempt=on_attempt
    )
    match = _JSON_BLOCK_PATTERN.search(repaired)
    if match is None:
        return full_text

    appended = f"\n\n```json\n{match.group(1)}\n```"
    await token_sink(appended)
    return full_text + appended


@dataclass(frozen=True)
class GenerationResult:
    # What the student saw: the answer with every ask_user_form/confirmed_metadata
    # block filtered out (FenceRedactor) - also what gets persisted in Java.
    response_text: str
    # The ```json ask_user_form blocks the model emitted (or the repair call added).
    ask_forms: tuple[dict[str, Any], ...] = ()


async def run_generation_synthesis(
    agent: Agent[None, str],
    *,
    user_query: str,
    resolved_query: str | None = None,
    security: AcademicSecurityContext,
    confirmed_metadata: dict[str, str],
    chunks: list[RetrievedChunk],
    web_results: Sequence[WebSearchResult] = (),
    token_sink: TokenSink,
    trace: GraphTrace,
    history: Sequence[HistoryMessage] = (),
    sub_queries: Sequence[str] | None = None,
    calculation_titles: Sequence[str] = (),
    purpose: str | None = None,
    credential: CredentialConfig | None = None,
    snapshot_version: int | None = None,
    on_failover: FailoverCallback | None = None,
    on_attempt: AttemptRecorder | None = None,
    budget: BudgetContext | None = None,
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
            web_results=web_results,
            history=history,
            calculation_titles=calculation_titles,
        )
    else:
        full_prompt = build_system_prompt(
            user_query=user_query,
            resolved_query=resolved_query,
            security=security,
            confirmed_metadata=confirmed_metadata,
            chunks=chunks,
            web_results=web_results,
            history=history,
            calculation_titles=calculation_titles,
        )
    # Only the two per-request blocks are worth dumping - the rest of the
    # prompt is static YAML that can be read from the templates directly.
    trace.prompt(
        "10_GenerationSynthesisNode",
        f"{build_metadata_section(security, confirmed_metadata)}\n"
        f"{build_prepared_context_section(chunks, web_results)}",
    )
    redactor = FenceRedactor()
    visible: list[str] = []

    async def redacting_sink(chunk: str) -> None:
        shown = redactor.feed(chunk)
        if shown:
            visible.append(shown)
            await token_sink(shown)

    full_text = await stream_agent_text(
        agent,
        full_prompt,
        redacting_sink,
        purpose=purpose,
        credential=credential,
        snapshot_version=snapshot_version,
        agent_factory=build_generation_agent,
        on_failover=on_failover,
        on_attempt=on_attempt,
        budget=budget,
    )
    tail = redactor.finish()
    if tail:
        visible.append(tail)
        await token_sink(tail)
    if settings.CHAT_ALLOW_REPAIR_JSON:
        # The repaired block is for the graph only - it never goes to the client.
        repaired = await _repair_missing_ask_form(
            agent,
            full_text,
            chunks=chunks,
            web_results=web_results,
            security=security,
            confirmed_metadata=confirmed_metadata,
            token_sink=_discard,
            credential=credential,
            on_attempt=on_attempt,
        )
        if repaired != full_text:
            redactor.captured.extend(
                block for block in _extract_json_blocks(repaired[len(full_text) :])
            )
            full_text = repaired
    return GenerationResult(
        response_text="".join(visible).rstrip(),
        ask_forms=tuple(
            block for block in redactor.captured if block.get("type") == "ask_user_form"
        ),
    )


async def _discard(_token: str) -> None:
    return None


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
