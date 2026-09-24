"""Message classification node - splits an incoming message into tasks, one
per distinct question, each labeled with an intent and a routing_mode.

The LLM is asked to output a small JSON object (see
`agents/message_classification.yaml`'s `## Output`), not a bare label:
`tasks` is what lets IntentRoutingNode (04) send different questions in the
same message to different branches (e.g. a calculation and a procedure
question in one turn), and each task's `routing_mode` is what lets it ask
QueryTransformationNode (06) for the decomposer on a comparison question -
all without a second LLM call.
"""

import json
import logging
import re
from collections.abc import Sequence
from typing import Any

from pydantic_ai import Agent
from pydantic_ai.models import Model

from app.rag.prompting import append_recent_history, get_templates
from app.schemas.chat_history import HistoryMessage
from app.schemas.intent import ClassifiedTask, IntentClassification, RoutingMode

logger = logging.getLogger(__name__)

# `academic_procedure`/`academic_calendar`/`academic_document` were merged
# into `academic_advisory` - all four always took the same path (06 → 08 →
# 09 → 10, one prompt frame) and node 10 never reads the label, so the
# split only cost the classifier three extra boundaries to get right. A
# model that still returns one of them is parsed like any unknown label.
INTENT_TYPES = (
    "greeting",
    "social_chat",
    "academic_advisory",
    "academic_calculation",
    "off_topic",
)

MAX_TASKS = 3

_DEFAULT_INTENT = "academic_advisory"

# Intents that never reach QueryTransformationNode - routing_mode is
# meaningless for them regardless of what the model returned.
_NO_ROUTING_MODE_INTENTS = frozenset(
    {"social_chat", "off_topic", "academic_calculation", "greeting"}
)

_JSON_OBJECT_PATTERN = re.compile(r"\{.*\}", re.DOTALL)


def build_classification_agent(model: Model | str) -> Agent[None, str]:
    return Agent(model=model, system_prompt=get_templates().agent_message_classification)


def _fallback_classification(message: str) -> IntentClassification:
    """The safe default for any output that can't be trusted: the whole
    message as one `academic_advisory` + `SINGLE` task. The default flow
    (retrieve + rerank + generate) is the safest guess for an academic
    question the classifier didn't cleanly label, and never worse than
    crashing the turn."""

    return IntentClassification(
        tasks=[ClassifiedTask(intent=_DEFAULT_INTENT, query=message, routing_mode="SINGLE")]
    )


def _load_json_object(raw_output: str) -> dict[str, Any] | None:
    """Tries the trimmed output as-is first (the common case: the model
    obeyed "no code fence"), then the first `{...}` block (a model that
    wrapped the JSON in ```json anyway)."""

    candidate = raw_output.strip().strip("`").strip()
    match = _JSON_OBJECT_PATTERN.search(candidate)
    for text in (candidate, match.group(0) if match else None):
        if text is None:
            continue
        try:
            loaded = json.loads(text)
        except json.JSONDecodeError:
            continue
        if isinstance(loaded, dict):
            return loaded
    return None


def _normalize_routing_mode(intent: str, raw_mode: object) -> RoutingMode | None:
    if intent in _NO_ROUTING_MODE_INTENTS:
        return None
    if raw_mode == "MULTI":
        return "MULTI"
    return "SINGLE"


def _normalize_task(raw_task: object, message: str) -> ClassifiedTask | None:
    """One task from the model's `tasks` array, made safe to route on: an
    unknown intent (including the merged-away `academic_procedure`/
    `academic_calendar`/`academic_document`) becomes `academic_advisory`,
    an empty `query` becomes the whole message, and `routing_mode` is forced
    to match the intent. A non-object entry is dropped."""

    if not isinstance(raw_task, dict):
        return None
    raw_intent = raw_task.get("intent")
    intent = raw_intent.strip().lower() if isinstance(raw_intent, str) else ""
    if intent not in INTENT_TYPES:
        intent = _DEFAULT_INTENT
    raw_query = raw_task.get("query")
    query = raw_query.strip() if isinstance(raw_query, str) and raw_query.strip() else message
    return ClassifiedTask(
        intent=intent,
        query=query,
        routing_mode=_normalize_routing_mode(intent, raw_task.get("routing_mode")),
    )


def parse_classification(raw_output: str, message: str) -> IntentClassification:
    """Best-effort parse of the model's output into `IntentClassification`.

    Unparseable JSON, a missing/non-list `tasks`, or no usable task at all
    fall back to one `academic_advisory` + `SINGLE` task over the whole
    message. More than `MAX_TASKS` tasks keeps the first `MAX_TASKS` (the
    rest are dropped and logged).
    """

    parsed = _load_json_object(raw_output)
    if parsed is None or not isinstance(parsed.get("tasks"), list):
        return _fallback_classification(message)

    tasks = [
        task
        for task in (_normalize_task(raw_task, message) for raw_task in parsed["tasks"])
        if task is not None
    ]
    if not tasks:
        return _fallback_classification(message)
    if len(tasks) > MAX_TASKS:
        logger.info("classification returned %d tasks, keeping the first %d", len(tasks), MAX_TASKS)
        tasks = tasks[:MAX_TASKS]

    raw_confidence = parsed.get("confidence")
    confidence = (
        float(raw_confidence)
        if isinstance(raw_confidence, int | float) and not isinstance(raw_confidence, bool)
        else None
    )
    return IntentClassification(tasks=tasks, confidence=confidence)


async def classify_intent(
    agent: Agent[None, str],
    message: str,
    *,
    history: Sequence[HistoryMessage] = (),
) -> IntentClassification:
    """Run the classification agent and parse its JSON output into tasks.

    `history` lets a short follow-up ("sao không có ngành X?") be read in
    the context of the previous turn instead of being judged off_topic on
    its own.
    """

    result = await agent.run(append_recent_history(message, history))
    return parse_classification(result.output or "", message)
