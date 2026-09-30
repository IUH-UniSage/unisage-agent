"""MessageClassificationNode - splits a message into tasks (one per distinct
question), each with an intent and a routing_mode."""

import json
import logging
import re
from collections.abc import Sequence
from typing import Any

from pydantic_ai import Agent
from pydantic_ai.models import Model

from app.core.registry.model_registry import CredentialConfig
from app.core.registry.model_router import ModelRouter
from app.graph.streaming import (
    AgentFactory,
    AttemptRecorder,
    BudgetContext,
    FailoverCallback,
    run_agent_text_with_failover,
)
from app.rag.prompting import append_recent_history, get_templates
from app.schemas.chat_history import HistoryMessage
from app.schemas.intent import ClassifiedTask, IntentClassification, RoutingMode

logger = logging.getLogger(__name__)

# academic_procedure/_calendar/_document are merged into academic_advisory.
INTENT_TYPES = (
    "greeting",
    "social_chat",
    "academic_advisory",
    "academic_calculation",
    "off_topic",
)

MAX_TASKS = 3

_DEFAULT_INTENT = "academic_advisory"

# Intents that never reach QueryTransformationNode.
_NO_ROUTING_MODE_INTENTS = frozenset(
    {"social_chat", "off_topic", "academic_calculation", "greeting"}
)

_JSON_OBJECT_PATTERN = re.compile(r"\{.*\}", re.DOTALL)


def build_classification_agent(model: Model | str) -> Agent[None, str]:
    return Agent(model=model, system_prompt=get_templates().agent_message_classification)


def _fallback_classification(message: str) -> IntentClassification:
    """Untrusted output → the whole message as one advisory SINGLE task."""

    return IntentClassification(
        tasks=[ClassifiedTask(intent=_DEFAULT_INTENT, query=message, routing_mode="SINGLE")]
    )


def _load_json_object(raw_output: str) -> dict[str, Any] | None:
    """The output as-is, else its first `{...}` block (e.g. inside a code fence)."""

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
    """Unknown intent → academic_advisory, empty query → whole message,
    routing_mode forced to match the intent; non-objects are dropped."""

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
    """Parse the model's JSON into tasks; anything unusable falls back to one
    advisory SINGLE task. Keeps at most `MAX_TASKS` tasks."""

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
    purpose: str | None = None,
    credential: CredentialConfig | None = None,
    snapshot_version: int | None = None,
    agent_factory: AgentFactory | None = None,
    router: ModelRouter | None = None,
    on_failover: FailoverCallback | None = None,
    on_attempt: AttemptRecorder | None = None,
    budget: BudgetContext | None = None,
) -> IntentClassification:
    """`history` lets a short follow-up be classified in context.

    `purpose`/`credential`/`snapshot_version`/`agent_factory`/`router`/
    `on_failover`/`on_attempt`/`budget` are the same opt-in failover/usage/budget
    wiring as `run_agent_text_with_failover()` - omitted (the default), a provider
    failure propagates immediately, same as before.
    """

    output = await run_agent_text_with_failover(
        agent,
        append_recent_history(message, history),
        purpose=purpose,
        credential=credential,
        snapshot_version=snapshot_version,
        agent_factory=agent_factory,
        router=router,
        on_failover=on_failover,
        on_attempt=on_attempt,
        budget=budget,
    )
    return parse_classification(output, message)
