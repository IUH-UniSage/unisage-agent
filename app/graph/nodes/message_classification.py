"""MessageClassificationNode - splits a message into tasks (one per distinct
question), each with an intent and a routing_mode."""

import json
import logging
import re
from collections.abc import Sequence
from typing import Any

from pydantic_ai import Agent
from pydantic_ai.models import Model

from app.core.config import settings
from app.core.registry.model_registry import CredentialConfig
from app.core.registry.model_router import ModelRouter
from app.graph.streaming import (
    AgentFactory,
    AttemptRecorder,
    BudgetContext,
    FailoverCallback,
    auxiliary_model_settings,
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
    templates = get_templates()
    system_prompt = templates.agent_message_classification
    if settings.CHAT_CLASSIFY_WITH_RETRIEVAL:
        system_prompt = f"{system_prompt}\n\n{templates.agent_message_classification_retrieval}"
    return Agent(
        model=model,
        system_prompt=system_prompt,
        model_settings=auxiliary_model_settings(model),
    )


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
    routing_mode = _normalize_routing_mode(intent, raw_task.get("routing_mode"))
    return ClassifiedTask(
        intent=intent,
        query=query,
        routing_mode=routing_mode,
        hyde_text=_hyde_text(raw_task) if routing_mode == "SINGLE" else None,
        sub_queries=_sub_queries(raw_task) if routing_mode == "MULTI" else None,
    )


def _hyde_text(raw_task: dict[str, Any]) -> str | None:
    """Same shape HyDE returns: the standalone question, a blank line, the passage."""

    question = raw_task.get("standalone_question")
    passage = raw_task.get("hyde_passage")
    if not (isinstance(question, str) and question.strip()):
        return None
    if not (isinstance(passage, str) and passage.strip()):
        return None
    return f"{question.strip()}\n\n{passage.strip()}"


def _sub_queries(raw_task: dict[str, Any]) -> list[str] | None:
    raw = raw_task.get("sub_queries")
    if not isinstance(raw, list):
        return None
    queries: list[str] = []
    for item in raw:
        if isinstance(item, str) and item.strip() and item.strip() not in queries:
            queries.append(item.strip())
    # Fewer than 2 is not a decomposition - QueryTransformationNode decides again.
    return queries[: settings.CHAT_MAX_SUB_QUERIES] if len(queries) >= 2 else None


def describe_classification(classification: IntentClassification) -> str:
    """JSON of what the graph acts on, including the retrieval text `model_dump` leaves out."""

    tasks = [
        {
            "intent": task.intent,
            "query": task.query,
            "routing_mode": task.routing_mode,
            "hyde_text": task.hyde_text,
            "sub_queries": task.sub_queries,
        }
        for task in classification.tasks
    ]
    return json.dumps(
        {"tasks": tasks, "confidence": classification.confidence}, ensure_ascii=False, indent=2
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
    previous_calculation: str | None = None,
) -> IntentClassification:
    """`history` lets a short follow-up be classified in context. `previous_calculation`
    names the conversation's latest computed calculation, so a free-typed follow-up on
    it ("thế cuối kỳ cần bao nhiêu") stays a calculation instead of being guessed from
    the chat text alone.

    `purpose`/`credential`/`snapshot_version`/`agent_factory`/`router`/
    `on_failover`/`on_attempt`/`budget` are the same opt-in failover/usage/budget
    wiring as `run_agent_text_with_failover()` - omitted (the default), a provider
    failure propagates immediately, same as before.
    """

    prompt = append_recent_history(message, history)
    if previous_calculation:
        prompt += (
            f"\n\n<previous_calculation_turn>{previous_calculation}</previous_calculation_turn>"
        )
    output = await run_agent_text_with_failover(
        agent,
        prompt,
        purpose=purpose,
        credential=credential,
        snapshot_version=snapshot_version,
        agent_factory=agent_factory,
        router=router,
        on_failover=on_failover,
        on_attempt=on_attempt,
        budget=budget,
        timeout_seconds=settings.CHAT_AUX_CALL_TIMEOUT_SECONDS,
    )
    return parse_classification(output, message)
