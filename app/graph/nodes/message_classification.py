"""MessageClassificationNode - splits a message into tasks (one per distinct
question), each with an intent and a routing_mode."""

import json
import logging
import re
from collections.abc import Mapping, Sequence
from typing import Any

from pydantic_ai import Agent
from pydantic_ai.models import Model

from app.core.config import settings
from app.core.registry.model_registry import CredentialConfig
from app.core.registry.model_router import ModelRouter
from app.core.timezone import now_ict
from app.graph.streaming import (
    AgentFactory,
    AttemptRecorder,
    BudgetContext,
    FailoverCallback,
    auxiliary_model_settings,
    run_agent_text_with_failover,
)
from app.rag.prompting import (
    append_current_time,
    append_recent_history,
    build_declared_attributes_section,
    build_known_metadata_fields_section,
    get_templates,
)
from app.rag.web_query import relative_year
from app.schemas.chat_history import HistoryMessage
from app.schemas.intent import ClassifiedTask, IntentClassification, RoutingMode
from app.schemas.web_search import WEB_TOPICS, WebSlots

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
    # `.replace`, not `.format`: the template is full of literal JSON braces.
    system_prompt = templates.agent_message_classification.replace(
        "{profile_fields}", build_known_metadata_fields_section()
    )
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
        **_task_web(raw_task, routing_mode) if intent == "academic_advisory" else {},
    )


def _task_web(raw_task: dict[str, Any], routing_mode: str | None) -> dict[str, Any]:
    """`web` is one object (SINGLE, or a MULTI written the old way) or, for MULTI, a list
    in the order of `sub_queries`: `sub_query_web`, padded / cut to their number."""

    raw = raw_task.get("web")
    question = _text(raw_task.get("standalone_question")) or _text(raw_task.get("query")) or ""
    if not isinstance(raw, list):
        return {"web": _with_year(_web_slots(raw), question)}
    per_sub = [_web_slots(item) for item in raw]
    sub_queries = _sub_queries(raw_task) if routing_mode == "MULTI" else None
    if sub_queries is None:
        slots = next((slots for slots in per_sub if slots is not None), None)
        return {"web": _with_year(slots, question)}
    per_sub = (per_sub + [None] * len(sub_queries))[: len(sub_queries)]
    per_sub = [_with_year(slots, sub) for slots, sub in zip(per_sub, sub_queries, strict=True)]
    return {"sub_query_web": per_sub} if any(per_sub) else {}


def _with_year(slots: WebSlots | None, question: str) -> WebSlots | None:
    """The year a relative time word in the question means ("đợt gần nhất", "năm nay"),
    when the classifier left `nam_hoc` out - it is told to write it, and did not for
    "lịch thi chứng chỉ CNTT đợt gần nhất" (14-10-2026)."""

    if slots is None or slots.nam_hoc is not None:
        return slots
    year = relative_year(question, now_ict().date())
    return slots.model_copy(update={"nam_hoc": year}) if year else slots


def _text(value: object) -> str | None:
    return value.strip() if isinstance(value, str) and value.strip() else None


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


def _web_slots(raw: object) -> WebSlots | None:
    """One `web` object as `WebSlots`: unknown topics dropped (none left -> `khac`), empty
    or "null" texts as None. No subject means nothing to search for -> None (the web search
    then falls back to the sub-query's question)."""

    if not isinstance(raw, dict):
        return None
    subject = _slot_text(raw.get("subject"))
    if subject is None:
        return None
    raw_topics = raw.get("topics", raw.get("topic"))
    if isinstance(raw_topics, str):
        raw_topics = [raw_topics]
    topics: list[str] = []
    for topic in raw_topics if isinstance(raw_topics, list) else []:
        name = topic.strip().lower() if isinstance(topic, str) else ""
        if name in WEB_TOPICS and name not in topics:
            topics.append(name)
    return WebSlots.model_validate(
        {
            "topics": topics or ["khac"],
            "subject": subject,
            **{
                name: _slot_text(raw.get(name))
                for name in ("bac_dao_tao", "he_dao_tao", "nganh_hoc", "co_so", "nam_hoc")
            },
        }
    )


def _slot_text(value: object) -> str | None:
    if isinstance(value, int) and not isinstance(value, bool):
        value = str(value)
    if not isinstance(value, str):
        return None
    text = value.strip()
    return text if text and text.lower() not in {"null", "none"} else None


def _metadata_updates(raw: object) -> dict[str, str | None]:
    """`{field: value | null}`; a number is kept as text, anything else is dropped.
    Which fields may change is decided later (`metadata_fields.apply_updates`)."""

    updates: dict[str, str | None] = {}
    for name, value in raw.items() if isinstance(raw, dict) else []:
        if not isinstance(name, str) or not name.strip():
            continue
        if value is None or isinstance(value, str):
            updates[name.strip()] = value
        elif isinstance(value, int | float) and not isinstance(value, bool):
            updates[name.strip()] = str(value)
    return updates


def describe_classification(classification: IntentClassification) -> str:
    """JSON of what the graph acts on, including the retrieval text `model_dump` leaves out."""

    tasks = [
        {
            "intent": task.intent,
            "query": task.query,
            "routing_mode": task.routing_mode,
            "hyde_text": task.hyde_text,
            "sub_queries": task.sub_queries,
            "web": (
                [
                    slots.model_dump(exclude_none=True) if slots else None
                    for slots in task.sub_query_web
                ]
                if task.sub_query_web
                else task.web.model_dump(exclude_none=True)
                if task.web
                else None
            ),
        }
        for task in classification.tasks
    ]
    return json.dumps(
        {
            "tasks": tasks,
            "confidence": classification.confidence,
            "metadata_updates": classification.metadata_updates,
        },
        ensure_ascii=False,
        indent=2,
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
    return IntentClassification(
        tasks=tasks,
        confidence=confidence,
        metadata_updates=_metadata_updates(parsed.get("metadata_updates")),
    )


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
    profile: Mapping[str, str] | None = None,
) -> IntentClassification:
    """`history` lets a short follow-up be classified in context. `previous_calculation`
    names the conversation's latest computed calculation, so a free-typed follow-up on
    it ("thế cuối kỳ cần bao nhiêu") stays a calculation instead of being guessed from
    the chat text alone. `profile` is what the student declared so far, so a correction
    ("tôi nhầm, khoá 2023 mới đúng") comes back as `metadata_updates`.

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
    prompt += "\n\n" + build_declared_attributes_section(profile or {})
    prompt = append_current_time(prompt)
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
