"""IntentRoutingNode - deterministic, no LLM.

Routes each classified task on its own, so one turn can take both the
CalculationNode and QueryTransformationNode branches.
"""

from dataclasses import dataclass, field
from typing import Literal

from app.schemas.intent import ClassifiedTask, IntentClassification, RoutingMode

EndRoute = Literal["SOCIAL_CHAT", "OFF_TOPIC"]

# If any of these is present, the message's other tasks are dropped.
_ACADEMIC_INTENTS = frozenset({"academic_advisory", "academic_calculation"})

SOCIAL_CHAT_TEMPLATE = "Không có gì đâu, bạn cần hỏi thêm gì cứ nhắn cho mình nhé!"


@dataclass(frozen=True)
class RoutePlan:
    """Where one turn goes: `end` (social chat / off-topic), or calculation
    and/or advisory tasks - possibly both."""

    end: EndRoute | None = None
    calculation_tasks: list[ClassifiedTask] = field(default_factory=list)
    advisory_tasks: list[tuple[ClassifiedTask, RoutingMode]] = field(default_factory=list)


def plan_route(classification: IntentClassification) -> RoutePlan:
    """Academic tasks win over social/off-topic ones; with no academic task,
    off_topic beats social_chat, and a lone greeting falls back to one
    advisory SINGLE task."""

    tasks = classification.tasks
    academic = [task for task in tasks if task.intent in _ACADEMIC_INTENTS]

    if not academic:
        intents = {task.intent for task in tasks}
        if "off_topic" in intents:
            return RoutePlan(end="OFF_TOPIC")
        if "social_chat" in intents:
            return RoutePlan(end="SOCIAL_CHAT")
        greeting = tasks[0]
        return RoutePlan(
            advisory_tasks=[
                (
                    ClassifiedTask(
                        intent="academic_advisory", query=greeting.query, routing_mode="SINGLE"
                    ),
                    "SINGLE",
                )
            ]
        )

    return RoutePlan(
        calculation_tasks=[task for task in academic if task.intent == "academic_calculation"],
        advisory_tasks=[
            (task, task.routing_mode or "SINGLE")
            for task in academic
            if task.intent == "academic_advisory"
        ],
    )
