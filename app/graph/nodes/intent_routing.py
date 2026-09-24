"""Intent routing node - deterministic, no LLM.

Turns MessageClassificationNode's task list into a `RoutePlan`. Every task
is routed on its own, so one turn can take BOTH the CalculationNode (07)
and QueryTransformationNode (06) branches - e.g. "tính GPA giúp mình và
cho mình biết thủ tục đăng ký tốt nghiệp" - which node 10 later merges into
one answer (see flow_design node 04, AD4/AD14 of the graph flow plan).
"""

from dataclasses import dataclass, field
from typing import Literal

from app.schemas.intent import ClassifiedTask, IntentClassification, RoutingMode

EndRoute = Literal["SOCIAL_CHAT", "OFF_TOPIC"]

# Intents that are answered by the graph's own branches (06/07). Any of them
# in a message makes the social_chat/off_topic/greeting tasks of that same
# message irrelevant - the turn answers the academic part only.
_ACADEMIC_INTENTS = frozenset({"academic_advisory", "academic_calculation"})

SOCIAL_CHAT_TEMPLATE = "Không có gì đâu, bạn cần hỏi thêm gì cứ nhắn cho mình nhé!"


@dataclass(frozen=True)
class RoutePlan:
    """Where one turn goes.

    `end` set → the turn ends on a static template (social chat) or the
    OffTopicRejectNode, and both task lists are empty. Otherwise at least one
    of `calculation_tasks` (node 07) / `advisory_tasks` (node 06, each with
    the QueryTransformation mode to run it in) is non-empty - possibly both.
    """

    end: EndRoute | None = None
    calculation_tasks: list[ClassifiedTask] = field(default_factory=list)
    advisory_tasks: list[tuple[ClassifiedTask, RoutingMode]] = field(default_factory=list)


def plan_route(classification: IntentClassification) -> RoutePlan:
    """Build the `RoutePlan` for one turn from its classified tasks.

    1. Any academic task → drop the social_chat/off_topic/greeting tasks and
       answer the academic part only.
    2. No academic task: any off_topic → OffTopicRejectNode; else any
       social_chat → the social-chat template; else (only `greeting`, which
       should never reach here after node 01 but must not dead-end) → one
       advisory SINGLE task, the same safe fallback as the baseline.
    3. `academic_calculation` → node 07; `academic_advisory` → node 06 with
       `mode = routing_mode` (SINGLE → HyDE, MULTI → decomposer).
    """

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
