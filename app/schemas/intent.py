"""MessageClassificationNode's output contract - exactly what
`agents/message_classification.yaml`'s `## Output` section asks the model
for: a list of tasks, one per distinct question in the message.

Two different kinds of "multi" live here, each owned by one node:
- several *different* questions in one message → several `ClassifiedTask`s
  (split by node 03, routed independently by node 04);
- one *comparison* question over several entities → one task with
  `routing_mode = "MULTI"` (node 06's decomposer splits it by entity).
"""

from typing import Literal

from pydantic import BaseModel, Field

RoutingMode = Literal["SINGLE", "MULTI"]


class ClassifiedTask(BaseModel):
    """One question inside the user's message.

    `routing_mode` is `None` for every intent that never reaches
    QueryTransformationNode (`social_chat`, `off_topic`,
    `academic_calculation`, `greeting`); `"MULTI"` only ever pairs with
    `academic_advisory` (a comparison question).
    """

    intent: str
    query: str
    routing_mode: RoutingMode | None = None


class IntentClassification(BaseModel):
    tasks: list[ClassifiedTask] = Field(min_length=1)
    confidence: float | None = None
