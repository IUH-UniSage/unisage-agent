"""MessageClassificationNode output: one task per distinct question. A
comparison question is one task with routing_mode MULTI."""

from typing import Literal

from pydantic import BaseModel, Field

RoutingMode = Literal["SINGLE", "MULTI"]


class ClassifiedTask(BaseModel):
    """One question in the message; `routing_mode` is None unless the task
    goes to QueryTransformationNode."""

    intent: str
    query: str
    routing_mode: RoutingMode | None = None


class IntentClassification(BaseModel):
    tasks: list[ClassifiedTask] = Field(min_length=1)
    confidence: float | None = None
