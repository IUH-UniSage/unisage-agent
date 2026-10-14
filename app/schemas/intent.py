"""MessageClassificationNode output: one task per distinct question. A
comparison question is one task with routing_mode MULTI."""

from typing import Literal

from pydantic import BaseModel, Field

from app.schemas.web_search import WebSlots

RoutingMode = Literal["SINGLE", "MULTI"]


class ClassifiedTask(BaseModel):
    """One question in the message; `routing_mode` is None unless the task
    goes to QueryTransformationNode."""

    intent: str
    query: str
    routing_mode: RoutingMode | None = None
    # Retrieval text classification wrote alongside (CHAT_CLASSIFY_WITH_RETRIEVAL), so
    # QueryTransformationNode can skip its own LLM call. Never persisted: a resumed
    # task re-runs on a new query.
    hyde_text: str | None = Field(default=None, exclude=True)
    sub_queries: list[str] | None = Field(default=None, exclude=True)
    # What WebSearchNode looks for, as slots code turns into a keyword query
    # (app/rag/web_query.py). Not persisted either: a resumed task searches by its question.
    web: WebSlots | None = Field(default=None, exclude=True)
    # A MULTI task's slots per sub-query (same order as `sub_queries`, None where the
    # classifier wrote none): each one is searched on its own - one keyword query for
    # "học phí ... và lịch thi chứng chỉ" found only fee pages (14-10-2026).
    sub_query_web: list[WebSlots | None] | None = Field(default=None, exclude=True)


class IntentClassification(BaseModel):
    tasks: list[ClassifiedTask] = Field(min_length=1)
    confidence: float | None = None
    # What the student said about themselves this turn ("tôi nhầm, khoá 2023 mới đúng"):
    # field -> new value, None = remove. Only registry profile fields are applied.
    metadata_updates: dict[str, str | None] = Field(default_factory=dict)
