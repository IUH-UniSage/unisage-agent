"""`PendingClarification`, exactly as specified by
`missing_metadata_clarification_design.md` section 5 (do not invent a
different "turns_waited" concept from an older draft — see tasks/plan.md).
"""

from pydantic import BaseModel, Field

from app.schemas.intent import ClassifiedTask


class PendingClarification(BaseModel):
    """One outstanding "please clarify" round, attached to a conversation.

    `origin_node` is the node to resume at once the user's reply resolves
    this — for Type B (content-driven fields, detected at GenerationSynthesisNode but always
    resumed at `QueryTransformationNode`) this deliberately does NOT equal
    the node that detected the missing field; see design doc section 5,
    "điểm phát hiện != điểm quay lại".
    """

    origin_node: str
    pending_sub_query_id: str | None = None
    origin_tasks: list[ClassifiedTask] | None = Field(
        default=None,
        description=(
            "The advisory tasks running the turn this round started on - resume "
            "re-runs exactly these, at their own modes. `None` for rows persisted "
            "before this field existed; resume then falls back to one SINGLE task "
            "over `original_query`."
        ),
    )
    missing_fields: list[str]
    options: list[list[str] | None] = Field(
        description="Parallel to missing_fields; None means a free-text field (e.g. a score)."
    )
    option_labels: list[list[str] | None] | None = Field(
        default=None,
        description=(
            "Parallel to `options`, the human-readable label shown for each option id "
            "(e.g. id 'cntt' -> label 'Công nghệ Thông tin'). Kept so the deterministic "
            "Clarification Guard can match a reply typed as the LABEL - which is what "
            "the user actually sees on the form chip - not just the internal id. `None` "
            "for rows persisted before this field existed; matching then falls back to "
            "ids only, exactly as before."
        ),
    )
    retry_count: int = 0
    original_query: str = Field(
        default="",
        description=(
            "The user's question that triggered this clarification round (NOT the "
            "reply that answers it) - retrieval on resume must search for this, not "
            "the reply text, or the topic (e.g. 'học phí') is lost entirely once the "
            "student starts answering the form. Defaults to '' for backward "
            "compatibility with rows persisted before this field existed; an empty "
            "value means resume falls back to the reply text, same as before."
        ),
    )
