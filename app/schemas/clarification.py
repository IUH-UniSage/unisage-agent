"""`PendingClarification`, exactly as specified by
`missing_metadata_clarification_design.md` section 5 (do not invent a
different "turns_waited" concept from an older draft — see tasks/plan.md).
"""

from pydantic import BaseModel, Field


class PendingClarification(BaseModel):
    """One outstanding "please clarify" round, attached to a conversation.

    `origin_node` is the node to resume at once the user's reply resolves
    this — for Type B (content-driven fields, detected at node 12 but always
    resumed at `QueryTransformationNode`) this deliberately does NOT equal
    the node that detected the missing field; see design doc section 5,
    "điểm phát hiện != điểm quay lại".
    """

    origin_node: str
    pending_sub_query_id: str | None = None
    missing_fields: list[str]
    options: list[list[str] | None] = Field(
        description="Parallel to missing_fields; None means a free-text field (e.g. a score)."
    )
    retry_count: int = 0
