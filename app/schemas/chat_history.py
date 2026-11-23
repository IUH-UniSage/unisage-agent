"""`HistoryMessage` - lives here (not in `app.graph.streaming_state`) so both
the graph layer and `app.rag.prompting` can depend on it without inverting
this project's stated dependency direction (API -> Graph -> RAG services ->
repositories, see AGENTS.md) - the same reason `PendingRound`/
`AcademicSecurityContext` live under `app.schemas` rather than in the graph
package.
"""

from pydantic import BaseModel, Field

# The ASSISTANT message metadata key holding the standalone questions that turn
# retrieved with (written by `run_and_persist`, read back from Java's turn context).
RETRIEVAL_QUESTIONS_KEY = "retrieval_questions"


class HistoryMessage(BaseModel):
    """One prior turn's message, trimmed down to what the prompt needs - just
    enough for the model to see how the conversation actually went,
    independent of `confirmed_metadata` (the structured, never-expiring
    memory). This is the raw, capped, most-recent-N conversational context.
    `role` is Java's `MsgRole` as a string ("USER"/"ASSISTANT").

    `retrieval_questions` (ASSISTANT only): the standalone questions the turn
    retrieved with. Each one already folds in the turns before it, so the
    latest one carries a follow-up chain ("k19 thì sao" -> "k21" -> "hệ chính
    quy á bạn") past the recent-history window. `calculation`: the turn
    answered a calculation, so a later follow-up is not about the last lookup."""

    role: str
    content: str
    retrieval_questions: list[str] = Field(default_factory=list)
    calculation: bool = False
