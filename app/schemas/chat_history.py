"""`HistoryMessage` - lives here (not in `app.graph.streaming_state`) so both
the graph layer and `app.rag.prompting` can depend on it without inverting
this project's stated dependency direction (API -> Graph -> RAG services ->
repositories, see AGENTS.md) - the same reason `PendingRound`/
`AcademicSecurityContext` live under `app.schemas` rather than in the graph
package.
"""

from pydantic import BaseModel


class HistoryMessage(BaseModel):
    """One prior turn's message, trimmed down to what the prompt needs - just
    enough for the model to see how the conversation actually went,
    independent of `confirmed_metadata` (the structured, never-expiring
    memory). This is the raw, capped, most-recent-N conversational context.
    `role` is Java's `MsgRole` as a string ("USER"/"ASSISTANT")."""

    role: str
    content: str
