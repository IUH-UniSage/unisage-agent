from dataclasses import dataclass, field
from typing import Any


@dataclass
class ChatState:
    """Mutable state passed through the RAG graph."""

    query: str
    user_id: str = "guest"
    user_role: str = "GUEST"
    user_faculty: str = "GLOBAL"
    user_level: int = 1
    trace_id: str | None = None
    intent: str | None = None
    sub_queries: list[str] = field(default_factory=list)
    retrieved_chunks: list[dict[str, Any]] = field(default_factory=list)
    final_response: str | None = None
    citations: list[dict[str, Any]] = field(default_factory=list)
