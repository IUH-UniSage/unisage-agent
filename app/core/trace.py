import logging
import uuid
from typing import Any

from app.schemas.trace import ExecutionStepTrace, GraphTrace

logger = logging.getLogger("unisage.trace")


class TraceLogger:
    """Execution Trace Audit Logger for UniSage Agent Graph."""

    def __init__(self, query: str, user_faculty: str = "GLOBAL", user_id: str | None = None):
        self.trace = GraphTrace(
            trace_id=str(uuid.uuid4()),
            query=query,
            user_faculty=user_faculty,
            user_id=user_id,
        )

    def log_step(
        self,
        node_name: str,
        input_data: Any | None = None,
        output_data: Any | None = None,
        latency_ms: float = 0.0,
    ) -> None:
        """Record an execution step in the graph state machine."""
        step = ExecutionStepTrace(
            node_name=node_name,
            input_data=input_data,
            output_data=output_data,
            latency_ms=latency_ms,
        )
        self.trace.steps.append(step)
        logger.info(
            f"[TRACE {self.trace.trace_id[:8]}] Node: {node_name} | Latency: {latency_ms:.2f}ms"
        )

    def log_tokens(self, prompt_tokens: int, completion_tokens: int) -> None:
        """Record LLM token consumption."""
        self.trace.prompt_tokens += prompt_tokens
        self.trace.completion_tokens += completion_tokens
        self.trace.total_tokens = self.trace.prompt_tokens + self.trace.completion_tokens

    def finalize(
        self, final_response: str, retrieved_chunk_ids: list[str] | None = None
    ) -> GraphTrace:
        """Finalize and persist trace record."""
        self.trace.final_response = final_response
        if retrieved_chunk_ids:
            self.trace.retrieved_chunk_ids = retrieved_chunk_ids
        logger.info(
            f"[TRACE FINALIZE {self.trace.trace_id[:8]}] Query: '{self.trace.query}' | "
            f"Steps: {len(self.trace.steps)} | Chunks: {len(self.trace.retrieved_chunk_ids)} | "
            f"Tokens: {self.trace.total_tokens}"
        )
        return self.trace
