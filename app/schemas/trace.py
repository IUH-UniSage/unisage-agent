from typing import Any

from pydantic import BaseModel, Field


class ExecutionStepTrace(BaseModel):
    node_name: str
    input_data: Any | None = None
    output_data: Any | None = None
    latency_ms: float = 0.0


class GraphTrace(BaseModel):
    trace_id: str
    user_id: str | None = None
    user_faculty: str = "GLOBAL"
    query: str
    intent: str | None = None
    steps: list[ExecutionStepTrace] = Field(default_factory=list)
    retrieved_chunk_ids: list[str] = Field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    final_response: str | None = None
