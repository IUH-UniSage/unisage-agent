"""Run the chat graph in-process for one question and record what happened (Task 5).

Nothing in `app/` changes. What the SSE stream never shows is captured here:

- `RecordingTrace` (a `GraphTrace`) - the nodes run, the classification line, TTFT;
- `RecordingRetrieval` - every Qdrant result before rerank, with its permission metadata;
- `install_hooks()` - wraps `run_generation_synthesis` / `search_web` in the graph module to
  keep the chunks (after rerank) and web pages the answer was generated from.

Each question runs with its own `Recording` in a `ContextVar`, so several questions can run
concurrently; `asyncio.to_thread` (how the graph calls retrieval) copies the context along.
"""

import asyncio
import contextvars
import time
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from app.core.observability.graph_trace import GraphTrace
from app.core.usage.usage_recorder import UsageRecorder
from app.graph import streaming_graph
from app.graph.streaming_graph import run_graph
from app.graph.streaming_state import GraphInput, GraphModels, GraphOutput
from app.rag.prompting.citations import source_title
from app.rag.retrieval.service import RetrievalServiceProtocol
from app.schemas.retrieval import RetrievedChunk
from app.schemas.security import AcademicSecurityContext
from app.schemas.web_search import WebSearchResult


@dataclass
class Recording:
    started: float = field(default_factory=time.perf_counter)
    nodes: list[str] = field(default_factory=list)
    prompts: dict[str, list[str]] = field(default_factory=dict)
    retrievals: list[dict[str, Any]] = field(default_factory=list)
    context_chunks: list[RetrievedChunk] = field(default_factory=list)
    web_results: list[WebSearchResult] = field(default_factory=list)
    web_queries: list[str] = field(default_factory=list)
    ttft_ms: float | None = None


_current: contextvars.ContextVar[Recording | None] = contextvars.ContextVar(
    "eval_recording", default=None
)


def current() -> Recording | None:
    return _current.get()


class RecordingTrace(GraphTrace):
    """`GraphTrace` that also keeps the node names, the `prompt` dumps and TTFT."""

    def __init__(self, recording: Recording, *, conversation_id: str) -> None:
        super().__init__(
            conversation_id=conversation_id, message_id="eval", user_id="eval", client_ip=None
        )
        self._recording = recording

    def node(self, name: str, **kwargs: Any) -> None:
        self._recording.nodes.append(name)
        super().node(name, **kwargs)

    def prompt(self, node_name: str, text: str) -> None:
        self._recording.prompts.setdefault(node_name, []).append(text)
        super().prompt(node_name, text)

    def first_token(self) -> None:
        if self._recording.ttft_ms is None:
            self._recording.ttft_ms = (time.perf_counter() - self._recording.started) * 1000
        super().first_token()


@dataclass(frozen=True)
class RecordingRetrieval:
    """Wraps the real retrieval service; logs every query and its chunks' metadata."""

    inner: RetrievalServiceProtocol

    def retrieve(
        self, query: str, *, security: AcademicSecurityContext, limit: int | None = None
    ) -> list[RetrievedChunk]:
        return self.retrieve_many([query], security=security, limit=limit)[0]

    def retrieve_many(
        self, queries: list[str], *, security: AcademicSecurityContext, limit: int | None = None
    ) -> list[list[RetrievedChunk]]:
        results = self.inner.retrieve_many(queries, security=security, limit=limit)
        recording = current()
        if recording is not None:
            for query, chunks in zip(queries, results, strict=True):
                recording.retrievals.append(
                    {"query": query, "chunks": [chunk_summary(chunk) for chunk in chunks]}
                )
        return results


def chunk_summary(chunk: RetrievedChunk) -> dict[str, Any]:
    return {
        "chunk_id": chunk.chunk_id,
        "document_id": chunk.metadata.get("document_id"),
        "title": source_title(chunk.source),
        "department": chunk.metadata.get("department"),
        "access_level": chunk.metadata.get("access_level"),
        "is_public": chunk.metadata.get("is_public"),
        "score": round(chunk.score, 4),
        "page_start": chunk.page_start,
    }


_hooks_installed = False


def install_hooks() -> None:
    """Wrap the two graph-module functions whose arguments the trace never exposes.
    Idempotent; the wrappers just record and call through."""

    global _hooks_installed
    if _hooks_installed:
        return
    _hooks_installed = True
    original_generation = getattr(streaming_graph, "run_generation_synthesis")  # noqa: B009
    original_search = getattr(streaming_graph, "search_web")  # noqa: B009

    async def recording_generation(*args: Any, **kwargs: Any) -> Any:
        recording = current()
        if recording is not None:
            recording.context_chunks = list(kwargs.get("chunks") or [])
            recording.web_results = list(kwargs.get("web_results") or [])
        return await original_generation(*args, **kwargs)

    async def recording_search(queries: Sequence[str], **kwargs: Any) -> Any:
        recording = current()
        if recording is not None:
            recording.web_queries.extend(queries)
        return await original_search(queries, **kwargs)

    setattr(streaming_graph, "run_generation_synthesis", recording_generation)  # noqa: B010
    setattr(streaming_graph, "search_web", recording_search)  # noqa: B010


@dataclass
class TurnResult:
    output: GraphOutput | None
    response_text: str
    recording: Recording
    total_ms: float
    usage_lines: list[dict[str, Any]]
    error: str | None = None
    error_code: int | None = None


async def run_question(
    question: str,
    security: AcademicSecurityContext,
    models: GraphModels,
    *,
    timeout_s: float = 180.0,
) -> TurnResult:
    """One fresh first-turn conversation for `question`, as `security`."""

    install_hooks()
    recording = Recording()
    token = _current.set(recording)
    conversation_id = f"eval-{uuid.uuid4()}"
    trace = RecordingTrace(recording, conversation_id=conversation_id)
    # Not closed on purpose: its lines are read below for tokens/cost, and closing would
    # enqueue a usage log for message ids that do not exist in the backend.
    usage = UsageRecorder(request_id=str(uuid.uuid4()), purpose="CHAT")
    tokens: list[str] = []

    async def sink(text: str) -> None:
        trace.first_token()
        tokens.append(text)

    graph_input = GraphInput(
        conversation_id=conversation_id,
        user_message=question,
        is_first_turn=True,
        security=security,
    )
    output: GraphOutput | None = None
    error: str | None = None
    error_code: int | None = None
    try:
        output = await asyncio.wait_for(
            run_graph(graph_input, models, sink, trace, usage), timeout=timeout_s
        )
    except TimeoutError:
        error = f"timeout after {timeout_s:.0f}s"
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        error_code = getattr(exc, "code", None) or getattr(
            getattr(exc, "error_code", None), "code", None
        )
    finally:
        trace.finish()
        _current.reset(token)
    return TurnResult(
        output=output,
        response_text=output.response_text if output is not None else "".join(tokens),
        recording=recording,
        total_ms=(time.perf_counter() - recording.started) * 1000,
        usage_lines=list(usage._lines),  # read-only, see above
        error=error,
        error_code=error_code if isinstance(error_code, int) else None,
    )
