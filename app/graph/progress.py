"""What the user sees the turn doing (`event: status`, contracts/chat-sse.md §2): the
stage of the node running now, by its trace name (`GraphTrace.node`). The client turns a
stage into text ("Đang tìm thông tin"); an unknown one reads as "thinking"."""

from collections.abc import Callable

THINKING = "thinking"
ANALYZING = "analyzing"
RETRIEVING = "retrieving"
CALCULATING = "calculating"
WEB_SEARCHING = "web_searching"
GENERATING = "generating"

# By trace-name prefix; the first match wins, anything else is THINKING (greeting,
# clarification resume, social chat, off-topic reply, ticket fallback).
_STAGE_BY_PREFIX: tuple[tuple[str, str], ...] = (
    ("03_", ANALYZING),
    ("04_IntentRoutingNode", ANALYZING),
    ("06_", ANALYZING),
    ("07_", CALCULATING),
    ("07a_", CALCULATING),
    ("08_", RETRIEVING),
    ("09_", RETRIEVING),
    ("09a_", RETRIEVING),
    ("09b_", WEB_SEARCHING),
    ("09c_", WEB_SEARCHING),
    ("10_", GENERATING),
)


def stage_for_node(node_name: str) -> str:
    for prefix, stage in _STAGE_BY_PREFIX:
        if node_name.startswith(prefix):
            return stage
    return THINKING


def status_reporter(emit: Callable[[str, str | None], None]) -> Callable[[str, str | None], None]:
    """A `GraphTrace(on_status=...)` callback calling `emit(stage, url)` once per change:
    consecutive nodes of one stage (retrieval, score rerank, LLM rerank) send one status;
    a web page always sends its own."""

    last_stage: str | None = None

    def report(node_name: str, url: str | None) -> None:
        nonlocal last_stage
        stage = stage_for_node(node_name)
        if url is None and stage == last_stage:
            return
        last_stage = stage
        emit(stage, url)

    return report
