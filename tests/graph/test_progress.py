"""`event: status` - node name -> stage, one status per change, and the SSE line."""

import asyncio

import pytest

from app.core.observability.graph_trace import GraphTrace
from app.graph.progress import stage_for_node, status_reporter
from app.graph.queue_items import DoneItem, QueueItem, StatusItem
from app.services.chat_stream_service import _sse_token_generator


@pytest.mark.parametrize(
    ("node", "stage"),
    [
        ("01_GreetingDetectionNode", "thinking"),
        ("02_ClarificationResume", "thinking"),
        ("03_MessageClassificationNode", "analyzing"),
        ("04_IntentRoutingNode", "analyzing"),
        ("04_IntentRouting_SocialChat", "thinking"),
        ("05_OffTopicRejectNode", "thinking"),
        ("06_QueryTransformationNode", "analyzing"),
        ("07_CalculationNode", "calculating"),
        ("07a_CalculationLLMNode", "calculating"),
        ("08_RetrievalFilteringNode", "retrieving"),
        ("09_PostRetrievalRerankNode", "retrieving"),
        ("09a_LLMRerankNode", "retrieving"),
        ("09b_WebSearchNode", "web_searching"),
        ("09c_WebJudgeNode", "web_searching"),
        ("10_GenerationSynthesisNode", "generating"),
        ("11_TicketFallbackNode", "thinking"),
    ],
)
def test_stage_for_node(node: str, stage: str) -> None:
    assert stage_for_node(node) == stage


def test_one_status_per_stage_change_and_one_per_web_page() -> None:
    sent: list[tuple[str, str | None]] = []
    report = status_reporter(lambda stage, url: sent.append((stage, url)))

    for node in (
        "03_MessageClassificationNode",
        "04_IntentRoutingNode",
        "08_RetrievalFilteringNode",
        "09_PostRetrievalRerankNode",
        "09a_LLMRerankNode",
        "09b_WebSearchNode",
    ):
        report(node, None)
    report("09b_WebSearchNode", "https://pdt.iuh.edu.vn/a")
    report("09b_WebSearchNode", "https://pdt.iuh.edu.vn/b")
    report("09c_WebJudgeNode", None)
    report("10_GenerationSynthesisNode", None)

    assert sent == [
        ("analyzing", None),
        ("retrieving", None),
        ("web_searching", None),
        ("web_searching", "https://pdt.iuh.edu.vn/a"),
        ("web_searching", "https://pdt.iuh.edu.vn/b"),
        ("generating", None),
    ]


def test_graph_trace_reports_nodes_and_web_pages() -> None:
    seen: list[tuple[str, str | None]] = []
    trace = GraphTrace(
        conversation_id="c",
        message_id="m",
        user_id=None,
        client_ip=None,
        on_status=lambda node, url: seen.append((node, url)),
    )

    trace.node("09b_WebSearchNode")
    trace.web_page("https://pdt.iuh.edu.vn/a")
    trace.step("07a_CalculationLLMNode", parent="07_CalculationNode")

    # A step inside a node is not a new status.
    assert seen == [("09b_WebSearchNode", None), ("09b_WebSearchNode", "https://pdt.iuh.edu.vn/a")]


@pytest.mark.asyncio
async def test_status_is_streamed_as_an_sse_event() -> None:
    queue: asyncio.Queue[QueueItem] = asyncio.Queue()
    for item in (
        StatusItem(stage="retrieving"),
        StatusItem(stage="web_searching", url="https://pdt.iuh.edu.vn/thông-báo"),
        DoneItem(),
    ):
        queue.put_nowait(item)

    body = "".join([chunk async for chunk in _sse_token_generator(queue)])

    assert body == (
        'event: status\ndata: {"stage": "retrieving"}\n\n'
        'event: status\ndata: {"stage": "web_searching", '
        '"url": "https://pdt.iuh.edu.vn/thông-báo"}\n\n'
        "event: done\ndata: {}\n\n"
    )
