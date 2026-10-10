"""run_and_persist and the panel: state first, projection second, event last."""

import asyncio
import json
import uuid
from typing import Any

import httpx
import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.usage.usage_recorder import UsageRecorder
from app.database.repositories.clarification_state import ClarificationRoundRepository
from app.graph import streaming_session
from app.graph.clarification_round import (
    TaskQuestions,
    advisory_questions,
    advisory_task,
    build_round,
)
from app.graph.queue_items import ClarificationItem, DoneItem, QueueItem, TokenItem
from app.graph.streaming_session import ClaimContext, run_and_persist
from app.graph.streaming_state import GraphInput, GraphModels, GraphOutput
from app.integrations import backend_java_client
from app.integrations.backend_java_client import BackendJavaClient
from app.schemas.clarification import PendingRound
from app.schemas.intent import ClassifiedTask
from app.schemas.security import AcademicSecurityContext
from tests.llm_mocks import FakeRetrievalService

ASSISTANT_ID = "00000000-0000-0000-0000-0000000000a2"
FORM = {
    "type": "ask_user_form",
    "fields": [{"field": "nganh", "label": "Ngành", "options": [{"id": "cntt"}, {"id": "kt"}]}],
}


def _round() -> PendingRound:
    pending = build_round(
        [
            TaskQuestions(
                task=advisory_task("T1", [ClassifiedTask(intent="academic_advisory", query="q")]),
                questions=advisory_questions([FORM], confirmed_metadata={}),
            )
        ],
        original_query="q",
        chain_depth=1,
    )
    assert pending is not None
    return pending


class _Java:
    def __init__(self, status: int = 200) -> None:
        self.status = status
        self.patches: list[dict[str, Any]] = []
        self.paths: list[str] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.paths.append(request.url.path)
        body = json.loads(request.read())
        if request.url.path == "/internal/calculation-traces":
            self.traces = body
            return httpx.Response(200, json={})
        self.patches.append(body)
        return httpx.Response(self.status, json={})


@pytest.fixture(autouse=True)
def _fast_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(backend_java_client, "JAVA_RETRY_DELAYS_SECONDS", (0.0, 0.0, 0.0))


async def _run(
    factory: async_sessionmaker[AsyncSession],
    java: _Java,
    output: GraphOutput,
    monkeypatch: pytest.MonkeyPatch,
    claim: ClaimContext | None = None,
) -> list[QueueItem]:
    async def fake_graph(*_args: object, **_kwargs: object) -> GraphOutput:
        return output

    monkeypatch.setattr(streaming_session, "run_graph", fake_graph)
    queue: asyncio.Queue[QueueItem] = asyncio.Queue()
    await run_and_persist(
        java_client=BackendJavaClient(
            base_url="http://java.test", transport=httpx.MockTransport(java.handler)
        ),
        conversation_id="c1",
        assistant_message_id=ASSISTANT_ID,
        authorization=None,
        graph_input=GraphInput(
            conversation_id="c1",
            user_message="q",
            is_first_turn=False,
            security=AcademicSecurityContext(),
        ),
        models=GraphModels(
            classification="m",
            query_transformation="m",
            generation="m",
            retrieval=FakeRetrievalService(),
        ),
        usage_recorder=UsageRecorder(
            request_id=str(uuid.uuid4()), purpose="CHAT", conversation_id="c1"
        ),
        queue=queue,
        session_factory=factory,
        claim=claim,
    )
    items: list[QueueItem] = []
    while not queue.empty():
        items.append(queue.get_nowait())
    return items


async def _state(factory: async_sessionmaker[AsyncSession]) -> tuple[str | None, uuid.UUID | None]:
    async with factory() as session:
        state = await ClarificationRoundRepository(session).get_round("c1")
        return state.status, state.round.panel.panel_id if state.round else None


@pytest.mark.asyncio
async def test_panel_is_stored_then_projected_then_sent(
    db_session_factory: async_sessionmaker[AsyncSession], monkeypatch: pytest.MonkeyPatch
) -> None:
    pending = _round()
    java = _Java()
    items = await _run(
        db_session_factory,
        java,
        GraphOutput(response_text="Bạn học ngành nào?", pending_round=pending),
        monkeypatch,
    )

    assert await _state(db_session_factory) == ("OPEN", pending.panel.panel_id)
    metadata = java.patches[0]["metadata"]["clarification"]
    assert metadata["status"] == "open"
    assert metadata["panel"]["panel_id"] == str(pending.panel.panel_id)
    assert "origin" not in metadata["panel"]["questions"][0]
    assert java.patches[0]["content"] == "Bạn học ngành nào?"
    kinds = [type(item) for item in items]
    assert kinds[-2:] == [ClarificationItem, DoneItem]
    clarification = items[-2]
    assert isinstance(clarification, ClarificationItem)
    assert clarification.panel == metadata["panel"]


@pytest.mark.asyncio
async def test_projection_failure_revokes_the_round_and_sends_no_panel(
    db_session_factory: async_sessionmaker[AsyncSession], monkeypatch: pytest.MonkeyPatch
) -> None:
    java = _Java(status=500)
    items = await _run(
        db_session_factory,
        java,
        GraphOutput(response_text="x", pending_round=_round()),
        monkeypatch,
    )
    assert len(java.patches) == 4
    assert await _state(db_session_factory) == (None, None)
    assert not any(isinstance(item, ClarificationItem) for item in items)
    assert isinstance(items[-1], DoneItem)


@pytest.mark.asyncio
async def test_existing_round_is_not_overwritten_and_no_metadata_is_sent(
    db_session_factory: async_sessionmaker[AsyncSession], monkeypatch: pytest.MonkeyPatch
) -> None:
    first = _round()
    async with db_session_factory() as session:
        await ClarificationRoundRepository(session).upsert_open("c1", first, {})
        await session.commit()
    java = _Java()
    items = await _run(
        db_session_factory,
        java,
        GraphOutput(response_text="x", pending_round=_round()),
        monkeypatch,
    )
    assert await _state(db_session_factory) == ("OPEN", first.panel.panel_id)
    assert java.patches[0].get("metadata") is None
    assert not any(isinstance(item, ClarificationItem) for item in items)


@pytest.mark.asyncio
async def test_turn_without_panel_saves_confirmed_metadata(
    db_session_factory: async_sessionmaker[AsyncSession], monkeypatch: pytest.MonkeyPatch
) -> None:
    java = _Java()
    await _run(
        db_session_factory,
        java,
        GraphOutput(response_text="x", confirmed_metadata={"nganh": "cntt"}),
        monkeypatch,
    )
    async with db_session_factory() as session:
        state = await ClarificationRoundRepository(session).get_round("c1")
    assert state.status is None and state.confirmed_metadata == {"nganh": "cntt"}


@pytest.mark.asyncio
async def test_claimed_turn_whose_claim_was_lost_writes_nothing_to_java(
    db_session_factory: async_sessionmaker[AsyncSession], monkeypatch: pytest.MonkeyPatch
) -> None:
    java = _Java()
    items = await _run(
        db_session_factory,
        java,
        GraphOutput(response_text="x"),
        monkeypatch,
        claim=ClaimContext(token=uuid.uuid4()),
    )
    assert java.patches == []
    assert isinstance(items[-1], DoneItem)
    assert not any(isinstance(item, TokenItem) for item in items)


@pytest.mark.asyncio
async def test_calculation_trace_is_pushed_before_the_public_summary_is_finalized(
    db_session_factory: async_sessionmaker[AsyncSession], monkeypatch: pytest.MonkeyPatch
) -> None:
    java = _Java()
    public = [{"item_id": "T1", "run_id": "r", "mode": "builtin", "status": "computed"}]
    private = [
        {"itemId": "T1", "runId": "r", "trace": {"question_raw": "điểm 8 7 6", "inputs": []}}
    ]
    await _run(
        db_session_factory,
        java,
        GraphOutput(response_text="x", calculation_items=public, calculation_traces=private),
        monkeypatch,
    )
    assert java.paths == ["/internal/calculation-traces", f"/messages/{ASSISTANT_ID}"]
    assert java.traces == {"messageId": ASSISTANT_ID, "items": private}
    metadata = java.patches[0]["metadata"]
    assert metadata == {"calculation": {"schema_version": 1, "items": public}}
    assert "question_raw" not in json.dumps(metadata, ensure_ascii=False)
