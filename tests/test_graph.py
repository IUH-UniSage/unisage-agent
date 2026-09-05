import pytest

import app.graph.nodes.retrieve as retrieve_node
from app.graph.deps import ChatDeps
from app.graph.graph import chat_graph
from app.graph.state import ChatState
from app.schemas.retrieval import RetrievedChunk
from tests.llm_mocks import FakeRetrievalService


@pytest.fixture(autouse=True)
def _fake_retrieval(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        retrieve_node,
        "retrieval_service",
        FakeRetrievalService(
            [RetrievedChunk(chunk_id="c1", content="nội dung mẫu", source="s", score=0.9)]
        ),
    )


@pytest.mark.asyncio
async def test_rag_graph_execution() -> None:
    """Test the complete graph pipeline without external providers."""

    state = ChatState(query="Xin chào trợ lý!")
    deps = ChatDeps(db_session=None, openai_api_key="test_key")

    result = await chat_graph.run(inputs=state.query, state=state, deps=deps)

    assert result is not None
    assert "UniSage" in result
    assert state.final_response == result
    assert state.intent == "SINGLE_INTENT"
    assert state.retrieved_chunks


@pytest.mark.asyncio
async def test_graph_detects_multiple_intents() -> None:
    state = ChatState(query="Học phí và đăng ký môn học như thế nào?")
    deps = ChatDeps(db_session=None)

    await chat_graph.run(inputs=state.query, state=state, deps=deps)

    assert state.intent == "MULTI_INTENT"
