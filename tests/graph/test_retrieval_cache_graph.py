"""The retrieval cache inside the graph (docs/specs/SPEC-retrieval-cache.md): L2 replays
the reranked chunks for a repeated standalone question, L1 replays the whole
classification for a repeated first-turn question, and every bypass rule holds. The
store is the in-memory fake; no Redis, Qdrant or provider is touched."""

from __future__ import annotations

import asyncio
import json
import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import pytest
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from app.core.config import settings
from app.core.observability.graph_trace import GraphTrace
from app.core.usage.usage_recorder import UsageRecorder
from app.graph.nodes import retrieval_cache as node
from app.graph.streaming import TokenSink
from app.graph.streaming_graph import _background_searches, run_graph
from app.graph.streaming_state import GraphInput, GraphModels
from app.integrations import retrieval_cache as rc
from app.integrations.retrieval_cache import RedisRetrievalCache
from app.schemas.chat_history import HistoryMessage
from app.schemas.retrieval import RetrievedChunk
from app.schemas.security import AcademicSecurityContext, DepartmentAccessEntry
from tests.fakes.fake_redis import FakeRedisStore
from tests.llm_mocks import RetrieveManyMixin

_TRACE = GraphTrace(conversation_id="c1", message_id="m1", user_id=None, client_ip=None)

CHUNKS = [
    RetrievedChunk(
        chunk_id="doc-a:0", content="Điều kiện tốt nghiệp ...", source="a.pdf", score=0.9
    ),
    RetrievedChunk(chunk_id="doc-b:3", content="Quy chế đào tạo ...", source="b.pdf", score=0.8),
]


@dataclass
class _CountingRetrieval(RetrieveManyMixin):
    """Counts searches and id-fetches; `missing` ids never come back from `fetch`."""

    chunks: list[RetrievedChunk] = field(default_factory=lambda: list(CHUNKS))
    searches: int = 0
    fetches: list[list[str]] = field(default_factory=list)
    missing: set[str] = field(default_factory=set)

    def retrieve(
        self, query: str, *, security: AcademicSecurityContext, limit: int | None = None
    ) -> list[RetrievedChunk]:
        del query, security
        self.searches += 1
        return self.chunks if limit is None else self.chunks[:limit]

    def fetch(
        self, chunk_ids: list[str], *, security: AcademicSecurityContext
    ) -> list[RetrievedChunk]:
        del security
        self.fetches.append(list(chunk_ids))
        by_id = {chunk.chunk_id: chunk for chunk in self.chunks}
        return [by_id[i] for i in chunk_ids if i in by_id and i not in self.missing]


def _counting_classifier(
    calls: list[int], *, metadata_updates: dict[str, str] | None = None
) -> FunctionModel:
    def function(messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
        calls.append(1)
        prompt = ""
        for part in messages[-1].parts:
            content = getattr(part, "content", None)
            if isinstance(content, str):
                prompt = content
        question = prompt.split("\n", 1)[0]
        payload = {
            "tasks": [
                {
                    "intent": "academic_advisory",
                    "query": question,
                    "routing_mode": "SINGLE",
                    "web": {"topics": ["quy_che_thu_tuc"], "subject": "điều kiện tốt nghiệp"},
                }
            ],
            "confidence": 0.9,
            "metadata_updates": metadata_updates or {},
        }
        return ModelResponse(parts=[TextPart(content=json.dumps(payload, ensure_ascii=False))])

    return FunctionModel(function=function)


def _sink() -> TokenSink:
    async def sink(_token: str) -> None:
        return None

    return sink


@pytest.fixture
def store(monkeypatch: pytest.MonkeyPatch) -> FakeRedisStore:
    fake = FakeRedisStore()
    cache = RedisRetrievalCache(
        "redis://unused", client_factory=fake.client, ttl_seconds=600, rng=random.Random(3)
    )
    monkeypatch.setattr(node, "default_cache", lambda: cache)
    monkeypatch.setattr(settings, "CHAT_RERANK_SCORE_THRESHOLD", 0.0)
    return fake


def _input(
    message: str, *, history: Sequence[HistoryMessage] = (), dept: str | None = None
) -> GraphInput:
    security = AcademicSecurityContext(
        role="SINH_VIEN" if dept else "KHACH",
        department_access=[DepartmentAccessEntry(department_id=dept, access_level=2)]
        if dept
        else [],
    )
    return GraphInput(
        conversation_id="c1",
        user_message=message,
        is_first_turn=False,
        security=security,
        history=list(history),
    )


async def _turn(
    graph_input: GraphInput,
    retrieval: _CountingRetrieval,
    classifier: FunctionModel,
    mock_sync_llm_model: Callable[[str], FunctionModel],
    mock_streaming_llm_model: Callable[[Sequence[str]], FunctionModel],
) -> str:
    models = GraphModels(
        classification=classifier,
        query_transformation=mock_sync_llm_model(
            "Điều kiện tốt nghiệp là gì?\n\nSinh viên tốt nghiệp khi ..."
        ),
        generation=mock_streaming_llm_model(["Trả lời [1]."]),
        retrieval=retrieval,
    )
    result = await run_graph(
        graph_input, models, _sink(), _TRACE, UsageRecorder(request_id="t", purpose="CHAT")
    )
    # Cache writes run in the background: let them land before the test asserts.
    if _background_searches:
        await asyncio.gather(*list(_background_searches), return_exceptions=True)
    return result.response_text


def _stats(store: FakeRedisStore) -> dict[str, int]:
    return {
        key.split(":", 4)[-1]: int(value)
        for key, value in store.values.items()
        if key.startswith("rc:v1:stats:")
    }


FOLLOW_UP = [
    HistoryMessage(role="USER", content="Học phí?"),
    HistoryMessage(role="ASSISTANT", content="..."),
]


@pytest.mark.asyncio
async def test_l2_replays_the_third_identical_standalone_question(
    store: FakeRedisStore,
    mock_sync_llm_model: Callable[[str], FunctionModel],
    mock_streaming_llm_model: Callable[[Sequence[str]], FunctionModel],
) -> None:
    retrieval = _CountingRetrieval()
    calls: list[int] = []
    # Follow-up turns (history present): L1 is skipped, only L2 applies.
    for _ in range(3):
        text = await _turn(
            _input("còn khóa 2025?", history=FOLLOW_UP),
            retrieval,
            _counting_classifier(calls),
            mock_sync_llm_model,
            mock_streaming_llm_model,
        )
        assert text == "Trả lời [1]."

    assert len(calls) == 3  # the classifier still runs on every follow-up
    assert retrieval.searches == 2  # third turn served from the cache
    assert retrieval.fetches == [["doc-a:0", "doc-b:3"]]
    assert _stats(store) == {"l2:miss_first": 1, "l2:miss_second": 1, "l2:hit": 1}
    entries = [k for k in store.values if k.startswith("rc:v1:l2:")]
    assert len(entries) == 1
    assert store.sets[rc.document_key("doc-a")] == {entries[0].encode()}


@pytest.mark.asyncio
async def test_l1_replays_the_classifier_on_the_third_first_turn_question(
    store: FakeRedisStore,
    mock_sync_llm_model: Callable[[str], FunctionModel],
    mock_streaming_llm_model: Callable[[Sequence[str]], FunctionModel],
) -> None:
    retrieval = _CountingRetrieval()
    calls: list[int] = []
    for message in (
        "Điều kiện tốt nghiệp là gì?",
        "điều kiện tốt nghiệp là gì",
        "Điều kiện tốt nghiệp là gì ?",
    ):
        text = await _turn(
            _input(message),
            retrieval,
            _counting_classifier(calls),
            mock_sync_llm_model,
            mock_streaming_llm_model,
        )
        assert text == "Trả lời [1]."

    assert len(calls) == 2  # third turn: no classifier call
    assert retrieval.searches == 2
    assert retrieval.fetches == [["doc-a:0", "doc-b:3"]]
    stats = _stats(store)
    assert stats["l1:hit"] == 1 and stats["l1:miss_second"] == 1 and stats["l1:miss_first"] == 1
    # The first turns also fed L2 (same standalone question), but L1 answered first on turn 3.
    assert stats["l2:miss_first"] == 1 and stats["l2:miss_second"] == 1
    assert "l2:hit" not in stats


@pytest.mark.asyncio
async def test_scope_separates_guest_and_department_entries(
    store: FakeRedisStore,
    mock_sync_llm_model: Callable[[str], FunctionModel],
    mock_streaming_llm_model: Callable[[Sequence[str]], FunctionModel],
) -> None:
    retrieval = _CountingRetrieval()
    calls: list[int] = []
    for _ in range(2):
        await _turn(
            _input("Điều kiện tốt nghiệp là gì?"),
            retrieval,
            _counting_classifier(calls),
            mock_sync_llm_model,
            mock_streaming_llm_model,
        )
    # A CNTT student asking the same thing: a different key, so a miss (and a search).
    await _turn(
        _input("Điều kiện tốt nghiệp là gì?", dept="CNTT"),
        retrieval,
        _counting_classifier(calls),
        mock_sync_llm_model,
        mock_streaming_llm_model,
    )
    assert retrieval.searches == 3
    assert (
        len([k for k in store.values if k.startswith("rc:v1:l1:")]) == 1
    )  # only the guest entry is written
    assert _stats(store)["l1:miss_first"] == 2


@pytest.mark.asyncio
async def test_stale_entry_is_dropped_and_recomputed(
    store: FakeRedisStore,
    mock_sync_llm_model: Callable[[str], FunctionModel],
    mock_streaming_llm_model: Callable[[Sequence[str]], FunctionModel],
) -> None:
    retrieval = _CountingRetrieval()
    calls: list[int] = []
    for _ in range(2):
        await _turn(
            _input("Điều kiện tốt nghiệp là gì?"),
            retrieval,
            _counting_classifier(calls),
            mock_sync_llm_model,
            mock_streaming_llm_model,
        )
    retrieval.missing.add("doc-b:3")  # the chunk was deleted (or became invisible) meanwhile
    await _turn(
        _input("Điều kiện tốt nghiệp là gì?"),
        retrieval,
        _counting_classifier(calls),
        mock_sync_llm_model,
        mock_streaming_llm_model,
    )

    assert retrieval.searches == 3  # recomputed, not served stale
    assert _stats(store)["l1:stale"] == 1
    assert not [k for k in store.values if k.startswith("rc:v1:l1:")]  # the stale entry went


@pytest.mark.asyncio
async def test_profile_update_turn_is_never_written_to_l1(
    store: FakeRedisStore,
    mock_sync_llm_model: Callable[[str], FunctionModel],
    mock_streaming_llm_model: Callable[[Sequence[str]], FunctionModel],
) -> None:
    retrieval = _CountingRetrieval()
    calls: list[int] = []
    classifier = _counting_classifier(calls, metadata_updates={"khoa_nhap_hoc": "2022"})
    for _ in range(3):
        await _turn(
            _input("em khóa 2022, điều kiện tốt nghiệp là gì?"),
            retrieval,
            classifier,
            mock_sync_llm_model,
            mock_streaming_llm_model,
        )
    assert len(calls) == 3
    assert not [k for k in store.values if k.startswith("rc:v1:l1:")]
    assert "l1:miss_first" not in _stats(store)
    # L2 is unaffected by the profile rule: the standalone question still repeats.
    assert retrieval.searches == 2


@pytest.mark.asyncio
async def test_kill_switch_bypasses_both_layers(
    store: FakeRedisStore,
    mock_sync_llm_model: Callable[[str], FunctionModel],
    mock_streaming_llm_model: Callable[[Sequence[str]], FunctionModel],
) -> None:
    store.values[rc.DISABLED_KEY] = b"1"
    retrieval = _CountingRetrieval()
    calls: list[int] = []
    for _ in range(3):
        await _turn(
            _input("Điều kiện tốt nghiệp là gì?"),
            retrieval,
            _counting_classifier(calls),
            mock_sync_llm_model,
            mock_streaming_llm_model,
        )
    assert retrieval.searches == 3
    assert not [k for k in store.values if k.startswith("rc:v1:l1:") or k.startswith("rc:v1:l2:")]
    assert _stats(store) == {"l2:bypass": 3}


@pytest.mark.asyncio
async def test_redis_outage_is_a_plain_miss(
    store: FakeRedisStore,
    mock_sync_llm_model: Callable[[str], FunctionModel],
    mock_streaming_llm_model: Callable[[Sequence[str]], FunctionModel],
) -> None:
    store.fail = True
    retrieval = _CountingRetrieval()
    calls: list[int] = []
    text = await _turn(
        _input("Điều kiện tốt nghiệp là gì?"),
        retrieval,
        _counting_classifier(calls),
        mock_sync_llm_model,
        mock_streaming_llm_model,
    )
    assert text == "Trả lời [1]."
    assert retrieval.searches == 1


@pytest.mark.asyncio
async def test_cache_off_means_no_store_traffic(
    monkeypatch: pytest.MonkeyPatch,
    mock_sync_llm_model: Callable[[str], FunctionModel],
    mock_streaming_llm_model: Callable[[Sequence[str]], FunctionModel],
) -> None:
    monkeypatch.setattr(node, "default_cache", lambda: None)
    retrieval = _CountingRetrieval()
    calls: list[int] = []
    for _ in range(3):
        await _turn(
            _input("Điều kiện tốt nghiệp là gì?"),
            retrieval,
            _counting_classifier(calls),
            mock_sync_llm_model,
            mock_streaming_llm_model,
        )
    assert retrieval.searches == 3 and len(calls) == 3


@pytest.mark.asyncio
async def test_generation_bump_changes_the_key_for_the_affected_scope_only(
    store: FakeRedisStore,
    mock_sync_llm_model: Callable[[str], FunctionModel],
    mock_streaming_llm_model: Callable[[Sequence[str]], FunctionModel],
) -> None:
    retrieval = _CountingRetrieval()
    calls: list[int] = []
    for _ in range(2):
        await _turn(
            _input("Điều kiện tốt nghiệp là gì?", dept="CNTT"),
            retrieval,
            _counting_classifier(calls),
            mock_sync_llm_model,
            mock_streaming_llm_model,
        )
        await _turn(
            _input("Điều kiện tốt nghiệp là gì?"),
            retrieval,
            _counting_classifier(calls),
            mock_sync_llm_model,
            mock_streaming_llm_model,
        )
    assert retrieval.searches == 4
    # New CNTT-only content: CNTT keys change, the guest key does not.
    store.values[rc.generation_key("dept:CNTT")] = b"1"
    await _turn(
        _input("Điều kiện tốt nghiệp là gì?", dept="CNTT"),
        retrieval,
        _counting_classifier(calls),
        mock_sync_llm_model,
        mock_streaming_llm_model,
    )
    assert retrieval.searches == 5
    await _turn(
        _input("Điều kiện tốt nghiệp là gì?"),
        retrieval,
        _counting_classifier(calls),
        mock_sync_llm_model,
        mock_streaming_llm_model,
    )
    assert retrieval.searches == 5  # guest: cache hit
