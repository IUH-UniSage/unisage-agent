"""The service's HTTP contract with a scorer double (no model, no GPU)."""

from __future__ import annotations

import asyncio

import pytest
from httpx import ASGITransport, AsyncClient

from reranker.app import Batcher, create_app
from reranker.scorer import PairScores, ScoreResult


class FakeScorer:
    model_name = "fake/reranker"
    device = "cpu"

    def __init__(self) -> None:
        self.calls: list[tuple[str, list[str]]] = []
        self.closed = False

    def score(self, query: str, passages: list[str]) -> ScoreResult:
        result = self.score_pairs([(query, p) for p in passages])
        return ScoreResult(scores=result.scores, truncated=sum(result.truncated), took_ms=3)

    def score_pairs(self, pairs: list[tuple[str, str]]) -> PairScores:
        # Longer passages score higher; the first pair of each call "truncated".
        self.calls.append((pairs[0][0], [p for _, p in pairs]))
        return PairScores(
            scores=[float(len(p)) for _, p in pairs],
            truncated=[i == 0 for i in range(len(pairs))],
            took_ms=3,
        )

    def close(self) -> None:
        self.closed = True


@pytest.fixture
def fake() -> FakeScorer:
    return FakeScorer()


@pytest.fixture
async def client(fake: FakeScorer):
    app = create_app(lambda: fake)
    async with app.router.lifespan_context(app):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
            yield c


async def test_rerank_scores_follow_input_order_and_ranking_is_best_first(client, fake):
    response = await client.post(
        "/rerank", json={"query": "học phí", "passages": ["a", "ccc", "bb"]}
    )

    assert response.status_code == 200
    body = response.json()
    assert body["scores"] == [1.0, 3.0, 2.0]
    assert body["ranking"] == [1, 2, 0]
    assert body["truncated"] == 1 and body["model"] == "fake/reranker"
    assert body["batched"] == 1
    assert fake.calls == [("học phí", ["a", "ccc", "bb"])]


async def test_top_k_cuts_the_ranking_not_the_scores(client):
    body = (
        await client.post(
            "/rerank", json={"query": "q", "passages": ["a", "ccc", "bb"], "top_k": 1}
        )
    ).json()

    assert body["ranking"] == [1]
    assert len(body["scores"]) == 3


async def test_ties_keep_input_order(client):
    body = (await client.post("/rerank", json={"query": "q", "passages": ["xx", "yy"]})).json()

    assert body["ranking"] == [0, 1]


async def test_empty_passages_is_a_400(client):
    response = await client.post("/rerank", json={"query": "q", "passages": []})

    assert response.status_code == 422


async def test_health_reports_the_loaded_model(client):
    body = (await client.get("/health")).json()

    assert body == {
        "status": "ok",
        "model": "fake/reranker",
        "device": "cpu",
        "loaded": True,
        "vram_used_mb": None,
        "batches": 0,
        "requests": 0,
    }


async def test_shutdown_closes_the_scorer(fake):
    app = create_app(lambda: fake)
    async with app.router.lifespan_context(app):
        assert fake.closed is False
    assert fake.closed is True


async def test_concurrent_requests_share_a_forward_pass_and_get_their_own_scores(fake):
    app = create_app(lambda: fake, batch_window_ms=50)
    async with app.router.lifespan_context(app):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
            bodies = await asyncio.gather(
                c.post("/rerank", json={"query": "q1", "passages": ["a", "ccc"]}),
                c.post("/rerank", json={"query": "q2", "passages": ["bbbb"]}),
                c.post("/rerank", json={"query": "q3", "passages": ["dd", "e", "ffffff"]}),
            )
            health = (await c.get("/health")).json()

    scores = [b.json()["scores"] for b in bodies]
    assert scores == [[1.0, 3.0], [4.0], [2.0, 1.0, 6.0]]
    assert [b.json()["ranking"] for b in bodies] == [[1, 0], [0], [2, 0, 1]]
    # One forward pass for all three (6 pairs), each response says so.
    assert len(fake.calls) == 1 and len(fake.calls[0][1]) == 6
    assert {b.json()["batched"] for b in bodies} == {3}
    # Only the first pair of the batch was "truncated": it belongs to request 1.
    assert [b.json()["truncated"] for b in bodies] == [1, 0, 0]
    assert health["batches"] == 1 and health["requests"] == 3


async def test_batch_is_cut_at_max_pairs():
    fake = FakeScorer()
    batcher = Batcher(fake, window_ms=50, max_pairs=3)
    batcher.start()
    try:
        results = await asyncio.gather(
            batcher.submit("q1", ["a", "bb"]),
            batcher.submit("q2", ["ccc"]),
            batcher.submit("q3", ["dddd"]),
        )
    finally:
        await batcher.stop()

    assert [r.scores for r in results] == [[1.0, 2.0], [3.0], [4.0]]
    # 2 + 1 pairs fill the first batch; the third request runs in a second one.
    assert [len(call[1]) for call in fake.calls] == [3, 1]
    assert [r.batched for r in results] == [2, 2, 1]


async def test_a_scorer_error_fails_every_request_in_the_batch():
    class Broken(FakeScorer):
        def score_pairs(self, pairs):
            raise RuntimeError("CUDA out of memory")

    app = create_app(Broken, batch_window_ms=50)
    async with app.router.lifespan_context(app):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
            responses = await asyncio.gather(
                c.post("/rerank", json={"query": "q", "passages": ["a"]}),
                c.post("/rerank", json={"query": "q", "passages": ["b"]}),
            )

    assert [r.status_code for r in responses] == [500, 500]
    assert "CUDA out of memory" in responses[0].json()["detail"]
