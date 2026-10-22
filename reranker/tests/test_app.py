"""The service's HTTP contract with a scorer double (no model, no GPU)."""

from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

from reranker.app import create_app
from reranker.scorer import ScoreResult


class FakeScorer:
    model_name = "fake/reranker"
    device = "cpu"

    def __init__(self) -> None:
        self.calls: list[tuple[str, list[str]]] = []
        self.closed = False

    def score(self, query: str, passages: list[str]) -> ScoreResult:
        self.calls.append((query, passages))
        # Longer passages score higher; one "truncated" to exercise the counter.
        return ScoreResult(scores=[float(len(p)) for p in passages], truncated=1, took_ms=3)

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
    }


async def test_shutdown_closes_the_scorer(fake):
    app = create_app(lambda: fake)
    async with app.router.lifespan_context(app):
        assert fake.closed is False
    assert fake.closed is True
