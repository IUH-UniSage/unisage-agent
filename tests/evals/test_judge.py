"""Task 11: which RAGAS metrics a row gets, the budget guard, the checkpoint, the quota
handling and the Gemini key pool - all without RAGAS, Gemini or a network."""

import asyncio
import json
import math
from pathlib import Path
from typing import Any

import pytest

from evals.judge import cli as judge
from evals.judge.gemini_pool import (
    CallFailed,
    GeminiCaller,
    KeyPool,
    ModelUnavailable,
    QuotaExhausted,
    classify,
)
from evals.judge.cli import JudgeRun, Score, Task


def _row(**overrides: Any) -> dict[str, Any]:
    row = {
        "id": "q1",
        "variant": "original",
        "asked": "Học phí ngành Kế toán bao nhiêu?",
        "category": "normal",
        "expect_visible": None,
        "expected_answer": "15 triệu đồng mỗi học kỳ.",
        "expected_document_ids": ["doc-a", "doc-b"],
        "response": "Học phí là 15 triệu đồng mỗi học kỳ.",
        "nodes": ["08_RetrievalFilteringNode"],
        "context": [{"document_id": "doc-a", "content": f"đoạn {i}"} for i in range(10)],
        "web_results": [],
        "citations": [],
        "error": None,
    }
    row.update(overrides)
    return row


# --- selection ------------------------------------------------------------------------------


def test_a_row_with_an_expected_answer_gets_every_answer_metric_and_refusal() -> None:
    tasks = {task.metric: task for task in judge.tasks_for(_row())}

    assert set(tasks) == {*judge.ANSWER_METRICS, "refusal"}
    assert len(tasks["faithfulness"].inputs["retrieved_contexts"]) == judge.MAX_CONTEXTS
    assert tasks["id_context_recall"].inputs == {
        "retrieved_context_ids": ["doc-a"],
        "reference_context_ids": ["doc-a", "doc-b"],
    }
    assert tasks["answer_coverage"].inputs == {
        "response": "Học phí là 15 triệu đồng mỗi học kỳ.",
        "reference": "15 triệu đồng mỗi học kỳ.",
    }


@pytest.mark.parametrize(
    "overrides",
    [
        {"category": "unanswerable", "expected_answer": None},
        {"category": "access", "expect_visible": False},
        {"category": "normal", "expected_answer": ""},
    ],
)
def test_rows_without_an_answer_to_check_only_get_the_refusal_metric(overrides: dict) -> None:
    assert [task.metric for task in judge.tasks_for(_row(**overrides))] == ["refusal"]


def test_an_answer_from_the_web_is_judged_against_the_fetched_pages() -> None:
    row = _row(
        used_web_search=True,
        citations=[{"sourceType": "WEB"}],
        web_results=[{"url": "https://iuh.edu.vn/a", "content": "trang web"}],
    )

    tasks = {task.metric: task for task in judge.tasks_for(row)}

    assert set(tasks) == {"web_faithfulness", "answer_coverage", "refusal"}
    assert tasks["web_faithfulness"].inputs["retrieved_contexts"] == ["trang web"]


def test_failed_turns_get_no_metric_and_metrics_can_be_chosen() -> None:
    assert judge.tasks_for(_row(error="boom")) == []
    assert judge.tasks_for(_row(response="")) == []
    chosen = judge.tasks_for(_row(), ["id_context_precision", "id_context_recall"])
    assert [task.metric for task in chosen] == ["id_context_precision", "id_context_recall"]


# --- budget ---------------------------------------------------------------------------------


def test_the_estimate_counts_only_llm_metrics_and_the_guard_stops_over_the_cap() -> None:
    tasks = judge.tasks_for(_row())
    estimated = judge.estimate(tasks, judge.DEFAULT_MODEL)

    assert (
        estimated["calls"] == 2 + 4 + 1 + 1 + 1
    )  # faithfulness, factual, recall, relevancy, refusal
    assert estimated["usd"] > 0
    assert (
        judge.estimate(judge.tasks_for(_row(), ["id_context_recall"]), judge.DEFAULT_MODEL)["usd"]
        == 0
    )
    with pytest.raises(judge.BudgetExceeded):
        judge.check_budget(estimated, max_cost=estimated["usd"] / 2)


def _results(tmp_path: Path, rows: list[dict[str, Any]]) -> Path:
    (tmp_path / "results.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
    )
    return tmp_path


def test_main_stops_before_any_call_when_the_estimate_exceeds_max_cost(tmp_path: Path) -> None:
    out = _results(tmp_path, [_row(id=f"q{i}") for i in range(5)])

    code = judge.main(["--out", str(out), "--max-cost", "0.000001", "--env-file", "none"])

    assert code == 2
    assert not (out / "judge.jsonl").exists()


# --- the run --------------------------------------------------------------------------------


class FakeScorer:
    def __init__(
        self,
        values: dict[str, float] | None = None,
        quota_after: int | None = None,
        fallback: bool = False,
    ) -> None:
        self.values = values or {}
        self.quota_after = quota_after
        self.fallback = fallback
        self.calls: list[tuple[str, str]] = []

    async def score(self, task: Task) -> Score:
        if self.quota_after is not None and len(self.calls) >= self.quota_after:
            raise QuotaExhausted(reset_at=1_000.0)
        self.calls.append((task.key, task.metric))
        value = self.values.get(task.metric, 0.9)
        return Score(
            value=value,
            reason=None if not math.isnan(value) else "failed",
            usage={
                "model": "fallback" if self.fallback else "main",
                "models": {"fallback" if self.fallback else "main": 1},
                "fallback": self.fallback,
                "calls": 1,
                "usd": 0.001,
            },
        )

    def stats(self) -> dict[str, Any]:
        return {}


def _run(out: Path, scorers: list[FakeScorer], **kwargs: Any) -> tuple[JudgeRun, list[bool]]:
    asked: list[bool] = []

    def factory(primary_only: bool) -> FakeScorer:
        asked.append(primary_only)
        return scorers[min(len(asked), len(scorers)) - 1]

    return JudgeRun(out=out, scorer_factory=factory, concurrency=2, **kwargs), asked


def test_every_score_is_checkpointed_and_a_rerun_only_scores_what_is_missing(
    tmp_path: Path,
) -> None:
    tasks = judge.tasks_of([_row(id="q1"), _row(id="q2")], ["refusal", "faithfulness"])
    scorer = FakeScorer(values={"faithfulness": math.nan})
    run, _ = _run(tmp_path, [scorer])

    assert asyncio.run(run.run(tasks))
    records = judge.load_records(tmp_path / "judge.jsonl")
    assert len(records) == 4
    assert records[("q1:original", "faithfulness")]["value"] is None  # NaN, tried once only
    assert len(scorer.calls) == 4

    # the next run picks up only the NaN scores
    assert [(t.key, t.metric) for t in judge.pending_tasks(tasks, records)] == [
        ("q1:original", "faithfulness"),
        ("q2:original", "faithfulness"),
    ]


def test_out_of_quota_stops_with_the_checkpoint_when_not_allowed_to_wait(tmp_path: Path) -> None:
    tasks = judge.tasks_of([_row(id=f"q{i}") for i in range(3)], ["refusal"])
    run, _ = _run(tmp_path, [FakeScorer(quota_after=1)])
    run.concurrency = 1

    assert asyncio.run(run.run(tasks)) is False
    assert len(judge.load_records(tmp_path / "judge.jsonl")) == 1


def test_out_of_quota_waits_for_the_reset_then_scores_the_rest(tmp_path: Path) -> None:
    tasks = judge.tasks_of([_row(id=f"q{i}") for i in range(3)], ["refusal"])
    scorer = FakeScorer(quota_after=1)
    slept: list[float] = []

    async def sleep(seconds: float) -> None:
        slept.append(seconds)
        scorer.quota_after = None  # quota is back after the wait

    run, _ = _run(tmp_path, [scorer], quota_wait_s=3600, sleep=sleep, clock=lambda: 900.0)
    run.concurrency = 1

    assert asyncio.run(run.run(tasks))
    assert slept == [160.0]  # until the reset (1000) + 60 s margin
    assert len(judge.load_records(tmp_path / "judge.jsonl")) == 3


def test_rows_scored_by_the_fallback_are_scored_again_with_the_main_model(tmp_path: Path) -> None:
    tasks = judge.tasks_of([_row(id=f"q{i}") for i in range(4)], ["refusal", "id_context_recall"])
    run, asked = _run(
        tmp_path, [FakeScorer(fallback=True), FakeScorer(fallback=False)], fallback_model="fallback"
    )

    assert asyncio.run(run.run(tasks))
    assert asked == [False, True]
    records = judge.load_records(tmp_path / "judge.jsonl")
    assert all(not r["fallback"] for (k, m), r in records.items() if m == "refusal")


def test_scores_are_merged_into_results_and_summarised(tmp_path: Path) -> None:
    rows = [_row(id="q1"), _row(id="q2", category="unanswerable", expected_answer=None)]
    out = _results(tmp_path, rows)
    run, _ = _run(out, [FakeScorer(values={"refusal": 0.0, "faithfulness": 0.5})])
    asyncio.run(run.run(judge.tasks_of(rows, ["refusal", "faithfulness"])))

    records = judge.load_records(out / "judge.jsonl")
    assert judge.merge_into_results(out, records) == 2
    merged = judge.load_rows(out)
    assert merged[0]["ragas"]["faithfulness"]["value"] == 0.5
    assert (out / "results.jsonl.bak").exists()

    summary = judge.summarize_scores(merged, records)
    assert summary["metrics"]["faithfulness"] == {
        "n": 1,
        "mean": 0.5,
        "pass_rate": 0.0,
        "nan": 0,
        "by_category": {"normal": {"n": 1, "mean": 0.5, "pass_rate": 0.0}},
    }
    assert summary["usage"]["calls"] == 3


def test_keys_come_from_the_environment_or_the_env_file(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEYS", raising=False)
    env = tmp_path / ".env.eval"
    env.write_text('OTHER=1\nGEMINI_API_KEYS="k1, k2\tk3"\n', encoding="utf-8")
    assert judge.read_keys(env) == ["k1", "k2", "k3"]
    monkeypatch.setenv("GEMINI_API_KEYS", "a;b")
    assert judge.read_keys(env) == ["a", "b"]


# --- key pool -------------------------------------------------------------------------------


class Clock:
    def __init__(self) -> None:
        self.now = 0.0
        self.slept: list[float] = []

    def __call__(self) -> float:
        return self.now

    async def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.now += seconds


def _pool(keys: list[str], clock: Clock, rpm: int = 2) -> KeyPool:
    return KeyPool(keys, rpm=rpm, clock=clock, wall=clock, sleep=clock.sleep)


def test_each_key_is_held_to_its_rpm_and_keys_are_used_round_robin() -> None:
    clock = Clock()
    pool = _pool(["key-one-123456", "key-two-123456"], clock)

    async def go() -> list[str]:
        return [(await pool.acquire()).key for _ in range(5)]

    used = asyncio.run(go())
    assert used[:4] == ["key-one-123456", "key-two-123456", "key-one-123456", "key-two-123456"]
    assert clock.slept and clock.now >= 60  # the 5th call waits for the first slot to free


def test_a_key_out_for_the_day_sits_out_and_no_key_left_raises() -> None:
    clock = Clock()
    pool = _pool(["key-one-123456", "key-two-123456"], clock)
    pool.exhaust(pool.keys[0], until=10_000)

    async def go() -> None:
        assert (await pool.acquire()).key == "key-two-123456"
        pool.exhaust(pool.keys[1], until=20_000)
        with pytest.raises(QuotaExhausted) as raised:
            await pool.acquire()
        assert raised.value.reset_at == 10_000

    asyncio.run(go())


class FakeError(Exception):
    def __init__(self, code: int, details: Any = None) -> None:
        super().__init__(f"{code} error")
        self.code = code
        self.details = details


def test_errors_are_classified_from_the_code_and_the_quota_details() -> None:
    per_day = {
        "error": {
            "details": [{"violations": [{"quotaId": "GenerateRequestsPerDayPerProjectPerModel"}]}]
        }
    }
    per_minute = {"error": {"details": [{"@type": "RetryInfo", "retryDelay": "34s"}]}}
    assert classify(FakeError(429, per_day)) == ("rpd", None)
    assert classify(FakeError(429, per_minute)) == ("rpm", 34.0)
    assert classify(FakeError(503)) == ("unavailable", None)
    assert classify(FakeError(504)) == ("timeout", None)
    assert classify(asyncio.TimeoutError()) == ("timeout", None)
    assert classify(FakeError(403)) == ("denied", None)
    assert classify(FakeError(401)) == ("denied", None)
    assert classify(FakeError(400)) == ("fatal", None)


def _caller(
    keys: list[str], clock: Clock, outcomes: dict[str, list[Any]], **options: Any
) -> GeminiCaller:
    """A caller whose client for a key is the key itself (see `_call`)."""

    return GeminiCaller(
        pools={"m": _pool(keys, clock, rpm=10)},
        client_for=lambda key: key,
        next_reset=lambda: 99_999.0,
        wall=clock,
        **options,
    )


def _call(caller: GeminiCaller, outcomes: dict[str, list[Any]]) -> Any:
    """`outcomes[key]` is consumed per call: an exception to raise or a value to return."""

    async def request(client: str) -> Any:
        outcome = outcomes[client].pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    return asyncio.run(caller.call("m", request))


def test_429_per_minute_moves_to_another_key_and_per_day_retires_the_key() -> None:
    clock = Clock()
    outcomes = {
        "key-one-123456": [FakeError(429, {"retryDelay": "20s"}), "ok"],
        "key-two-123456": [FakeError(429, {"quotaId": "PerDay"})],
    }
    caller = _caller(list(outcomes), clock, outcomes)

    response, state = _call(caller, outcomes)

    assert response == "ok"
    pool = caller.pools["m"]
    assert pool.keys[0].stats["rpm"] == 1 and pool.keys[0].cooldown_until == 20.0
    assert pool.keys[1].stats["rpd"] == 1 and pool.keys[1].exhausted_until == 99_999.0
    assert state.key == "key-one-123456"  # back on the first key after its 20 s cool-down


def test_per_day_429_can_bench_a_key_only_until_a_recheck() -> None:
    clock = Clock()
    outcomes = {"key-one-123456": [FakeError(429, {"quotaId": "PerDay"}), "ok"]}
    caller = _caller(list(outcomes), clock, outcomes, rpd_recheck_s=600)

    with pytest.raises(QuotaExhausted) as raised:
        _call(caller, outcomes)
    assert raised.value.reset_at == 600.0

    clock.now = 601.0
    assert _call(caller, outcomes)[0] == "ok"


def test_503_and_504_cool_the_key_down_and_move_on_to_another_key() -> None:
    clock = Clock()
    outcomes = {
        "key-one-123456": [FakeError(503), FakeError(504), "ok"],
        "key-two-123456": [FakeError(503), FakeError(504)],
    }
    caller = _caller(list(outcomes), clock, outcomes)

    response, state = _call(caller, outcomes)

    assert response == "ok" and state.key == "key-one-123456"
    keys = caller.pools["m"].keys
    assert [k.stats["unavailable"] for k in keys] == [1, 1]
    assert [k.stats["timeout"] for k in keys] == [1, 1]
    assert outcomes == {"key-one-123456": [], "key-two-123456": []}


def test_the_model_is_given_up_after_too_many_503_and_a_bad_request_fails_at_once() -> None:
    clock = Clock()
    outcomes = {"key-one-123456": [FakeError(503)] * 3, "key-two-123456": [FakeError(503)] * 3}
    with pytest.raises(ModelUnavailable):
        _call(_caller(list(outcomes), clock, outcomes, max_unavailable=5), outcomes)

    outcomes = {"key-one-123456": [FakeError(504)] * 3}
    with pytest.raises(CallFailed):
        _call(_caller(list(outcomes), clock, outcomes, max_timeouts=2), outcomes)

    outcomes = {"key-one-123456": [FakeError(400)]}
    with pytest.raises(CallFailed):
        _call(_caller(list(outcomes), clock, outcomes), outcomes)


def test_a_denied_key_is_dropped_and_the_call_moves_on_to_another_key() -> None:
    clock = Clock()
    denied = FakeError(403, {"error": {"message": "Your project has been denied access."}})
    outcomes = {"key-one-123456": [denied], "key-two-123456": ["ok", "ok"]}
    caller = _caller(list(outcomes), clock, outcomes)

    assert _call(caller, outcomes) == ("ok", caller.pools["m"].keys[1])
    assert _call(caller, outcomes)[0] == "ok"  # the denied key is not tried again

    keys = caller.pools["m"].keys
    assert keys[0].denied and keys[0].stats["denied"] == 1
    assert outcomes == {"key-one-123456": [], "key-two-123456": []}


def test_no_key_left_after_every_one_is_denied_fails_the_call() -> None:
    clock = Clock()
    outcomes = {"key-one-123456": [FakeError(403)], "key-two-123456": [FakeError(401)]}
    with pytest.raises(CallFailed, match="denied"):
        _call(_caller(list(outcomes), clock, outcomes), outcomes)


def test_an_embedding_call_is_not_counted_as_the_fallback_model() -> None:
    records = {
        (f"q{i}:original", "answer_relevancy"): {
            "value": 0.9,
            "fallback": True,  # written before the fix
            "models": {"main": 1, "gemini-embedding-001": 2},
        }
        for i in range(10)
    }

    assert judge.fallback_rows(records, "fallback") == set()
    records[("q0:original", "refusal")] = {"value": 1.0, "models": {"fallback": 1}}
    records[("q1:original", "refusal")] = {"value": 1.0, "models": {"fallback": 1}}
    assert judge.fallback_rows(records, "fallback") == {"q0:original", "q1:original"}


def test_answer_coverage_keeps_precision_and_f1_as_extra_values(tmp_path: Path) -> None:
    rows = [_row(id="q1")]
    out = _results(tmp_path, rows)

    class Coverage(FakeScorer):
        async def score(self, task: Task) -> Score:
            score = await super().score(task)
            score.extra = {"answer_precision": 0.0, "factual_correctness_f1": math.nan}
            return score

    run, _ = _run(out, [Coverage(values={"answer_coverage": 1.0})])
    asyncio.run(run.run(judge.tasks_of(rows, ["answer_coverage"])))
    records = judge.load_records(out / "judge.jsonl")

    judge.merge_into_results(out, records)
    ragas = judge.load_rows(out)[0]["ragas"]
    assert ragas["answer_coverage"]["value"] == 1.0
    assert ragas["answer_precision"]["value"] == 0.0
    assert ragas["factual_correctness_f1"]["value"] is None
    summary = judge.summarize_scores(rows, records)["metrics"]
    assert summary["answer_coverage"]["mean"] == 1.0
    assert summary["answer_precision"]["mean"] == 0.0
    assert summary["factual_correctness_f1"] == {"n": 0, "nan": 1, "by_category": {}}
