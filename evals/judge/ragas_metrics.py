"""RAGAS metrics for `evals.judge` (Task 11), on whichever provider adapter the run picked
(`evals.judge.providers`). Needs the judge environment (`evals/judge/requirements.txt`).

The metrics are RAGAS' own (`ragas.metrics.collections`: Faithfulness, FactualCorrectness,
ContextRecall, AnswerRelevancy; ID-based context precision/recall; a `DiscreteMetric` for
refusal, the 0.4 form of AspectCritic). Only the model they talk to is ours: `JudgeLLM` sends
every structured-output request to the adapter, switches to the fallback model when the main
one stays unavailable, parses the JSON into the metric's model and books tokens, cost, model
and key per score (`evals.judge.ledger`).

The example parts of the RAGAS prompts are translated to Vietnamese once
(`--adapt-prompts`, RAGAS `BasePrompt.adapt`) and cached in `evals/judge/prompts/`; later runs
load the cache, so the judge reads Vietnamese examples next to Vietnamese answers.
"""

import asyncio
import json
import math
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from ragas.dataset_schema import SingleTurnSample
from ragas.embeddings.base import BaseRagasEmbedding
from ragas.llms.base import InstructorBaseRagasLLM
from ragas.metrics import DiscreteMetric
from ragas.metrics._context_precision import IDBasedContextPrecision
from ragas.metrics._context_recall import IDBasedContextRecall
from ragas.metrics.collections import (
    AnswerRelevancy,
    ContextRecall,
    FactualCorrectness,
    Faithfulness,
)
from ragas.metrics.utils import fbeta_score
from ragas.prompt.metrics.base_prompt import BasePrompt

from evals.judge import ledger
from evals.judge.providers.base import JudgeAdapter, extract_json
from evals.judge.providers.pool import CallFailed, ModelUnavailable, QuotaExhausted
from evals.judge.tasks import Score, Task

LANGUAGE = "vietnamese"
REFUSAL_PROMPT = (
    "Bạn là giám khảo chấm câu trả lời của trợ lý ảo một trường đại học.\n"
    "Câu hỏi của người dùng: {user_input}\n"
    "Câu trả lời của trợ lý: {response}\n\n"
    "Câu trả lời có TỪ CHỐI trả lời không? Từ chối là khi trợ lý nói không tìm thấy, không có "
    "thông tin, không có quyền xem, hoặc đề nghị gửi yêu cầu hỗ trợ (ticket) / liên hệ phòng ban "
    "THAY VÌ đưa ra nội dung trả lời câu hỏi. Chọn 'refused' nếu từ chối, 'answered' nếu câu trả "
    "lời có nội dung trả lời câu hỏi (kể cả khi kèm gợi ý liên hệ)."
)


class JudgeLLM(InstructorBaseRagasLLM):
    """Structured output for the RAGAS metrics from any adapter, with model fallback."""

    def __init__(self, adapter: JudgeAdapter, model: str, fallback: str | None) -> None:
        self.adapter = adapter
        self.model = model
        self.fallback = fallback

    async def agenerate(self, prompt: str, response_model: type[Any]) -> Any:
        models = [self.model, *([self.fallback] if self.fallback else [])]
        schema = response_model.model_json_schema()
        last: Exception | None = None
        for model in models:
            try:
                answer = await self.adapter.generate_json(model, prompt, schema)
            except ModelUnavailable as error:
                last = error
                continue
            ledger.book(model, answer.key, answer.input_tokens, answer.output_tokens)
            parsed = response_model.model_validate_json(extract_json(answer.text))
            ledger.step(response_model.__name__, parsed)
            return parsed
        raise last or ModelUnavailable(self.model)

    def generate(self, prompt: str, response_model: type[Any]) -> Any:
        raise NotImplementedError("the judge is async: use agenerate")


class JudgeEmbeddings(BaseRagasEmbedding):
    """Embeddings (for AnswerRelevancy) from any adapter that can embed."""

    def __init__(self, adapter: JudgeAdapter, model: str) -> None:
        super().__init__(cache=None)
        self.adapter = adapter
        self.model = model

    def embed_text(self, text: str, **kwargs: Any) -> list[float]:
        raise NotImplementedError("the judge is async: use aembed_text")

    async def aembed_text(self, text: str, **kwargs: Any) -> list[float]:
        return (await self.aembed_texts([text]))[0]

    async def aembed_texts(self, texts: list[str], **kwargs: Any) -> list[list[float]]:
        result = await self.adapter.embed(self.model, texts)
        ledger.book(self.model, result.key, result.input_tokens)
        return result.vectors


class AnswerCorrectness(FactualCorrectness):
    """RAGAS FactualCorrectness, keeping both directions of its claim checks.

    RAGAS returns one number, and its recall reuses the true positives of the answer-side
    check, so a correct answer with extra detail (`... đề tài ZABBIX, HK2 2024-2025 là ThS. X`
    against `Do ThS. X hướng dẫn.`) scores 0 in every mode. The same two checks give coverage
    (expected claims the answer holds), precision (answer claims the expected answer backs)
    and the F1 RAGAS would report."""

    async def parts(self, response: str, reference: str) -> tuple[float, float, float]:
        answer_claims = await self._decompose_and_verify_claims(response, reference)
        expected_claims = await self._decompose_and_verify_claims(reference, response)
        coverage = float(np.mean(expected_claims)) if expected_claims.size else math.nan
        precision = float(np.mean(answer_claims)) if answer_claims.size else math.nan
        tp = int(np.sum(answer_claims))
        fp = int(np.sum(~answer_claims))
        fn = int(np.sum(~expected_claims))
        return coverage, precision, float(fbeta_score(tp, fp, fn, self.beta))


# --- prompts in Vietnamese ------------------------------------------------------------------


def prompts_of(metric: Any) -> dict[str, BasePrompt]:
    return {name: value for name, value in vars(metric).items() if isinstance(value, BasePrompt)}


def _dump(prompt: BasePrompt) -> dict[str, Any]:
    return {
        "language": prompt.language,
        "instruction": prompt.instruction,
        "examples": [[i.model_dump(), o.model_dump()] for i, o in prompt.examples],
    }


def _load(prompt: BasePrompt, data: dict[str, Any]) -> None:
    prompt.language = data["language"]
    prompt.instruction = data["instruction"]
    prompt.examples = [
        (prompt.input_model.model_validate(i), prompt.output_model.model_validate(o))
        for i, o in data["examples"]
    ]


async def localize(metrics: dict[str, Any], llm: JudgeLLM, folder: Path, adapt: bool) -> int:
    """Load the cached Vietnamese prompts; with `adapt`, translate and cache what is missing.
    Returns how many prompts are in Vietnamese."""

    localized = 0
    for metric_name, metric in metrics.items():
        for attr, prompt in prompts_of(metric).items():
            path = folder / f"{metric_name}.{attr}.json"
            if path.is_file():
                _load(prompt, json.loads(path.read_text(encoding="utf-8")))
                localized += 1
            elif adapt:
                adapted = await prompt.adapt(LANGUAGE, llm)
                folder.mkdir(parents=True, exist_ok=True)
                path.write_text(
                    json.dumps(_dump(adapted), ensure_ascii=False, indent=1), encoding="utf-8"
                )
                setattr(metric, attr, adapted)
                localized += 1
    return localized


# --- scoring --------------------------------------------------------------------------------


@dataclass
class RagasScorer:
    llm: JudgeLLM
    metrics: dict[str, Any]
    adapters: list[JudgeAdapter]
    fallback_model: str | None = None
    embedding_model: str | None = None
    ready: asyncio.Event = field(default_factory=asyncio.Event)
    setup: Any = None

    async def score(self, task: Task) -> Score:
        if self.setup is not None and not self.ready.is_set():
            await self.setup(self)
        booked = ledger.Ledger(
            fallback_model=self.fallback_model, embedding_model=self.embedding_model
        )
        token = ledger.current.set(booked)
        try:
            value, reason, extra = await self._run(task)
        except QuotaExhausted:
            raise
        except (ModelUnavailable, CallFailed) as error:
            value, reason, extra = math.nan, str(error)[:300], {}
        except Exception as error:  # a metric that cannot score this row: NaN + why
            value, reason, extra = math.nan, f"{type(error).__name__}: {str(error)[:300]}", {}
        finally:
            ledger.current.reset(token)
        return Score(
            value=value, reason=reason, usage=booked.as_usage(), extra=extra, steps=booked.steps
        )

    async def _run(self, task: Task) -> tuple[float, str | None, dict[str, float]]:
        inputs = task.inputs
        name = task.metric
        if name in ("faithfulness", "web_faithfulness") and not inputs["retrieved_contexts"]:
            return math.nan, "no retrieved context", {}
        if name == "context_recall" and not inputs["retrieved_contexts"]:
            return 0.0, "no retrieved context", {}
        if name in ("id_context_precision", "id_context_recall"):
            if not inputs["reference_context_ids"]:
                return math.nan, "no expected document", {}
            if not inputs["retrieved_context_ids"]:
                empty = 0.0 if name == "id_context_recall" else math.nan
                return empty, "nothing retrieved", {}
            value = await self.metrics[name].single_turn_ascore(SingleTurnSample(**inputs))
            return float(value), None, {}
        if name == "refusal":
            result = await self.metrics[name].ascore(llm=self.llm, **inputs)
            verdict = 1.0 if result.value == "refused" else 0.0
            return verdict, getattr(result, "reason", None), {}
        if name == "answer_coverage":
            coverage, precision, f1 = await self.metrics[name].parts(**inputs)
            return coverage, None, {"answer_precision": precision, "factual_correctness_f1": f1}
        metric = self.metrics["faithfulness" if name == "web_faithfulness" else name]
        result = await metric.ascore(**inputs)
        return float(result.value), getattr(result, "reason", None), {}

    def stats(self) -> dict[str, Any]:
        return merge_stats(self.adapters)


def merge_stats(adapters: list[JudgeAdapter]) -> dict[str, Any]:
    """{model: {masked key: counts}} summed over the adapters (judge + embeddings)."""

    merged: dict[str, Any] = {}
    for adapter in dict.fromkeys(adapters):
        for model, keys in adapter.stats().items():
            bucket = merged.setdefault(model, {})
            for key, counts in keys.items():
                total = Counter(bucket.get(key, {}))
                total.update(counts)
                bucket[key] = dict(total)
    return merged


@dataclass
class RagasScorerFactory:
    """Builds the scorer `JudgeRun` asks for (main model only, when asked), on the adapters
    the CLI picked: `adapter` judges, `embedder` embeds for answer_relevancy (None when
    answer_relevancy is not scored)."""

    adapter: JudgeAdapter
    model: str
    fallback: str | None
    prompts_dir: Path
    adapt_prompts: bool
    embedder: JudgeAdapter | None = None
    embedding_model: str | None = None

    def __call__(self, primary_only: bool) -> RagasScorer:
        llm = JudgeLLM(self.adapter, self.model, None if primary_only else self.fallback)
        metrics: dict[str, Any] = {
            "faithfulness": Faithfulness(llm=llm),
            "answer_coverage": AnswerCorrectness(llm=llm, mode="f1"),
            "context_recall": ContextRecall(llm=llm),
            "id_context_precision": IDBasedContextPrecision(),
            "id_context_recall": IDBasedContextRecall(),
            "refusal": DiscreteMetric(
                name="refusal", prompt=REFUSAL_PROMPT, allowed_values=["refused", "answered"]
            ),
        }
        if self.embedder is not None and self.embedding_model:
            embeddings = JudgeEmbeddings(self.embedder, self.embedding_model)
            metrics["answer_relevancy"] = AnswerRelevancy(
                llm=llm, embeddings=embeddings, strictness=1
            )
        lock = asyncio.Lock()

        async def setup(scorer: RagasScorer) -> None:
            async with lock:
                if scorer.ready.is_set():
                    return
                llm_metrics = {k: v for k, v in metrics.items() if prompts_of(v)}
                count = await localize(llm_metrics, llm, self.prompts_dir, self.adapt_prompts)
                if not count:
                    print(
                        "RAGAS prompts in English (run once with --adapt-prompts to cache a "
                        "Vietnamese version)"
                    )
                scorer.ready.set()

        return RagasScorer(
            llm=llm,
            metrics=metrics,
            adapters=[a for a in (self.adapter, self.embedder) if a is not None],
            fallback_model=self.fallback,
            embedding_model=self.embedding_model,
            setup=setup,
        )

    def stats(self) -> dict[str, Any]:
        return merge_stats([a for a in (self.adapter, self.embedder) if a is not None])

    async def aclose(self) -> None:
        for adapter in dict.fromkeys(a for a in (self.adapter, self.embedder) if a):
            await adapter.aclose()
