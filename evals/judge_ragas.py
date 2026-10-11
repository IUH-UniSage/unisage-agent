"""RAGAS metrics on Gemini, for `evals.judge` (Task 11). Needs the judge environment
(`evals/requirements-judge.txt`): RAGAS 0.4 and `google-genai`.

The metrics are RAGAS' own (`ragas.metrics.collections`: Faithfulness, FactualCorrectness,
ContextRecall, AnswerRelevancy; ID-based context precision/recall; a `DiscreteMetric` for
refusal, the 0.4 form of AspectCritic). Only the model they talk to is ours: `GeminiJudgeLLM`
sends every structured-output request through the key pool of `evals.gemini_pool`, switches to
the fallback model when the main one stays overloaded, and books tokens, cost, model and key
per score.

The example parts of the RAGAS prompts are translated to Vietnamese once
(`--adapt-prompts`, RAGAS `BasePrompt.adapt`) and cached in `evals/ragas_prompts/`; later runs
load the cache, so the judge reads Vietnamese examples next to Vietnamese answers.
"""

import asyncio
import contextvars
import json
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from google import genai
from google.genai import types
from pydantic import BaseModel
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

from evals.gemini_pool import (
    CallFailed,
    GeminiCaller,
    KeyPool,
    KeyState,
    ModelUnavailable,
    QuotaExhausted,
)
from evals.judge import PRICES, Score, Task

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
_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$")

_ledger: contextvars.ContextVar["Ledger | None"] = contextvars.ContextVar(
    "judge_ledger", default=None
)


@dataclass
class Ledger:
    """What one score cost: calls, tokens, USD, and which models/keys produced it."""

    fallback_model: str | None = None
    embedding_model: str | None = None
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    usd: float = 0.0
    models: Counter[str] = field(default_factory=Counter)
    keys: set[str] = field(default_factory=set)

    def book(self, model: str, key: KeyState, tokens_in: int, tokens_out: int) -> None:
        price_in, price_out = PRICES.get(model, (0.0, 0.0))
        self.calls += 1
        self.input_tokens += tokens_in
        self.output_tokens += tokens_out
        self.usd += (tokens_in * price_in + tokens_out * price_out) / 1_000_000
        self.models[model] += 1
        self.keys.add(key.masked)

    def as_usage(self) -> dict[str, Any]:
        # the judge model is the LLM that scored; embeddings (answer_relevancy) are not one
        llm_models = Counter({m: n for m, n in self.models.items() if m != self.embedding_model})
        return {
            "model": llm_models.most_common(1)[0][0] if llm_models else None,
            "models": dict(self.models),
            "keys": sorted(self.keys),
            "fallback": bool(self.fallback_model and self.fallback_model in self.models),
            "calls": self.calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "usd": round(self.usd, 6),
        }


def _book(model: str, key: KeyState, response: Any) -> None:
    ledger = _ledger.get()
    if ledger is None:
        return
    usage = getattr(response, "usage_metadata", None)
    tokens_in = int(getattr(usage, "prompt_token_count", 0) or 0)
    # thinking tokens are billed as output
    tokens_out = int(getattr(usage, "candidates_token_count", 0) or 0) + int(
        getattr(usage, "thoughts_token_count", 0) or 0
    )
    ledger.book(model, key, tokens_in, tokens_out)


def _thinking(model: str, level: str) -> types.ThinkingConfig | None:
    if not level:
        return None
    if "gemini-3" in model:
        return types.ThinkingConfig(thinking_level=level)
    # 2.x models take a token budget; 0 turns thinking off where the model allows it
    return types.ThinkingConfig(thinking_budget=0)


class GeminiJudgeLLM(InstructorBaseRagasLLM):
    """Structured output from Gemini for the RAGAS metrics, through the key pool."""

    def __init__(
        self, caller: GeminiCaller, model: str, fallback: str | None, thinking: str
    ) -> None:
        self.caller = caller
        self.model = model
        self.fallback = fallback
        self.thinking = thinking

    def _config(self, model: str, response_model: type[BaseModel]) -> types.GenerateContentConfig:
        return types.GenerateContentConfig(
            temperature=0.0,
            response_mime_type="application/json",
            response_json_schema=response_model.model_json_schema(),
            thinking_config=_thinking(model, self.thinking),
        )

    async def agenerate(self, prompt: str, response_model: type[Any]) -> Any:
        models = [self.model, *([self.fallback] if self.fallback else [])]
        last: Exception | None = None
        for model in models:
            config = self._config(model, response_model)
            try:
                response, key = await self.caller.call(
                    model,
                    lambda client, m=model, c=config: client.aio.models.generate_content(
                        model=m, contents=prompt, config=c
                    ),
                )
            except ModelUnavailable as error:
                last = error
                continue
            _book(model, key, response)
            text = _FENCE.sub("", (response.text or "").strip())
            return response_model.model_validate_json(text)
        raise last or ModelUnavailable(self.model)

    def generate(self, prompt: str, response_model: type[Any]) -> Any:
        raise NotImplementedError("the judge is async: use agenerate")


class GeminiEmbeddings(BaseRagasEmbedding):
    """Gemini embeddings (for AnswerRelevancy) through the key pool."""

    def __init__(self, caller: GeminiCaller, model: str) -> None:
        super().__init__(cache=None)
        self.caller = caller
        self.model = model

    def embed_text(self, text: str, **kwargs: Any) -> list[float]:
        raise NotImplementedError("the judge is async: use aembed_text")

    async def aembed_text(self, text: str, **kwargs: Any) -> list[float]:
        return (await self.aembed_texts([text]))[0]

    async def aembed_texts(self, texts: list[str], **kwargs: Any) -> list[list[float]]:
        response, key = await self.caller.call(
            self.model,
            lambda client: client.aio.models.embed_content(model=self.model, contents=texts),
        )
        _book(self.model, key, response)
        return [list(embedding.values) for embedding in response.embeddings]


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


async def localize(metrics: dict[str, Any], llm: GeminiJudgeLLM, folder: Path, adapt: bool) -> int:
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
    llm: GeminiJudgeLLM
    metrics: dict[str, Any]
    caller: GeminiCaller
    fallback_model: str | None = None
    embedding_model: str | None = None
    ready: asyncio.Event = field(default_factory=asyncio.Event)
    setup: Any = None

    async def score(self, task: Task) -> Score:
        if self.setup is not None and not self.ready.is_set():
            await self.setup(self)
        ledger = Ledger(fallback_model=self.fallback_model, embedding_model=self.embedding_model)
        token = _ledger.set(ledger)
        try:
            value, reason, extra = await self._run(task)
        except QuotaExhausted:
            raise
        except (ModelUnavailable, CallFailed) as error:
            value, reason, extra = math.nan, str(error)[:300], {}
        except Exception as error:  # a metric that cannot score this row: NaN + why
            value, reason, extra = math.nan, f"{type(error).__name__}: {str(error)[:300]}", {}
        finally:
            _ledger.reset(token)
        return Score(value=value, reason=reason, usage=ledger.as_usage(), extra=extra)

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
        return self.caller.stats()


@dataclass
class RagasScorerFactory:
    """Builds the scorer `evals.judge.JudgeRun` asks for (main model only, when asked)."""

    keys: list[str]
    model: str
    fallback: str | None
    embedding_model: str
    thinking: str
    rpm: int
    prompts_dir: Path
    adapt_prompts: bool
    metrics: list[str]
    _callers: list[GeminiCaller] = field(default_factory=list)

    def __call__(self, primary_only: bool) -> RagasScorer:
        models = {self.model, self.embedding_model, *([self.fallback] if self.fallback else [])}
        caller = GeminiCaller(
            pools={model: KeyPool(self.keys, rpm=self.rpm) for model in models},
            client_for=_client_cache(),
        )
        self._callers.append(caller)
        llm = GeminiJudgeLLM(
            caller, self.model, None if primary_only else self.fallback, self.thinking
        )
        embeddings = GeminiEmbeddings(caller, self.embedding_model)
        metrics: dict[str, Any] = {
            "faithfulness": Faithfulness(llm=llm),
            "answer_coverage": AnswerCorrectness(llm=llm, mode="f1"),
            "context_recall": ContextRecall(llm=llm),
            "answer_relevancy": AnswerRelevancy(llm=llm, embeddings=embeddings, strictness=1),
            "id_context_precision": IDBasedContextPrecision(),
            "id_context_recall": IDBasedContextRecall(),
            "refusal": DiscreteMetric(
                name="refusal", prompt=REFUSAL_PROMPT, allowed_values=["refused", "answered"]
            ),
        }
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
            caller=caller,
            fallback_model=self.fallback,
            embedding_model=self.embedding_model,
            setup=setup,
        )

    def stats(self) -> dict[str, Any]:
        merged: dict[str, Any] = {}
        for caller in self._callers:
            for model, keys in caller.stats().items():
                bucket = merged.setdefault(model, {})
                for key, counts in keys.items():
                    total = Counter(bucket.get(key, {}))
                    total.update(counts)
                    bucket[key] = dict(total)
        return merged


def _client_cache() -> Any:
    clients: dict[str, genai.Client] = {}

    def client_for(key: str) -> genai.Client:
        if key not in clients:
            clients[key] = genai.Client(api_key=key)
        return clients[key]

    return client_for
