"""What one score cost: calls, tokens, USD and which models and (masked) keys produced it.
`JudgeLLM`/`JudgeEmbeddings` book every call into the ledger of the score being computed."""

import contextvars
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from evals.judge.cost import PRICES

current: contextvars.ContextVar["Ledger | None"] = contextvars.ContextVar(
    "judge_ledger", default=None
)


@dataclass
class Ledger:
    fallback_model: str | None = None
    embedding_model: str | None = None
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    usd: float = 0.0
    models: Counter[str] = field(default_factory=Counter)
    keys: set[str] = field(default_factory=set)
    steps: list[dict[str, Any]] = field(default_factory=list)

    def book(self, model: str, key: str, tokens_in: int, tokens_out: int) -> None:
        # a model without a known price is booked at 0 (`cost.PRICES`)
        price_in, price_out = PRICES.get(model, (0.0, 0.0))
        self.calls += 1
        self.input_tokens += tokens_in
        self.output_tokens += tokens_out
        self.usd += (tokens_in * price_in + tokens_out * price_out) / 1_000_000
        self.models[model] += 1
        self.keys.add(key)

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


def step(name: str, output: Any) -> None:
    """Keep what the judge answered at one RAGAS step (a pydantic model or a dict)."""

    ledger = current.get()
    if ledger is not None:
        data = output.model_dump() if hasattr(output, "model_dump") else output
        ledger.steps.append({"step": name, "output": data})


def book(model: str, key: str, tokens_in: int, tokens_out: int = 0) -> None:
    ledger = current.get()
    if ledger is not None:
        ledger.book(model, key, tokens_in, tokens_out)
