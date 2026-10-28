"""Cost estimate at list price and the budget guard: nothing is called past `--max-cost`."""

from collections.abc import Sequence
from typing import Any

from evals.judge.tasks import CALLS, Task

# USD per 1M tokens (input, output) - LiteLLM prices of 10-10-2026, thinking billed as output.
# A model missing here is estimated at the highest known price and booked at 0 (unknown).
PRICES = {
    "gemini-3.1-flash-lite-preview": (0.25, 1.50),
    "gemini-3.5-flash-lite": (0.30, 2.50),
}
# Estimate: a RAGAS prompt adds ~600 tokens of instructions and examples; ~3 characters of
# Vietnamese per token; ~250 tokens of JSON out; 20% for retries.
PROMPT_OVERHEAD_TOKENS = 600
CHARS_PER_TOKEN = 3.0
OUTPUT_TOKENS_PER_CALL = 250
RETRY_MARGIN = 1.2


def _tokens(*texts: Any) -> int:
    chars = 0
    for text in texts:
        if isinstance(text, list):
            chars += sum(len(str(item)) for item in text)
        elif text:
            chars += len(str(text))
    return int(chars / CHARS_PER_TOKEN)


def price_of(model: str) -> tuple[tuple[float, float], bool]:
    """(input, output) USD per 1M tokens and whether the model's price is known; an unknown
    model is estimated at the highest known price, so the budget guard errs on the safe side."""

    if model in PRICES:
        return PRICES[model], True
    return (max(p[0] for p in PRICES.values()), max(p[1] for p in PRICES.values())), False


def estimate_task(task: Task, prices: tuple[float, float]) -> tuple[int, int, int, float]:
    """(calls, input tokens, output tokens, USD) one task is expected to cost."""

    calls = CALLS[task.metric]
    if not calls:
        return 0, 0, 0, 0.0
    inputs = task.inputs
    if task.metric in ("faithfulness", "web_faithfulness"):
        # statements from the answer, then each statement against the whole context
        text_in = _tokens(inputs["user_input"], inputs["response"]) * 2 + _tokens(
            inputs["retrieved_contexts"]
        )
    elif task.metric == "answer_coverage":
        text_in = (_tokens(inputs["response"]) + _tokens(inputs["reference"])) * 2
    elif task.metric == "context_recall":
        text_in = _tokens(inputs["user_input"], inputs["retrieved_contexts"], inputs["reference"])
    else:
        text_in = _tokens(inputs.get("user_input"), inputs.get("response"))
    tokens_in = int((text_in + calls * PROMPT_OVERHEAD_TOKENS) * RETRY_MARGIN)
    tokens_out = int(calls * OUTPUT_TOKENS_PER_CALL * RETRY_MARGIN)
    usd = (tokens_in * prices[0] + tokens_out * prices[1]) / 1_000_000
    return calls, tokens_in, tokens_out, usd


def estimate(tasks: Sequence[Task], model: str) -> dict[str, Any]:
    prices, _known = price_of(model)
    calls = tokens_in = tokens_out = 0
    usd = 0.0
    for task in tasks:
        c, i, o, u = estimate_task(task, prices)
        calls, tokens_in, tokens_out, usd = calls + c, tokens_in + i, tokens_out + o, usd + u
    return {
        "tasks": len(tasks),
        "calls": calls,
        "input_tokens": tokens_in,
        "output_tokens": tokens_out,
        "usd": round(usd, 4),
        "model": model,
        "price_known": _known,
    }


class BudgetExceeded(Exception):
    pass


def check_budget(estimated: dict[str, Any], max_cost: float) -> None:
    if estimated["usd"] > max_cost:
        raise BudgetExceeded(
            f"estimated {estimated['usd']:.2f} USD for {estimated['calls']} calls exceeds "
            f"--max-cost {max_cost:.2f} USD - nothing was called"
        )
