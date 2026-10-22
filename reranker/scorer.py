"""The cross-encoder itself: `BAAI/bge-reranker-v2-m3` through transformers.

FlagEmbedding's `FlagReranker.compute_score` is a thin wrapper around
`AutoModelForSequenceClassification` (tokenize the (query, passage) pairs, take logit 0);
calling transformers directly gives the same scores without FlagEmbedding's training
dependencies (datasets, peft, ir-datasets) and lets us report how many passages the
`max_length` cut - the spec's open question about chunks longer than 1024 tokens.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from typing import Protocol

logger = logging.getLogger("reranker")

DEFAULT_MODEL = "BAAI/bge-reranker-v2-m3"
DEFAULT_MAX_LENGTH = 1024
DEFAULT_BATCH_SIZE = 16


@dataclass(frozen=True)
class ScoreResult:
    scores: list[float]
    truncated: int  # passages whose (query + passage) did not fit max_length
    took_ms: int


class Scorer(Protocol):
    model_name: str
    device: str

    def score(self, query: str, passages: list[str]) -> ScoreResult: ...

    def close(self) -> None: ...


class CrossEncoderScorer:
    """Loads the model once; `score` is synchronous and holds the GPU for one call."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        *,
        max_length: int = DEFAULT_MAX_LENGTH,
        batch_size: int = DEFAULT_BATCH_SIZE,
        device: str | None = None,
        fp16: bool = True,
    ) -> None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._torch = torch
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = max(1, batch_size)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        use_fp16 = fp16 and self.device.startswith("cuda")
        started = time.perf_counter()
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModelForSequenceClassification.from_pretrained(
            model_name, torch_dtype=torch.float16 if use_fp16 else torch.float32
        )
        self._model.to(self.device)
        self._model.eval()
        logger.info(
            "reranker loaded %s on %s (%s) in %.1fs",
            model_name,
            self.device,
            "fp16" if use_fp16 else "fp32",
            time.perf_counter() - started,
        )

    def score(self, query: str, passages: list[str]) -> ScoreResult:
        torch = self._torch
        started = time.perf_counter()
        scores: list[float] = []
        truncated = 0
        with torch.inference_mode():
            for start in range(0, len(passages), self.batch_size):
                batch = passages[start : start + self.batch_size]
                # Count what max_length drops: tokenize once untruncated for the lengths.
                lengths = self._tokenizer(
                    [query] * len(batch), batch, truncation=False, padding=False
                )["input_ids"]
                truncated += sum(1 for ids in lengths if len(ids) > self.max_length)
                encoded = self._tokenizer(
                    [query] * len(batch),
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                ).to(self.device)
                logits = self._model(**encoded, return_dict=True).logits.view(-1).float()
                scores.extend(logits.cpu().tolist())
        return ScoreResult(
            scores=scores, truncated=truncated, took_ms=int((time.perf_counter() - started) * 1000)
        )

    def vram_used_mb(self) -> int | None:
        if not self.device.startswith("cuda"):
            return None
        return int(self._torch.cuda.memory_allocated() / (1024 * 1024))

    def close(self) -> None:
        """Drop the model and give the VRAM back (the service's shutdown hook)."""

        torch = self._torch
        del self._model
        del self._tokenizer
        import gc

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        logger.info("reranker unloaded, VRAM released")


def scorer_from_env() -> CrossEncoderScorer:
    return CrossEncoderScorer(
        os.environ.get("RERANKER_MODEL", DEFAULT_MODEL),
        max_length=int(os.environ.get("RERANKER_MAX_LENGTH", DEFAULT_MAX_LENGTH)),
        batch_size=int(os.environ.get("RERANKER_BATCH_SIZE", DEFAULT_BATCH_SIZE)),
        device=os.environ.get("RERANKER_DEVICE") or None,
        fp16=os.environ.get("RERANKER_FP16", "true").lower() != "false",
    )
