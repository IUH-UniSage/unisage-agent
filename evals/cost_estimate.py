"""Estimate the cost of ingesting the crawled PDFs and of running the evaluation.

Chunk counts are measured, not guessed: a stratified sample of manifest PDFs
goes through the production chunker (`app.rag.chunking.strategy.dispatch`,
`markdown_aware`, no LLM involved) and the chunks-per-character ratio is
extrapolated to the whole corpus. Token counts use tiktoken `o200k_base` as a
proxy for every provider (Gemini tokenizes Vietnamese differently; treat the
figures as +/- 30%).

Prices come from the same LiteLLM price table the project syncs into
`model_prices` (see changes/29-09-2026-Model-Pricing-Sync), per 1M tokens.

Usage:
    python -m evals.cost_estimate --dataset ../unisage-gateway/dataset --prices litellm.json
"""

import argparse
import asyncio
import json
import random
import statistics
from dataclasses import dataclass
from pathlib import Path

import tiktoken

from app.rag.chunking.strategy import dispatch
from app.schemas.ingestion import ChunkingStrategyName
from evals.crawl.download import is_probably_scanned, read_csv

AGENT_ROOT = Path(__file__).resolve().parent.parent
PROMPTS = AGENT_ROOT / "app/rag/prompting/prompt_templates"
ADVISORY_PROMPT_SNAPSHOT = AGENT_ROOT / "tests/fixtures/advisory_prompt_snapshot.txt"

ENCODING = tiktoken.get_encoding("o200k_base")

# Output sizes that can't be measured without calling a model: the budget the
# prompts ask for, rounded up.
ENRICH_OUTPUT_TOKENS = 220  # JSON: 1-2 sentence summary + 3 questions
SUMMARY_TOKENS = 70
QUESTIONS_TOKENS = 90
CLASSIFICATION_OUTPUT_TOKENS = 120
HYDE_OUTPUT_TOKENS = 300
ANSWER_OUTPUT_TOKENS = 450
QUERY_EMBED_TOKENS = 350  # question + HyDE document
EVAL_QUESTIONS = 300
QUESTIONGEN_CALLS = 110  # ~3 questions per call, plus retries/rejects
QUESTIONGEN_INPUT_TOKENS = 3500  # document excerpt + instructions
QUESTIONGEN_OUTPUT_TOKENS = 700
JUDGE_INPUT_TOKENS = 3200  # question + reference + answer + top chunks (truncated)
JUDGE_OUTPUT_TOKENS = 150


def tokens(text: str) -> int:
    return len(ENCODING.encode(text, disallowed_special=()))


@dataclass
class ChunkSample:
    files: int
    text_chars: int
    chunks: int
    chunk_tokens: list[int]

    @property
    def chunks_per_kchar(self) -> float:
        return 1000 * self.chunks / max(self.text_chars, 1)

    @property
    def mean_chunk_tokens(self) -> float:
        return statistics.fmean(self.chunk_tokens) if self.chunk_tokens else 0.0


def stratified_sample(rows: list[dict[str, str]], size: int, seed: int) -> list[dict[str, str]]:
    """Spread the sample across small, medium and large files."""

    ordered = sorted(rows, key=lambda row: int(row["text_chars"]))
    if len(ordered) <= size:
        return ordered
    rng = random.Random(seed)
    cut = len(ordered) // 3
    thirds = [ordered[:cut], ordered[cut : 2 * cut], ordered[2 * cut :]]
    picked: list[dict[str, str]] = []
    for third in thirds:
        picked.extend(rng.sample(third, min(len(third), size // 3)))
    return picked


async def measure_chunks(dataset: Path, rows: list[dict[str, str]]) -> ChunkSample:
    sample = ChunkSample(files=0, text_chars=0, chunks=0, chunk_tokens=[])
    for row in rows:
        content = (dataset / row["local_path"]).read_bytes()
        try:
            chunks = await dispatch(
                ChunkingStrategyName.MARKDOWN_AWARE,
                {},
                content,
                f"{row['file_id']}.pdf",
                document_id=row["file_id"],
            )
        except Exception as exc:  # a broken PDF must not stop the estimate
            print(f"  skip {row['file_id']}: {type(exc).__name__}")
            continue
        sample.files += 1
        sample.text_chars += int(row["text_chars"])
        sample.chunks += len(chunks)
        sample.chunk_tokens.extend(tokens(chunk.content) for chunk in chunks)
    return sample


def price(prices: dict[str, dict[str, float]], model: str) -> tuple[float, float]:
    """(input, output) USD per 1M tokens."""

    entry = prices.get(model) or prices.get(f"gemini/{model}")
    if entry is None:
        raise KeyError(f"{model} not in price table")
    return (
        (entry.get("input_cost_per_token") or 0.0) * 1e6,
        (entry.get("output_cost_per_token") or 0.0) * 1e6,
    )


@dataclass
class Usage:
    chat_in: float = 0.0
    chat_out: float = 0.0
    embed: float = 0.0

    def cost(self, chat: tuple[float, float], embed: tuple[float, float]) -> float:
        return (self.chat_in * chat[0] + self.chat_out * chat[1] + self.embed * embed[0]) / 1e6


def ingest_usage(total_chunks: float, mean_chunk_tokens: float) -> Usage:
    enrich_prompt = tokens((PROMPTS / "agents/multi_representation_enricher.yaml").read_text())
    return Usage(
        chat_in=total_chunks * (enrich_prompt + mean_chunk_tokens),
        chat_out=total_chunks * ENRICH_OUTPUT_TOKENS,
        embed=total_chunks * (mean_chunk_tokens + SUMMARY_TOKENS + QUESTIONS_TOKENS),
    )


def eval_run_usage(mean_chunk_tokens: float, retrieved_chunks: int) -> Usage:
    classification = tokens((PROMPTS / "agents/message_classification.yaml").read_text())
    hyde = tokens((PROMPTS / "agents/hyde_generator.yaml").read_text())
    advisory = tokens(ADVISORY_PROMPT_SNAPSHOT.read_text())
    per_question_in = (
        classification + 60 + hyde + 60 + advisory + retrieved_chunks * mean_chunk_tokens
    )
    per_question_out = CLASSIFICATION_OUTPUT_TOKENS + HYDE_OUTPUT_TOKENS + ANSWER_OUTPUT_TOKENS
    return Usage(
        chat_in=EVAL_QUESTIONS * per_question_in,
        chat_out=EVAL_QUESTIONS * per_question_out,
        embed=EVAL_QUESTIONS * QUERY_EMBED_TOKENS,
    )


def judge_usage() -> Usage:
    return Usage(
        chat_in=EVAL_QUESTIONS * JUDGE_INPUT_TOKENS, chat_out=EVAL_QUESTIONS * JUDGE_OUTPUT_TOKENS
    )


def questiongen_usage() -> Usage:
    return Usage(
        chat_in=QUESTIONGEN_CALLS * QUESTIONGEN_INPUT_TOKENS,
        chat_out=QUESTIONGEN_CALLS * QUESTIONGEN_OUTPUT_TOKENS,
    )


async def run(args: argparse.Namespace) -> dict[str, object]:
    manifest = read_csv(args.dataset / "manifest.csv")
    text_rows = [
        row
        for row in manifest
        if int(row["pages"] or 0) > 0
        and not is_probably_scanned(int(row["pages"]), int(row["text_chars"]))
    ]
    sample_rows = stratified_sample(text_rows, args.sample, args.seed)
    print(f"measuring chunking on {len(sample_rows)} of {len(text_rows)} text PDFs...")
    sample = await measure_chunks(args.dataset, sample_rows)

    corpus_chars = sum(int(row["text_chars"]) for row in text_rows)
    total_chunks = corpus_chars / 1000 * sample.chunks_per_kchar
    prices = json.loads(args.prices.read_text())
    ingest = ingest_usage(total_chunks, sample.mean_chunk_tokens)
    run_usage = eval_run_usage(sample.mean_chunk_tokens, args.retrieved_chunks)
    judge = judge_usage()
    questiongen = questiongen_usage()

    scenarios = []
    for chat_model, embed_model in args.scenario:
        chat_price, embed_price = price(prices, chat_model), price(prices, embed_model)
        scenarios.append(
            {
                "chat_model": chat_model,
                "embed_model": embed_model,
                "chat_price_per_m": chat_price,
                "embed_price_per_m": embed_price[0],
                "ingest_usd": ingest.cost(chat_price, embed_price),
                "ingest_enrich_usd": Usage(ingest.chat_in, ingest.chat_out).cost(
                    chat_price, (0, 0)
                ),
                "ingest_embed_usd": Usage(embed=ingest.embed).cost((0, 0), embed_price),
                "questiongen_usd": questiongen.cost(chat_price, embed_price),
                "eval_run_graph_usd": run_usage.cost(chat_price, embed_price),
                "eval_run_judge_usd": judge.cost(chat_price, embed_price),
            }
        )

    scanned = [
        row
        for row in manifest
        if is_probably_scanned(int(row["pages"] or 0), int(row["text_chars"] or 0))
    ]
    return {
        "manifest_files": len(manifest),
        "text_files": len(text_rows),
        "scanned_files": len(scanned),
        "unreadable_files": sum(1 for row in manifest if int(row["pages"] or 0) == 0),
        "total_pages": sum(int(row["pages"] or 0) for row in manifest),
        "text_pages": sum(int(row["pages"]) for row in text_rows),
        "corpus_text_chars": corpus_chars,
        "sample_files": sample.files,
        "sample_chunks": sample.chunks,
        "chunks_per_kchar": sample.chunks_per_kchar,
        "mean_chunk_tokens": sample.mean_chunk_tokens,
        "estimated_total_chunks": total_chunks,
        "ingest_tokens": ingest.__dict__,
        "eval_run_tokens": run_usage.__dict__,
        "judge_tokens": judge.__dict__,
        "questiongen_tokens": questiongen.__dict__,
        "prompt_tokens": {
            "advisory_snapshot": tokens(ADVISORY_PROMPT_SNAPSHOT.read_text()),
            "classification": tokens((PROMPTS / "agents/message_classification.yaml").read_text()),
            "hyde": tokens((PROMPTS / "agents/hyde_generator.yaml").read_text()),
        },
        "scenarios": scenarios,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--prices", type=Path, required=True, help="LiteLLM price JSON")
    parser.add_argument("--sample", type=int, default=45)
    parser.add_argument("--seed", type=int, default=30092026)
    parser.add_argument("--retrieved-chunks", type=int, default=8)
    parser.add_argument(
        "--scenario",
        nargs=2,
        action="append",
        metavar=("CHAT_MODEL", "EMBED_MODEL"),
        required=True,
    )
    parser.add_argument("--out", type=Path, help="write the result JSON here")
    args = parser.parse_args()
    result = asyncio.run(run(args))
    text = json.dumps(result, indent=2, ensure_ascii=False)
    if args.out:
        args.out.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
