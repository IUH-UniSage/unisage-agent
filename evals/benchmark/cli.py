"""`python -m evals.benchmark <run> [-a|-p|-c|--all] [--variant V]... [--repeats N] [--limit N]`
(`task benchmark:run env=eval -- hybrid-on -a`).

No mode flag = `--all`; no `--variant` = all three. The service is spawned when nothing
answers at CROSS_ENCODER_URL (`--no-serve` to refuse) and is always stopped in `finally`,
after which the report records whether nvidia-smi's used VRAM came back.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path
from typing import Any

from app.core.config import settings
from app.integrations.cross_encoder_client import CrossEncoderClient, CrossEncoderUnavailableError
from app.rag.retrieval.service import RetrievalService
from evals.benchmark.candidates import load_candidates, run_dir
from evals.benchmark.report import DEFAULT_OUTPUT_ROOT, run_timestamp, write_reports
from evals.benchmark.resource_sampler import gpu_name, vram_used_mb
from evals.benchmark.runner import (
    DEFAULT_CONC_REQUESTS,
    DEFAULT_CONCURRENCY_LEVELS,
    DEFAULT_LLM_REPEATS,
    DEFAULT_PERF_ITERATIONS,
    DEFAULT_WARMUP,
    AccuracyResult,
    ConcurrencyPoint,
    PerfResult,
    run_accuracy,
    run_concurrency,
    run_performance,
)
from evals.benchmark.service import ensure_service
from evals.benchmark.variants import CE_VARIANTS, LLM_VARIANTS, VARIANT_NAMES, build_variants

logger = logging.getLogger("evals.benchmark")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m evals.benchmark",
        description=(
            "So sánh LLM rerank với cross-encoder trên các ứng viên một run eval đã lấy về."
        ),
    )
    parser.add_argument("run", help="tên run trong runs/ (vd hybrid-on) hoặc đường dẫn thư mục run")
    parser.add_argument(
        "-a", "--accuracy", action="store_true", help="Bảng 1 — accuracy trên cùng bộ ứng viên"
    )
    parser.add_argument(
        "-p",
        "--performance",
        action="store_true",
        help="Bảng 2 — độ trễ bước rerank, concurrency 1",
    )
    parser.add_argument(
        "-c",
        "--concurrency",
        action="store_true",
        help="Bảng 3 — chịu tải đồng thời, 1 câu cố định",
    )
    parser.add_argument(
        "--all", action="store_true", help="cả 3 bảng (mặc định khi không có cờ chế độ)"
    )
    parser.add_argument(
        "--variant",
        action="append",
        dest="variants",
        default=[],
        choices=VARIANT_NAMES,
        help="chỉ chạy biến thể này (lặp lại được); mặc định cả 3",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=DEFAULT_LLM_REPEATS,
        help=f"số lượt -a cho biến thể LLM (mặc định {DEFAULT_LLM_REPEATS}); "
        "cross-encoder luôn 2 lượt để kiểm tra tất định",
    )
    parser.add_argument("--limit", type=int, default=None, help="chỉ N câu đầu dùng được")
    parser.add_argument(
        "--category",
        action="append",
        dest="categories",
        default=[],
        help="lọc category (normal, calculation)",
    )
    parser.add_argument(
        "--id", action="append", dest="ids", default=[], help="chỉ các id câu hỏi này"
    )
    parser.add_argument(
        "--keep",
        type=int,
        default=None,
        help="K chunk cross-encoder giữ "
        f"(mặc định CHAT_CROSS_ENCODER_KEEP={settings.CHAT_CROSS_ENCODER_KEEP})",
    )
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    parser.add_argument(
        "--iterations", type=int, default=DEFAULT_PERF_ITERATIONS, help="số vòng -p qua toàn bộ câu"
    )
    parser.add_argument(
        "--concurrency-levels", default=",".join(map(str, DEFAULT_CONCURRENCY_LEVELS))
    )
    parser.add_argument("--conc-requests", type=int, default=DEFAULT_CONC_REQUESTS)
    parser.add_argument(
        "--include-llm",
        action="store_true",
        help="-c cũng bắn biến thể LLM (tốn phí API); mặc định -c chỉ chạy cross_encoder",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help=f"thư mục báo cáo (mặc định {DEFAULT_OUTPUT_ROOT}/<timestamp>)",
    )
    serve = parser.add_mutually_exclusive_group()
    serve.add_argument(
        "--serve",
        dest="serve",
        action="store_true",
        default=True,
        help="tự chạy service nếu chưa có (mặc định)",
    )
    serve.add_argument(
        "--no-serve",
        dest="serve",
        action="store_false",
        help="không tự chạy service; lỗi nếu chưa có",
    )
    return parser


def _environment(args: argparse.Namespace, run: Path, keep: int) -> dict[str, Any]:
    return {
        "GPU / driver": gpu_name() or "không có nvidia-smi",
        "Model cross-encoder": "BAAI/bge-reranker-v2-m3 (FP16, max_length 1024) qua "
        + settings.CROSS_ENCODER_URL,
        "Run nguồn": str(run),
        "Collection": settings.QDRANT_COLLECTION,
        "K giữ (cross-encoder)": keep,
        "CHAT_CONTEXT_MAX_CHUNKS": settings.CHAT_CONTEXT_MAX_CHUNKS,
        "Biến thể": ", ".join(args.variants),
        "Chế độ": ", ".join(
            m
            for m, on in (("-a", args.accuracy), ("-p", args.performance), ("-c", args.concurrency))
            if on
        ),
    }


async def run_benchmark(args: argparse.Namespace) -> int:
    if not (args.accuracy or args.performance or args.concurrency) or args.all:
        args.accuracy = args.performance = args.concurrency = True
    args.variants = list(dict.fromkeys(args.variants)) or list(VARIANT_NAMES)
    keep = args.keep or settings.CHAT_CROSS_ENCODER_KEEP
    run = run_dir(args.run)
    out = args.out or DEFAULT_OUTPUT_ROOT / run_timestamp()
    levels = tuple(int(x) for x in args.concurrency_levels.split(",") if x.strip())

    candidates, stats = load_candidates(
        run,
        RetrievalService(),
        limit=args.limit,
        categories=set(args.categories) or None,
        ids=set(args.ids) or None,
    )
    if not candidates:
        logger.error("no usable question: %s", stats.describe())
        return 2

    client = CrossEncoderClient()
    vram: dict[str, Any] = {"trước khi benchmark": vram_used_mb()}
    handle = None
    accuracy: list[AccuracyResult] = []
    performance: list[PerfResult] = []
    concurrency: list[ConcurrencyPoint] = []
    try:
        if any(name in CE_VARIANTS for name in args.variants):
            handle = await ensure_service(client, url=settings.CROSS_ENCODER_URL, serve=args.serve)
            vram["khi service đã nạp model"] = vram_used_mb()
        variants = await build_variants(args.variants, keep=keep, client=client)
        if args.accuracy:
            accuracy = await run_accuracy(candidates, variants, llm_repeats=args.repeats)
        if args.performance:
            performance = await run_performance(
                candidates, variants, warmup=args.warmup, iterations=args.iterations
            )
        if args.concurrency:
            conc_variants = [v for v in variants if args.include_llm or v.name not in LLM_VARIANTS]
            if conc_variants:
                concurrency = await run_concurrency(
                    candidates,
                    conc_variants,
                    levels=levels,
                    requests=args.conc_requests,
                    warmup=args.warmup,
                )
            else:
                logger.warning("-c: no variant left (LLM variants need --include-llm)")
    except CrossEncoderUnavailableError as exc:
        logger.error("cross-encoder service: %s", exc)
        return 3
    finally:
        if handle is not None:
            handle.stop()
            vram["sau khi tắt service"] = handle.vram_after_stop_mb
            vram["đã nhả VRAM"] = (
                "service có sẵn, không tắt"
                if not handle.spawned
                else "có"
                if handle.released
                else "KHÔNG"
                if handle.released is False
                else "n/a"
            )
    environment = _environment(args, run, keep)
    if handle is not None:
        environment["Service"] = handle.note
    md_path, json_path = write_reports(
        out,
        environment=environment,
        candidate_stats=stats,
        accuracy=accuracy,
        performance=performance,
        concurrency=concurrency,
        vram=vram,
    )
    print(f"\nreport: {md_path}\njson:   {json_path}")
    print(md_path.read_text(encoding="utf-8"))
    return 0


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    # The node logs every kept/dropped chunk per call: thousands of lines in a benchmark.
    logging.getLogger("app.graph.nodes.cross_encoder_rerank").setLevel(logging.WARNING)
    logging.getLogger("app.graph.nodes.llm_rerank").setLevel(logging.WARNING)
    args = build_parser().parse_args(argv)
    try:
        return asyncio.run(run_benchmark(args))
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130
