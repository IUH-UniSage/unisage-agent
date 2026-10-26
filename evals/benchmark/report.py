"""`benchmark_results/<timestamp>/rerank_benchmark.md` + `.json` (both gitignored)."""

from __future__ import annotations

import json
import platform
from collections.abc import Sequence
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

from evals.benchmark.candidates import CandidateStats
from evals.benchmark.metrics import percentile
from evals.benchmark.runner import AccuracyResult, ConcurrencyPoint, PerfResult

DEFAULT_OUTPUT_ROOT = Path("benchmark_results")

ACCURACY_NOTE = (
    "recall@k: phần tài liệu kỳ vọng xuất hiện trong k chunk giữ đầu; NDCG@16 nhị phân theo "
    "document_id với lý tưởng là mọi chunk đúng trong bộ ứng viên xếp trước; MRR: 1/hạng chunk "
    "đúng đầu tiên; Loại hết: câu không giữ chunk nào; Lỗi: lượt biến thể không trả kết quả "
    "(service/model lỗi)."
)


def run_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def _f(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def _ms(value: float) -> str:
    return f"{value:.0f}"


def _na(value: Any, suffix: str = "") -> str:
    return "n/a" if value is None else f"{value}{suffix}"


def _table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> list[str]:
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    lines += ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
    return lines


def env_table(environment: dict[str, Any]) -> str:
    lines = ["## Môi trường", "", *_table(("Mục", "Giá trị"), list(environment.items()))]
    return "\n".join(lines) + "\n"


def accuracy_table(results: list[AccuracyResult]) -> str:
    if not results:
        return ""
    rows: list[list[Any]] = []
    for result in results:
        for category in [None, *result.categories()]:
            agg = result.aggregate(category)
            if agg.runs == 0:
                continue
            rows.append(
                [
                    result.variant,
                    category or "tất cả",
                    agg.questions,
                    agg.runs,
                    _f(agg.recall_at_8),
                    _f(agg.recall_at_5),
                    _f(agg.ndcg_at_16),
                    _f(agg.mrr),
                    _pct(agg.empty_rate),
                    f"{agg.avg_kept:.1f}",
                    agg.errors,
                    _ms(agg.p50_ms),
                    _ms(agg.p95_ms),
                ]
            )
    headers = (
        "Biến thể", "Nhóm", "Câu", "Lượt", "recall@8", "recall@5", "NDCG@16", "MRR",
        "Loại hết", "Giữ TB", "Lỗi", "p50 ms", "p95 ms",
    )  # fmt: skip
    stability_rows: list[list[Any]] = []
    for result in results:
        stability = result.stability()
        if stability is None:
            stability_rows.append([result.variant, result.repeats, "—", "—", "—"])
            continue
        stability_rows.append(
            [
                result.variant,
                stability.repeats,
                _f(stability.mean_jaccard),
                _pct(stability.identical_rate),
                _f(stability.recall_at_8_spread),
            ]
        )
    lines = [
        "## Bảng 1 — Accuracy (chọn top-K trên cùng bộ ứng viên)",
        "",
        *_table(headers, rows),
        "",
        ACCURACY_NOTE,
        "",
        "### Độ ổn định giữa các lượt",
        "",
        *_table(
            ("Biến thể", "Lượt", "Jaccard TB tập giữ", "Cùng thứ tự", "Lệch recall@8"),
            stability_rows,
        ),
    ]
    return "\n".join(lines) + "\n"


def performance_table(results: list[PerfResult]) -> str:
    if not results:
        return ""
    first = results[0]
    rows: list[list[Any]] = []
    for result in results:
        s = result.summary()
        r = result.resources
        rows.append(
            [
                result.variant,
                len(result.latencies_ms),
                _ms(s["p50_ms"]),
                _ms(s["p95_ms"]),
                _ms(s["avg_ms"]),
                _ms(s["max_ms"]),
                f"{s['throughput_rps']:.2f}",
                result.errors,
                _na(r.avg_cpu_pct),
                _na(r.peak_ram_gb),
                _na(r.peak_vram_mb),
                _na(r.avg_gpu_util_pct),
            ]
        )
    headers = (
        "Biến thể", "Phép đo", "p50 ms", "p95 ms", "TB ms", "Max ms", "req/s", "Lỗi",
        "CPU TB %", "RAM đỉnh GB", "VRAM đỉnh MiB", "GPU TB %",
    )  # fmt: skip
    lines = [
        f"## Bảng 2 — Performance (warm-up {first.warmup}, {first.iterations} vòng x "
        f"{first.questions} câu, concurrency 1)",
        "",
        *_table(headers, rows),
    ]
    return "\n".join(lines) + "\n"


def concurrency_table(points: list[ConcurrencyPoint]) -> str:
    if not points:
        return ""
    rows = [
        [
            point.variant,
            point.concurrency,
            _ms(percentile(point.latencies_ms, 50)),
            _ms(percentile(point.latencies_ms, 95)),
            f"{point.throughput_rps:.2f}",
            _pct(point.error_rate),
            _na(point.resources.peak_vram_mb),
            _na(point.resources.avg_gpu_util_pct),
        ]
        for point in points
    ]
    headers = (
        "Biến thể", "Đồng thời", "p50 ms", "p95 ms", "req/s", "Lỗi", "VRAM đỉnh MiB", "GPU TB %",
    )  # fmt: skip
    lines = [
        f"## Bảng 3 — Concurrency (1 câu cố định `{points[0].question_id}`, "
        f"{points[0].requests} request mỗi mức)",
        "",
        *_table(headers, rows),
    ]
    return "\n".join(lines) + "\n"


def _accuracy_json(results: list[AccuracyResult]) -> list[dict[str, Any]]:
    payload = []
    for result in results:
        stability = result.stability()
        payload.append(
            {
                "variant": result.variant,
                "repeats": result.repeats,
                "aggregate": asdict(result.aggregate()),
                "by_category": {c: asdict(result.aggregate(c)) for c in result.categories()},
                "stability": asdict(stability) if stability else None,
                "runs": [asdict(run) for run in result.runs],
            }
        )
    return payload


def _performance_json(results: list[PerfResult]) -> list[dict[str, Any]]:
    payload = []
    for result in results:
        fields = {
            k: v for k, v in asdict(result).items() if k not in ("latencies_ms", "service_ms")
        }
        payload.append(
            {
                **fields,
                "summary": result.summary(),
                "latencies_ms": [round(x, 1) for x in result.latencies_ms],
            }
        )
    return payload


def _concurrency_json(points: list[ConcurrencyPoint]) -> list[dict[str, Any]]:
    return [
        {
            **{k: v for k, v in asdict(point).items() if k != "latencies_ms"},
            "p50_ms": percentile(point.latencies_ms, 50),
            "p95_ms": percentile(point.latencies_ms, 95),
            "throughput_rps": point.throughput_rps,
        }
        for point in points
    ]


def write_reports(
    output_dir: Path,
    *,
    environment: dict[str, Any],
    candidate_stats: CandidateStats,
    accuracy: list[AccuracyResult],
    performance: list[PerfResult],
    concurrency: list[ConcurrencyPoint],
    vram: dict[str, Any],
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    md = [
        "# Benchmark rerank — LLM rerank vs cross-encoder",
        "",
        env_table({**environment, "Câu hỏi": candidate_stats.describe()}),
        accuracy_table(accuracy),
        performance_table(performance),
        concurrency_table(concurrency),
        "## VRAM trước / sau",
        "",
        *_table(("Mục", "MiB"), [(key, _na(value)) for key, value in vram.items()]),
        "",
    ]
    md_path = output_dir / "rerank_benchmark.md"
    md_path.write_text("\n".join(md), encoding="utf-8")
    payload = {
        "environment": environment,
        "platform": platform.platform(),
        "candidates": {
            "kept": candidate_stats.kept,
            "rows": candidate_stats.rows,
            "skipped": dict(candidate_stats.skipped),
            "chunks_missing": candidate_stats.chunks_missing,
        },
        "accuracy": _accuracy_json(accuracy),
        "performance": _performance_json(performance),
        "concurrency": _concurrency_json(concurrency),
        "vram": vram,
    }
    json_path = output_dir / "rerank_benchmark.json"
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    return md_path, json_path
