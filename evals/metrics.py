"""Deterministic scoring of one recorded turn (Task 10, `scoring-rules.md`).

Pure functions over plain dicts - the `results.jsonl` rows written by `evals.run` - so they
are tested without a graph or a network. LLM-judged criteria (RAGAS, refusal critics) are
added later by `evals.judge`; here a row gets everything that needs no model.
"""

import json
import math
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from typing import Any

WILDCARD = "*"
RECALL_K = 5
# A refusal/ticket answer, recognised without an LLM (the judge refines it later).
REFUSAL_HINTS = (
    "không tìm thấy",
    "không có thông tin",
    "chưa có thông tin",
    "không đủ thông tin",
    "không thể trả lời",
    "không có quyền",
    "tạo ticket",
    "gửi yêu cầu hỗ trợ",
    "liên hệ",
)


# --- outcome (scoring-rules §1.2) ------------------------------------------------------------


def outcome_of(row: dict[str, Any]) -> list[str]:
    """What the turn did, from the recorded nodes and output flags. A multi-task turn can
    have several (e.g. calculation + rag)."""

    if row.get("error"):
        return ["error"]
    nodes = set(row.get("nodes") or [])
    retrieved = "08_RetrievalFilteringNode" in nodes
    found: list[str] = []
    if not retrieved and (
        ("01_GreetingDetectionNode" in nodes and len(nodes) == 1)
        or "04_IntentRouting_SocialChat" in nodes
    ):
        found.append("social")
    if not retrieved and "05_OffTopicRejectNode" in nodes:
        found.append("off_topic")
    if "07_CalculationNode" in nodes:
        found.append("calculation")
    if row.get("used_ticket_fallback"):
        found.append("ticket")
    elif row.get("used_web_search") and any(
        c.get("sourceType") == "WEB" for c in row.get("citations") or []
    ):
        found.append("web")
    elif retrieved and row.get("context"):
        found.append("rag")
    return found or ["none"]


def intents_of(row: dict[str, Any]) -> list[str]:
    """Intents the classifier chose (from the `03_MessageClassificationNode` dump)."""

    for text in (row.get("prompts") or {}).get("03_MessageClassificationNode", []):
        try:
            return [task["intent"] for task in json.loads(text)["tasks"]]
        except (ValueError, KeyError, TypeError):
            continue
    nodes = row.get("nodes") or []
    if nodes == ["01_GreetingDetectionNode"]:
        return ["social_chat"]
    return []


# --- permissions (scoring-rules §1.5, qdrant_store.build_access_filter) ----------------------


def can_see(meta: dict[str, Any], access: Sequence[dict[str, Any]]) -> bool:
    """Same rule as the Qdrant pre-filter: public, or a grant on the department (or `*`) at a
    level >= the document's."""

    if meta.get("is_public"):
        return True
    level = meta.get("access_level")
    if level is None:
        return False
    return any(
        grant["department_id"] in (meta.get("department"), WILDCARD)
        and int(grant["access_level"]) >= int(level)
        for grant in access
    )


def leaked_documents(row: dict[str, Any]) -> list[str]:
    """Private documents the asker may not see that still came back: in any retrieved chunk
    (the Qdrant pre-filter should never return one), the context after rerank, or a cited
    `documentId`."""

    access = row.get("access") or []
    leaked = {chunk["document_id"] for chunk in _all_chunks(row) if not can_see(chunk, access)}
    known = {chunk["document_id"]: chunk for chunk in _all_chunks(row)}
    for citation in row.get("citations") or []:
        document_id = citation.get("documentId")
        if document_id in known and not can_see(known[document_id], access):
            leaked.add(document_id)
    return sorted(d for d in leaked if d)


def _all_chunks(row: dict[str, Any]) -> Iterable[dict[str, Any]]:
    yield from row.get("context") or []
    for retrieval in row.get("retrievals") or []:
        yield from retrieval["chunks"]


# --- retrieval and citations ----------------------------------------------------------------


def recall_at_k(row: dict[str, Any], k: int = RECALL_K) -> bool | None:
    """At least one expected document among the first `k` context chunks (rerank order).
    None when the row has no expected document."""

    expected = set(row.get("expected_document_ids") or [])
    if not expected:
        return None
    top = [chunk["document_id"] for chunk in (row.get("context") or [])[:k]]
    return bool(expected & set(top))


def recall_all_at_k(row: dict[str, Any], k: int = RECALL_K) -> bool | None:
    expected = set(row.get("expected_document_ids") or [])
    if len(expected) < 2:
        return None
    top = {chunk["document_id"] for chunk in (row.get("context") or [])[:k]}
    return expected <= top


def citation_accuracy(row: dict[str, Any]) -> float | None:
    """Share of document citations pointing at an expected document (#7)."""

    expected = set(row.get("expected_document_ids") or [])
    cited = [c["documentId"] for c in row.get("citations") or [] if c.get("documentId")]
    if not expected or not cited:
        return None
    return sum(d in expected for d in cited) / len(cited)


# --- answers --------------------------------------------------------------------------------

_NUMBER = re.compile(r"\d+(?:[.,\s]\d{3})*(?:[.,]\d+)?")


def numbers_in(text: str) -> set[float]:
    """Every number in `text`, accepting 1.200.000 / 1,200,000 / 1 200 000 and 7,5 / 7.5."""

    found: set[float] = set()
    for raw in _NUMBER.findall(text):
        token = raw.replace(" ", "")
        for candidate in _readings(token):
            found.add(round(candidate, 6))
    return found


def _readings(token: str) -> list[float]:
    readings: list[float] = []
    if "," in token and "." in token:
        decimal = "," if token.rfind(",") > token.rfind(".") else "."
        thousands = "." if decimal == "," else ","
        token = token.replace(thousands, "").replace(decimal, ".")
        return [float(token)]
    for sep in (",", "."):
        if sep in token:
            parts = token.split(sep)
            if all(len(p) == 3 for p in parts[1:]):
                readings.append(float(token.replace(sep, "")))  # thousands
            if len(parts) == 2:
                readings.append(float(token.replace(sep, ".")))  # decimal
            return readings
    return [float(token)]


def numbers_match(expected: Sequence[float], text: str) -> bool:
    found = numbers_in(text)
    return all(round(float(n), 6) in found for n in expected)


def looks_like_refusal(text: str) -> bool:
    lowered = text.lower()
    return any(hint in lowered for hint in REFUSAL_HINTS)


# --- per-row verdict (scoring-rules §2, deterministic part) ---------------------------------


def score_row(row: dict[str, Any]) -> dict[str, Any]:
    """Deterministic checks for one row. `pass` is None when the verdict needs the LLM judge
    (e.g. a normal answer: outcome and recall are known, correctness is not yet)."""

    outcome = outcome_of(row)
    category = row["category"]
    visible = row.get("expect_visible")
    answer = row.get("response") or ""
    checks: dict[str, Any] = {
        "outcome": outcome,
        "intents": intents_of(row),
        "recall_at_5": recall_at_k(row),
        "recall_all_at_5": recall_all_at_k(row),
        "citation_accuracy": citation_accuracy(row),
        "leaked": leaked_documents(row),
        "refusal_hint": looks_like_refusal(answer),
        "asked_back": bool(row.get("asked_back")),
    }
    passed: bool | None = None
    if "error" in outcome:
        passed = None
    elif category in ("off_topic", "social"):
        passed = outcome == [category] and not row.get("retrievals")
    elif category == "web_search":
        passed = None if row.get("web") != "on" else "web" in outcome
    elif category == "unanswerable" or (category == "access" and visible is False):
        refused = "ticket" in outcome or checks["refusal_hint"]
        passed = not checks["leaked"] and refused
    elif category == "calculation" or row.get("expected_intent") == "academic_calculation":
        numbers = row.get("expected_numbers")
        checks["numbers_match"] = numbers_match(numbers, answer) if numbers else None
        if "calculation" not in outcome or checks["asked_back"]:
            passed = False
        elif checks["numbers_match"] is not None:
            passed = checks["numbers_match"] and not checks["leaked"]
    else:  # normal, access visible
        if "rag" not in outcome or checks["recall_at_5"] is False or checks["leaked"]:
            passed = False
    checks["pass"] = passed
    return checks


# --- aggregation ----------------------------------------------------------------------------


def wilson(successes: int, n: int, z: float = 1.96) -> tuple[float, float, float] | None:
    """Rate and Wilson 95% interval."""

    if n == 0:
        return None
    p = successes / n
    denominator = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denominator
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return p, max(0.0, centre - margin), min(1.0, centre + margin)


def summarize(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Counts per (variant, category) and overall rates for the report."""

    table: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    leaks: list[dict[str, Any]] = []
    latencies: list[float] = []
    ttfts: list[float] = []
    for row in rows:
        checks = row["checks"]
        bucket = table[(row["variant"], row["category"])]
        bucket["rows"] += 1
        if "error" in checks["outcome"]:
            bucket["error"] += 1
            continue
        if checks["pass"] is True:
            bucket["pass"] += 1
        elif checks["pass"] is False:
            bucket["fail"] += 1
        else:
            bucket["pending"] += 1
        if checks["recall_at_5"] is not None:
            bucket["recall_n"] += 1
            bucket["recall_hit"] += checks["recall_at_5"]
        if row.get("expect_visible") is False:
            bucket["leak_n"] += 1
            bucket["leak"] += bool(checks["leaked"])
        if checks["leaked"]:
            leaks.append({"id": row["id"], "persona": row.get("persona"), "docs": checks["leaked"]})
        latencies.append(row["total_ms"])
        if row.get("ttft_ms") is not None:
            ttfts.append(row["ttft_ms"])
    return {
        "table": {f"{v}/{c}": dict(counts) for (v, c), counts in sorted(table.items())},
        "leaks": leaks,
        "latency_ms": _percentiles(latencies),
        "ttft_ms": _percentiles(ttfts),
    }


def _percentiles(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    ordered = sorted(values)
    return {
        "p50": ordered[len(ordered) // 2],
        "p95": ordered[min(len(ordered) - 1, math.ceil(0.95 * len(ordered)) - 1)],
        "n": len(ordered),
    }
