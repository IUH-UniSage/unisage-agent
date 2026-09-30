"""Plan the evaluation question set: which document backs which question slot.

Marks the ingest set (`selected=true`: `quality=ok`, at most `--max-pages`
pages - option B in cost-estimate.md) and, from it, draws one source document
per slot so questions spread across departments and access levels:

- `normal`       academic_advisory, one per document, split across departments
                 in proportion to their document count;
- `calculation`  academic_calculation, from documents that mention fees,
                 credits or grade averages;
- `access`       academic_advisory on a *private* document; build.py later
                 expands each into four askers (see PERSONAS there);
- `unanswerable` academic_advisory with no source, only a department context;
- `off_topic` / `social` no source.

For every sourced slot the document's text excerpt is written to
`<work-dir>/excerpts/<file_id>.txt` (stored relative to the work dir); the
question author works only from that excerpt so the reference answer is grounded
in text the pipeline also sees.

Usage:
    python -m evals.questions.plan --dataset ../unisage-gateway/dataset --work-dir /tmp/qwork
"""

import argparse
import json
import random
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import pymupdf

from evals.crawl.download import DownloadState

SLOT_COUNTS = {
    "normal": 150,
    "calculation": 30,
    "access": 13,
    "unanswerable": 40,
    "off_topic": 18,
    "social": 12,
}
EXPECTED_INTENT = {
    "normal": "academic_advisory",
    "calculation": "academic_calculation",
    "access": "academic_advisory",
    "unanswerable": "academic_advisory",
    "off_topic": "off_topic",
    "social": "social_chat",
}
MIN_TEXT_CHARS = 800
EXCERPT_CHARS = 5000
EXCERPT_PAGES = 6
CALCULATION_HINT = re.compile(
    r"học phí|tín chỉ|điểm trung bình|lệ phí|đồng/tín chỉ|thang điểm|GPA", re.IGNORECASE
)


@dataclass
class Slot:
    slot_id: str
    category: str
    expected_intent: str
    department_id: str
    file_id: str = ""
    file_name: str = ""
    is_public: str = ""
    access_level: str = ""
    excerpt_path: str = ""


def extract_text(path: Path, pages: int = EXCERPT_PAGES) -> str:
    document = pymupdf.open(path)
    try:
        return "\n".join(document[i].get_text() for i in range(min(pages, document.page_count)))
    finally:
        document.close()


def excerpt(text: str, focus: re.Pattern[str] | None = None) -> str:
    """At most EXCERPT_CHARS of `text`, centred on the first `focus` match if any."""

    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n", text).strip()
    start = 0
    match = focus.search(text) if focus else None
    if match:
        start = max(0, match.start() - EXCERPT_CHARS // 3)
    return text[start : start + EXCERPT_CHARS]


def proportional_quota(sizes: dict[str, int], total: int, minimum: int = 2) -> dict[str, int]:
    """Split `total` across keys in proportion to `sizes`, never above a key's size.

    Largest-remainder rounding; every key gets at least min(minimum, size).
    """

    quota = {key: min(minimum, size) for key, size in sizes.items()}
    remaining = total - sum(quota.values())
    weight = sum(sizes.values())
    if remaining <= 0 or weight == 0:
        return quota
    shares = {key: remaining * size / weight for key, size in sizes.items()}
    for key, share in shares.items():
        quota[key] = min(sizes[key], quota[key] + int(share))
    leftovers = sorted(shares, key=lambda key: shares[key] - int(shares[key]), reverse=True)
    while sum(quota.values()) < total and any(quota[k] < sizes[k] for k in sizes):
        for key in leftovers:
            if sum(quota.values()) >= total:
                break
            if quota[key] < sizes[key]:
                quota[key] += 1
    return quota


def plan(
    rows: list[dict[str, str]], dataset: Path, work_dir: Path, *, max_pages: int, seed: int
) -> list[Slot]:
    for row in rows:
        row["selected"] = str(row.get("quality") == "ok" and int(row["pages"]) <= max_pages).lower()
    pool = [
        row
        for row in rows
        if row["selected"] == "true" and int(row["text_chars"]) >= MIN_TEXT_CHARS
    ]
    rng = random.Random(seed)
    pool.sort(key=lambda row: row["file_id"])
    texts: dict[str, str] = {}

    def text_of(row: dict[str, str]) -> str:
        if row["file_id"] not in texts:
            texts[row["file_id"]] = extract_text(dataset / row["local_path"])
        return texts[row["file_id"]]

    used: set[str] = set()
    slots: list[Slot] = []

    def add(category: str, row: dict[str, str] | None, department: str = "") -> None:
        slot = Slot(
            slot_id=f"{category[:3]}-{sum(s.category == category for s in slots) + 1:03d}",
            category=category,
            expected_intent=EXPECTED_INTENT[category],
            department_id=row["department_id"] if row else department,
        )
        if row is not None:
            used.add(row["file_id"])
            focus = CALCULATION_HINT if category == "calculation" else None
            path = work_dir / "excerpts" / f"{row['file_id']}.txt"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(excerpt(text_of(row), focus), encoding="utf-8")
            slot.file_id, slot.file_name = row["file_id"], row["file_name"]
            slot.is_public, slot.access_level = row["is_public"], row["access_level"]
            slot.excerpt_path = path.relative_to(work_dir).as_posix()
        slots.append(slot)

    # access: private documents, round-robin over levels so 1-4 are all covered.
    private = [row for row in pool if row["is_public"] == "false"]
    by_level: dict[str, list[dict[str, str]]] = {}
    for row in rng.sample(private, len(private)):
        by_level.setdefault(row["access_level"], []).append(row)
    departments_seen: set[str] = set()
    level_cycle = sorted(by_level)
    while sum(s.category == "access" for s in slots) < SLOT_COUNTS["access"] and any(
        by_level.values()
    ):
        for level in level_cycle:
            candidates = by_level[level]
            if (
                not candidates
                or sum(s.category == "access" for s in slots) >= SLOT_COUNTS["access"]
            ):
                continue
            fresh = [r for r in candidates if r["department_id"] not in departments_seen]
            row = (fresh or candidates)[0]
            candidates.remove(row)
            departments_seen.add(row["department_id"])
            add("access", row)

    # calculation: documents whose text mentions fees/credits/grades.
    calc_candidates = [
        row
        for row in rng.sample(pool, len(pool))
        if row["file_id"] not in used and CALCULATION_HINT.search(text_of(row))
    ]
    for row in calc_candidates[: SLOT_COUNTS["calculation"]]:
        add("calculation", row)

    # normal: proportional per department over the remaining pool.
    remaining = [row for row in pool if row["file_id"] not in used]
    by_department: dict[str, list[dict[str, str]]] = {}
    for row in rng.sample(remaining, len(remaining)):
        by_department.setdefault(row["department_id"], []).append(row)
    quota = proportional_quota(
        {dept: len(docs) for dept, docs in by_department.items()}, SLOT_COUNTS["normal"]
    )
    for department in sorted(by_department):
        for row in by_department[department][: quota[department]]:
            add("normal", row)

    # unanswerable: department context only, spread evenly.
    departments = sorted({row["department_id"] for row in pool})
    for i in range(SLOT_COUNTS["unanswerable"]):
        add("unanswerable", None, departments[i % len(departments)])
    for category in ("off_topic", "social"):
        for _ in range(SLOT_COUNTS[category]):
            add(category, None)
    return slots


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--max-pages", type=int, default=100)
    parser.add_argument("--seed", type=int, default=30092026)
    args = parser.parse_args()

    state = DownloadState(args.dataset)
    slots = plan(
        state.manifest, args.dataset, args.work_dir, max_pages=args.max_pages, seed=args.seed
    )
    state.save()  # persists the `selected` column
    args.work_dir.mkdir(parents=True, exist_ok=True)
    with (args.work_dir / "slots.jsonl").open("w", encoding="utf-8") as handle:
        for slot in slots:
            handle.write(json.dumps(asdict(slot), ensure_ascii=False) + "\n")
    selected = sum(row["selected"] == "true" for row in state.manifest)
    print(f"selected for ingest: {selected}")
    print("slots:", dict(Counter(slot.category for slot in slots)))
    print(
        "departments with normal slots:",
        len({s.department_id for s in slots if s.category == "normal"}),
    )
    print(
        "access levels:",
        sorted(Counter(s.access_level for s in slots if s.category == "access").items()),
    )


if __name__ == "__main__":
    main()
