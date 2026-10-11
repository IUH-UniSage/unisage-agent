"""Assemble `questions.jsonl` from planned slots and authored answers.

Inputs (from `<work-dir>`): `slots.jsonl` (plan.py) and `answers/*.jsonl`, one
object per slot: `{"slot_id", "question", "expected_answer", "evidence"}`, or
`{"slot_id", "skip": true, "reason"}` when the excerpt can't support a question.

Checks every grounded answer's `evidence` really occurs in the slot's excerpt
(whitespace-insensitive) and records it as `evidence_found`, so a reviewer can
start with the rows where it's false.

Each `access` slot becomes four rows, one per persona in PERSONAS, with
`expect_visible` following the Qdrant visibility rule; other rows get a single
asker that can see the source (a guest for public documents).

Usage:
    python -m evals.questions.build --work-dir /tmp/qwork \
        --dataset ../unisage-gateway/dataset/official
"""

import argparse
import csv
import json
import random
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any

from evals.label import DEFAULT_MAP, load_label_map

GENERATOR = "claude-opus-5-5 (subagents), reviewed=false until a human checks it"
MAX_LEVEL = 5
WILDCARD = "*"
GROUNDED = {"normal", "calculation", "access"}


def _normalize(text: str) -> str:
    # NFC: some PDFs store "toán" as "toa" + a combining accent.
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", text)).strip().lower()


def _nfc(text: str) -> str:
    return unicodedata.normalize("NFC", text).strip()


def evidence_found(evidence: str, excerpt_text: str) -> bool:
    return bool(evidence.strip()) and _normalize(evidence) in _normalize(excerpt_text)


def is_visible(asker: list[dict[str, Any]], department: str, level: int | None) -> bool:
    """Same rule as `app/rag/vectorstore/qdrant_store.py`."""

    if level is None:
        return True
    return any(
        entry["department_id"] in (department, WILDCARD) and entry["access_level"] >= level
        for entry in asker
    )


def personas(
    department: str, level: int, other_department: str
) -> list[tuple[str, list[dict[str, Any]]]]:
    return [
        ("same_department_exact_level", [{"department_id": department, "access_level": level}]),
        (
            "same_department_one_level_below",
            [{"department_id": department, "access_level": level - 1}],
        ),
        (
            "other_department_max_level",
            [{"department_id": other_department, "access_level": MAX_LEVEL}],
        ),
        ("wildcard_max_level", [{"department_id": WILDCARD, "access_level": MAX_LEVEL}]),
    ]


def build(
    slots: list[dict[str, str]],
    answers: dict[str, dict[str, Any]],
    departments: list[str],
    seed: int,
    work_dir: Path,
) -> tuple[list[dict[str, Any]], Counter[str]]:
    rng = random.Random(seed)
    problems: Counter[str] = Counter()
    rows: list[dict[str, Any]] = []
    for slot in slots:
        answer = answers.get(slot["slot_id"])
        if answer is None:
            problems["missing_answer"] += 1
            continue
        if answer.get("skip"):
            problems["skipped_by_author"] += 1
            continue
        if not answer.get("question") or not answer.get("expected_answer"):
            problems["incomplete_answer"] += 1
            continue
        category = slot["category"]
        grounded = category in GROUNDED
        excerpt_text = (
            (work_dir / slot["excerpt_path"]).read_text(encoding="utf-8") if grounded else ""
        )
        evidence = str(answer.get("evidence", "")) if grounded else ""
        found = evidence_found(evidence, excerpt_text) if grounded else None
        if grounded and not found:
            problems["evidence_not_in_excerpt"] += 1
        level = int(slot["access_level"]) if slot.get("access_level") else None
        base: dict[str, Any] = {
            "slot_id": slot["slot_id"],
            "category": category,
            "question": _nfc(answer["question"]),
            "expected_answer": _nfc(answer["expected_answer"]),
            "evidence": _nfc(evidence),
            "evidence_found": found,
            "expected_doc_ids": [slot["file_id"]] if grounded else [],
            "expected_intent": slot["expected_intent"],
            "department_id": slot["department_id"],
            "doc_is_public": slot.get("is_public") == "true" if grounded else None,
            "doc_access_level": level,
            "reviewed": False,
            "generator": GENERATOR,
        }
        if category == "access" and level is not None:
            others = [d for d in departments if d != slot["department_id"]]
            other = rng.choice(others)
            for persona, asker in personas(slot["department_id"], level, other):
                rows.append(
                    base
                    | {
                        "persona": persona,
                        "asker": {"department_access": asker},
                        "expect_visible": is_visible(asker, slot["department_id"], level),
                    }
                )
            continue
        if grounded and level is not None:
            asker = [{"department_id": slot["department_id"], "access_level": level}]
            persona = "same_department_exact_level"
        else:
            asker, persona = [], "guest"
        rows.append(
            base
            | {
                "persona": persona,
                "asker": {"department_access": asker},
                "expect_visible": True if grounded else None,
            }
        )
    for index, row in enumerate(rows, start=1):
        row["id"] = f"q{index:04d}"
    return [{"id": row.pop("id"), **row} for row in rows], problems


REVIEW_FIELDS = [
    "id", "category", "department_id", "persona", "expect_visible", "question",
    "expected_answer", "evidence", "evidence_found", "source_file", "source_url",
    "review_note", "reviewed", "reviewer_comment",
]  # fmt: skip


def write_review_sheet(
    path: Path,
    rows: list[dict[str, Any]],
    manifest: dict[str, dict[str, str]],
    notes: dict[str, str],
) -> None:
    """Spreadsheet view for the human review (Task 9): rows with a note first."""

    ordered = sorted(rows, key=lambda row: (row["slot_id"] not in notes, row["id"]))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REVIEW_FIELDS, lineterminator="\n")
        writer.writeheader()
        for row in ordered:
            source = manifest.get(row["expected_doc_ids"][0], {}) if row["expected_doc_ids"] else {}
            writer.writerow(
                {
                    **{field: row.get(field, "") for field in REVIEW_FIELDS},
                    "source_file": source.get("file_name", ""),
                    "source_url": source.get("source_url", ""),
                    "review_note": notes.get(row["slot_id"], ""),
                    "reviewed": "",
                    "reviewer_comment": "",
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=30092026)
    parser.add_argument(
        "--notes", type=Path, help="JSON {slot_id: note} flagging rows to review first"
    )
    args = parser.parse_args()

    slots = [json.loads(line) for line in (args.work_dir / "slots.jsonl").open(encoding="utf-8")]
    answers: dict[str, dict[str, Any]] = {}
    for path in sorted((args.work_dir / "answers").glob("*.jsonl")):
        for line in path.open(encoding="utf-8"):
            if line.strip():
                item = json.loads(line)
                answers[item["slot_id"]] = item
    departments = sorted({rule.department for rule in load_label_map(DEFAULT_MAP).units.values()})
    rows, problems = build(slots, answers, departments, args.seed, args.work_dir)

    out = args.dataset / "questions.jsonl"
    with out.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    notes: dict[str, str] = json.loads(args.notes.read_text(encoding="utf-8")) if args.notes else {}
    with (args.dataset / "manifest.csv").open(newline="", encoding="utf-8") as handle:
        manifest = {row["file_id"]: row for row in csv.DictReader(handle)}
    write_review_sheet(args.dataset / "questions_review.csv", rows, manifest, notes)
    print(f"wrote {len(rows)} rows to {out} (+ questions_review.csv)")
    print("by category:", dict(Counter(row["category"] for row in rows)))
    print("problems:", dict(problems))


if __name__ == "__main__":
    main()
